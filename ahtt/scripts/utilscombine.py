#!/usr/bin/env python
# utilities containing functions used throughout - combine file

import glob
import os
import sys
import re
import fileinput
import shutil

from desalinator import remove_quotes, remove_spaces, tokenize_to_list, clamp_with_quote
from utilspy import syscall, right_now

from ROOT import TFile, gDirectory, TH1, TH1D
TH1.AddDirectory(False)
TH1.SetDefaultSumw2(True)

min_g = 0.
max_g = 3.

def problematic_datacard_log(logfile):
    if not hasattr(problematic_datacard_log, "problems"):
        problematic_datacard_log.problems = [
            r"'up/down templates vary the yield in the same direction'",
            r"'up/down templates are identical'",
            r"'At least one of the up/down systematic uncertainty templates is empty'",
            r"'Empty process'",
            r"'Bins of the template empty in background'",
        ]

    with open(logfile) as lf:
        for line in lf:
            for problem in problematic_datacard_log.problems:
                if problem in line and 'no warnings' not in line:
                    lf.close()
                    return True
        lf.close()
    return False

def list_of_processes(datacard, hint = "TT"):
    with open(datacard) as dc:
        for line in dc:
            processes = tokenize_to_list(line, token = " ")
            if "process" in processes[0]:
                processes = [remove_spaces(it) for it in processes[1:]]
                processes = [it for it in processes if it != "" and it != "\n"]
                if hint in processes:
                    break
    return sorted(list(set(processes)))

def list_of_channels(datacard):
    with open(datacard) as dc:
        for line in dc:
            nchannel = tokenize_to_list(line, token = " ")
            if "imax" in nchannel[0] and "bins" in nchannel[-1]:
                nchannel = int(nchannel[1])
                break

        for line in dc:
            channels = tokenize_to_list(line, token = " ")
            if "bin" in channels[0]:
                channels = [remove_spaces(it) for it in channels[1:]]
                channels = [it for it in channels if it != "" and it != "\n"]
                channels = sorted(list(set(channels)))
                if len(channels) == nchannel:
                    break
    return channels

def set_range(parameters):
    '''
    given a list of parameters, return the set of combine setting their ranges
    parameter is a string in the following syntax: name[: min, max]
    where the range in the square bracket is optional
    when the range is not given, it defaults to -20, 20
    '''
    parameters = ['='.join(remove_spaces(param).split(':')) if ':' in param else param + '=-20,20' for param in parameters]
    return "--setParameterRanges '{ranges}'".format(ranges = ':'.join(parameters))

def set_parameter(set_freeze, extopt, masks):
    '''
    problem is, setparameters and freezeparameters may appear only once
    so --extra-option is not usable to study shifting them up if we set g etc
    this method harmonizes the set/freezeParameter options and ignore all others
    '''
    setpar = list(set_freeze[0])
    frzpar = [par for par in set_freeze[1] if "__grp__" not in par]
    grppar = [par.replace("__grp__", "") for par in set_freeze[1] if "__grp__" in par]

    extopt = [] if extopt == "" else extopt.split(' ')
    for option in ['--setParameters', '--freezeParameters', '--freezeNuisanceGroups']:
        while option in extopt:
            iopt = extopt.index(option)
            parameters = tokenize_to_list(remove_quotes(extopt.pop(iopt + 1))) if iopt + 1 < len(extopt) else []
            extopt.pop(iopt)
            if option == '--setParameters':
                setpar += parameters
            elif option == '--freezeParameters':
                frzpar += parameters
            elif option == '--freezeNuisanceGroups':
                grppar += parameters

    return '{stp} {frz} {grp}'.format(
        stp = "--setParameters '" + ",".join(setpar + masks) + "'" if len(setpar + masks) > 0 else "",
        frz = "--freezeParameters '" + ",".join(frzpar) + "'" if len(frzpar) > 0 else "",
        grp = "--freezeNuisanceGroups '" + ",".join(grppar) + "'" if len(grppar) > 0 else ""
    )

def nonparametric_option(extopt):
    '''
    removes the parametric part of extopt, and returns the rest as one option string to be passed to combine
    '''
    if extopt == "":
        return ""

    extopt = extopt.split(' ')
    for option in ['--setParameters', '--freezeParameters', "--freezeNuisanceGroups"]:
        while option in extopt:
            iopt = extopt.index(option)
            parameters = tokenize_to_list(remove_quotes(extopt.pop(iopt + 1))) if iopt + 1 < len(extopt) else []
            extopt.pop(iopt)

    return " ".join(extopt)

def get_fit(dname, attributes, qexp_eq_m1 = True):
    if not os.path.isfile(dname):
        return None

    dfile = TFile.Open(dname)
    dtree = dfile.Get("limit")

    bf = None
    for i in dtree:
        if (dtree.quantileExpected == -1. and qexp_eq_m1) or (dtree.quantileExpected != -1. and not qexp_eq_m1):
            bf = [getattr(dtree, attr) for attr in attributes]
            if 'deltaNLL' in attributes:
                idx = attributes.index('deltaNLL')
                bf[idx] = max(bf[idx], 0.)
            bf = tuple(bf)

        if bf is not None:
            break

    dfile.Close()
    return bf

def get_best_fit(dcdir, point, tags, usedefault, useexisting, default, asimov, runmode,
                 modifier, scenario, poiset, ranges, set_freeze, extopt = "", masks = []):
    ptag = lambda pnt, tag: "{pnt}{tag}".format(pnt = point, tag = tag)

    if usedefault:
        return default
    elif useexisting:
        workspace = glob.glob("{dcd}{ptg}_best-fit_{asm}*{mod}.root".format(
            dcd = dcdir,
            ptg = ptag(point, tags[0]),
            asm = "exp" if asimov else "obs",
            mod = "_" + modifier if modifier != "" else "",
        ))

        if len(workspace) == 0 or not os.path.isfile(workspace[0]):
            # try again, but using tag instead of otag
            workspace = glob.glob("{dcd}{ptg}_best-fit_{asm}*{mod}.root".format(
                dcd = dcdir,
                ptg = ptag(point, tags[1]),
                asm = "exp" if asimov else "obs",
                mod = "_" + modifier if modifier != "" else "",
            ))

        if len(workspace) and os.path.isfile(workspace[0]):
            return workspace[0]
        else:
            useexisting = False

    if not usedefault and not useexisting:
        # ok there really isnt a best fit file, make them
        print ("\nxxx_point_ahtt :: making best fits")
        for asm in [not asimov, asimov]:
            workspace = make_best_fit(dcdir, default, point, asm, poiset, ranges, set_freeze, extopt, masks)
            syscall("rm robustHesse_*.root", False, True)

            newname = "{dcd}{ptg}_{rnm}_{asm}{sce}{mod}.root".format(
                dcd = dcdir,
                ptg = ptag(point, tags[0]),
                rnm = "single" if "single" in runmode else "best-fit",
                asm = "exp" if asm else "obs",
                sce = "_" + scenario if scenario != "" else "",
                mod = "_" + modifier if modifier != "" else "",
            )
            syscall("mv {wsp} {nwn}".format(wsp = workspace, nwn = newname), False)
            workspace = newname

    nll = get_fit(workspace, ["nll"])
    print ("\nxxx_point_ahtt :: the dNLL of the best fit point wrt the model zero point (0, ...) is {nll}".format(poi = ', '.join(poiset), nll = nll))
    print ("WARNING :: the model zero point is based on the 'nll0' branch, which includes the values of ALL NPs, not only POIs!!")
    print ("WARNING :: this means no NP profiling is done, so do NOT use this value directly for compatibility tests!!")
    print ("\n")
    sys.stdout.flush()
    return workspace

def starting_nuisance(freeze_zero, freeze_post):
    set_freeze = [[], []]
    setp, frzp = set_freeze

    for frz in freeze_zero | freeze_post:
        if frz in ["autoMCStats", "mcstat"]:
            param = r"rgx{prop_bin.*}"
        elif frz in ["experiment", "theory", "norm", "expth"]:
            param = "__grp__{frz}".format(frz = frz)
        else:
            param = frz

        # --setNuisanceGroups xxx=0 ain't a thing
        if frz in freeze_zero and "__grp__" not in param:
            setp.append("{param}=0".format(param = param))
        frzp.append("{param}".format(param = param))

    return set_freeze

def fit_strategy(strategy, optimize = True, robust = False, use_hesse = False, tolerance = 0):
    fstr = "--X-rtd OPTIMIZE_BOUNDS=0 --X-rtd MINIMIZER_MaxCalls=9999999"
    if optimize:
        fstr += " --X-rtd FAST_VERTICAL_MORPH --X-rtd CACHINGPDF_NOCLONE"
    fstr += " --cminPreScan --cminDefaultMinimizerAlgo Combined --cminDefaultMinimizerStrategy {ss}".format(ss = strategy)
    fstr += " --cminDefaultMinimizerTolerance {tol}".format(tol = 2.**(tolerance - 4))
    for algo in ["Minuit2,Migrad", "Minuit2,Simplex", "GSLMultiMin,BFGS2"]:
        fstr += " --cminFallbackAlgo {aa},{ss}:{tol}".format(aa = algo, ss = strategy, tol = 2.**(tolerance - 4))

    if robust:
        fstr += " --robustFit 1 --setRobustFitAlgo Minuit2 --maxFailedSteps 9999999 --setRobustFitStrategy {ss} {t0} {t1} {t2} {hh}".format(
            ss = strategy,
            t0 = "--setRobustFitTolerance {tol}".format(tol = 2.**(tolerance - 4)),
            t1 = "--stepSize {tol}".format(tol = 2.**(tolerance - 4)),
            t2 = "--setCrossingTolerance {tol}".format(tol = 2.**(tolerance - 12)),
            hh = "--robustHesse 1" if use_hesse else ""
        )
    return fstr

def is_good_fit(fit_fname, fit_names):
    '''
    checks if fit is good
    fit_fname is the filename of the file containing buncha RooFitResults to be checked
    fit_names is an iterable containing the names of said RooFitResults
    returns True if all fit_names are good, otherwise False
    '''
    ffile = TFile.Open(fit_fname, "read")
    fgood = []
    for fname in fit_names:
        fresult = ffile.Get("{fname}".format(fname = fname))
        fit_quality = fit_result.covQual()
        print ("\nxxx_point_ahtt :: fit with name {fname} has a covariance matrix of status {fql}".format(fname = fname, fql = fit_quality))
        sys.stdout.flush()
        fgood.append(fit_quality != 3)
    ffile.Close()

    all_good = all(fgood)
    if not all_good:
        syscall("rm {ffn}".format(ffn = fit_fname), False, True)
        print ("xxx_point_ahtt :: one of the matrices is bad.")
        sys.stdout.flush()

    return all_good

def never_gonna_give_you_up(command, optimize = True, followups = [], fit_result_names = None, post_conditions = [], failure_cleanups = [],
                            usehesse = False, robustness = [True, False], strategies = list(range(3)), tolerances = list(range(2)),
                            all_strategies = None, throw_upon_failure = True, first_fit_strategy = -1):
    '''
    run fits with multiple settings until one works
    command: command to run. should contain a {fit_strategy} in there to be substituted in

    optimize is just an extra flag for fit strategy, due to prepost needing it to be False, and elsewhere True

    followups: a list of lists where each inner list is the function to run
    and arguments to be passed to it

    fit_result_names: a list where first element is the name of the file containing the fit result,
    and the second the list of fit result names to run is_good_fit() on. pass None to skip running it

    post_condition: like followups, but said function should return a truthy value

    fit is accepted if is_good_fit and all of post_conditions are true

    failure_cleanups has the same syntax as post_conditions, but the functions do not need to return a truthy value
    they are steps to be run when a fit is not accepted

    the function throws if no fit is accepted

    the rest of the args are bunch of fit settings to try
    '''
    if first_fit_strategy > 0:
        istrat = strategies.index(first_fit_strategy)
        strategies[0], strategies[istrat] = strategies[istrat], strategies[0]
    all_strategies = [(irobust, istrat, itol) for irobust in robustness for istrat in strategies for itol in tolerances] if all_strategies is None else all_strategies
    for irobust, istrat, itol in all_strategies:
        if usehesse and not irobust:
            continue
        robusthesse = irobust and usehesse

        syscall(command.replace(
            '{fit_strategy}',
            fit_strategy(strategy = istrat, robust = irobust, use_hesse = robusthesse, tolerance = itol, optimize = optimize)
        ))

        for fu in followups:
            fu[0](*fu[1:])

        fgood = True if fit_result_names is None or robusthesse else is_good_fit(*fit_result_names)
        pgood = all([pc[0](*pc[1:]) for pc in post_conditions])

        if robusthesse:
            syscall("rm robustHesse_*.root", False, True)

        if fgood and pgood:
            return True
        else:
            for fc in failure_cleanups:
                fc[0](*fc[1:])

    print ("\nnever_gonna_give_you_up :: no accepted fit found. argument and state variables:")
    print (locals())
    print ("\n\n")
    sys.stdout.flush()
    if throw_upon_failure:
        raise RuntimeError("never_gonna_give_you_up :: unfortunately, with this set, the function has to give up...")
    else:
        return False

def make_best_fit(dcdir, workspace, point, asimov, poiset, ranges, set_freeze, extopt = "", masks = []):
    fname = point + "_best_fit_" + right_now()
    never_gonna_give_you_up(
        command = "combineTool.py -v 0 -M MultiDimFit -d {dcd} -n _{bff} {stg} {prg} {asm} {poi} {wsp} {prm} {ext}".format(
            dcd = workspace,
            bff = fname,
            stg = "{fit_strategy}",
            prg = ranges,
            asm = "-t -1" if asimov else "",
            poi = "--redefineSignalPOIs '{poi}'".format(poi = ','.join(poiset)),
            wsp = "--saveWorkspace --saveSpecifiedNuis=all --saveNLL",
            prm = set_parameter(set_freeze, extopt, masks),
            ext = nonparametric_option(extopt)
        ),

        failure_cleanups = [
            [syscall, "rm higgsCombine*{bff}.MultiDimFit*.root".format(bff = fname), False]
        ]
    )
    syscall("mv higgsCombine*{bff}.MultiDimFit*.root {dcd}{bff}.root".format(dcd = dcdir, bff = fname), False)
    return "{dcd}{bff}.root".format(dcd = dcdir, bff = fname)

def make_datacard_with_args(scriptdir, args):
    syscall("{scr}/make_datacard.py --signal {sig} --background {bkg} --point {pnt} --channel {ch} --year {yr} "
            "{psd} {inj} {ass} {exc} {tag} {drp} {kee} {kfc} {thr} {lns} {shp} {mcs} {rpr} "
            "{igb} {prj} {cho} {rep} {arn} {rsd}".format(
                scr = scriptdir,
                pnt = ','.join(args.point),
                sig = args.signal,
                bkg = args.background,
                ch = args.channel,
                yr = args.year,
                psd = "--add-pseudodata" if args.asimov else "",
                inj = clamp_with_quote(string = args.inject, prefix = '--inject-signal '),
                ass = clamp_with_quote(string = args.assignal, prefix = '--as-signal '),
                exc = clamp_with_quote(string = args.excludeproc, prefix = '--exclude-process '),
                tag = clamp_with_quote(string = args.tag, prefix = '--tag '),
                drp = clamp_with_quote(string = args.drop, prefix = '--drop '),
                kee = clamp_with_quote(string = args.keep, prefix = '--keep '),
                kfc = "--sushi-kfactor" if args.kfactor else "",
                thr = clamp_with_quote(string = args.threshold, prefix = '--threshold '),
                lns = "--lnN-under-threshold" if args.lnNsmall else "",
                shp = "--use-shape-always" if args.alwaysshape else "",
                mcs = "--no-mc-stats" if not args.mcstat else "",
                rpr = clamp_with_quote(string = args.rateparam, prefix = '--float-rate '),
                msk = clamp_with_quote(string = ','.join(args.mask), prefix = '--mask '),
                igb = clamp_with_quote(string = args.ignorebin, prefix = '--ignore-bin '),
                prj = clamp_with_quote(string = args.projection, prefix = '--projection '),
                cho = clamp_with_quote(string = args.chop, prefix = '--chop-up '),
                rep = clamp_with_quote(string = args.repnom, prefix = '--replace-nominal '),
                arn = clamp_with_quote(string = args.arbnorm, prefix = '--arbitrary-resonance-normalization '),
                rsd = clamp_with_quote(string = str(args.seed), prefix = '--seed '),
            ))

def update_mask(masks):
    new_masks = []
    for mask in masks:
        channel, year = mask.split("_")
        if channel == "lx":
            channels = ["ee", "em", "mm", "e3j", "e4pj", "m3j", "m4pj"]
        elif channel == "ll":
            channels = ["ee", "em", "mm"]
        elif channel == "sf":
            channels = ["ee", "mm"]
        elif channel == "lj":
            channels = ["e3j", "e4pj", "m3j", "m4pj"]
        elif channel == "l3j":
            channels = ["e3j", "m3j"]
        elif channel == "l4pj":
            channels = ["e4pj", "m4pj"]
        elif channel == "ej":
            channels = ["e3j", "e4pj"]
        elif channel == "mj":
            channels = ["m3j", "m4pj"]
        else:
            channels = [channel]

        if year == "all" or year == "run2":
            years = ["2016pre", "2016post", "2017", "2018"]
        elif year == "2016":
            years = ["2016pre", "2016post"]
        else:
            years = [year]

        for cc in channels:
            for yy in years:
                new_masks.append(cc + "_" + yy)

    return sorted(list(set(new_masks)))

def expected_scenario(exp, gvalues_syntax = False, resonly = False):
    specials = {
        "exp-b":  "g1=0,g2=0",
        "exp-s":  "g1=1,g2=1",
        "exp-01": "g1=0,g2=1",
        "exp-10": "g1=1,g2=0",
        "obs":    ""
    }
    tag = None

    if exp in specials:
        tag = exp
        parameters = [str(float(ss)) for ss in tokenize_to_list(specials[exp].replace("g1=", "").replace("g2=", ""))] if gvalues_syntax else specials[exp]

    if not re.search(r'[eo][xb].*', exp):
        gvalues = tokenize_to_list(exp)
        if len(gvalues) != 2 or not all([float(gg) >= 0. for gg in gvalues]):
            return None
        g1, g2 = gvalues
        tag = "exp-{g1}-{g2}".format(g1 = round(float(g1), 5), g2 = round(float(g2), 5))
        parameters = [str(float(ss)) for ss in tokenize_to_list("{g1},{g2}".format(g1 = g1, g2 = g2))] if gvalues_syntax else "g1={g1},g2={g2}".format(g1 = g1, g2 = g2)

    if tag is not None and resonly:
        parameters = [pp.replace("g1", "r1").replace("g2", "r2") for pp in parameters]

    return (tag, parameters) if tag is not None else None

def channel_compatibility_hackery(datacard, extopt):
    # channel compatibility test, as is currently implemented in the version we use, is far too slow
    # so we go the hacky way of adding the params by hand
    # currently assumes that this is only of interest for 2D A/H, etat, yukawa
    if "_combined.txt" not in datacard:
        print ("method makes no sense if not for combined datacard, aborting")
        return
    processes = list_of_processes(datacard)
    channels = list_of_channels(datacard)

    shutil.copyfile(datacard, datacard.replace("_combined.txt", "_chancomp.txt"))
    datacard = datacard.replace("_combined.txt", "_chancomp.txt")

    # renumber the indices to be all background
    for line in fileinput.input(datacard, inplace = True):
        indices = tokenize_to_list(line, token = " ")
        if "process" in line and "TT" not in line:
            for ii in range(indices):
                try:
                    idx = int(indices[ii])
                    indices[ii] = " " + str(idx + 6) if idx < 0 else str(iproc + 6)
                except:
                    continue
            print('{} {}'.format(fileinput.filelineno(), " ".join(indices)), end = '')

    # delete irrelevant/to be redone lines
    syscall("'/group =/d' {dc}".format(dc = datacard))
    syscall("'/EWK_yukawa/d' {dc}".format(dc = datacard))
    syscall("'/CMS_EtaT_norm_13TeV/d' {dc}".format(dc = datacard))

    # process tags
    ahres = [proc for proc in processes if (proc.startswith("A") or proc.startswith("H")) and proc.endswith("_res")]
    ahint = len([proc for proc in processes if (proc.startswith("A") or proc.startswith("H")) and (proc.endswith("_pos") or proc.endswith("_neg"))]) > 0
    ewktt = len([proc for proc in processes if proc.startswith("EWK_TT")]) > 0
    etat = len([proc for proc in processes if proc == "EtaT"]) > 0

    if len(ahres):
        for idx in range(2):
            iah = idx + 1
            pah = ahres[idx].replace("_res", "")

            if ahint:
                txt.write("\ng{iah}_global extArg 1 [0,5]".format(iah = iah))
                for cc in channels:
                    txt.write("\ng{iah}_{cc} extArg 1 [0,5]".format(iah = iah, cc = cc))
                txt.write("\n")
                for cc in channels:
                    txt.write("\ng4{iah}_{cc}_product rateParam {cc} {pah} (@0*@0*@0*@0*@1*@1*@1*@1) g{iah}_global g{iah}_{cc}".format(iah = iah, cc = cc, pah = pah + "_res"))
                    txt.write("\ng2{iah}_{cc}_product rateParam {cc} {pah} (@0*@0*@1*@1) g{iah}_global g{iah}_{cc}".format(iah = iah, cc = cc, pah = pah + "_pos"))
                    txt.write("\nmg2{iah}_{cc}_product rateParam {cc} {pah} (-@0*@0*@1*@1) g{iah}_global g{iah}_{cc}".format(iah = iah, cc = cc, pah = pah + "_neg"))
                txt.write("\n")
            else:
                with open(datacard, 'a') as txt:
                    txt.write("\nr{iah}_global extArg 1 [-5,5]".format(iah = iah))
                    for cc in channels:
                        txt.write("\nr{iah}_{cc} extArg 1 [-5,5]".format(iah = iah, cc = cc))
                    txt.write("\n")
                    for cc in channels:
                        txt.write("\nr{iah}_{cc}_product rateParam {cc} {pah} (@0*@1) r{iah}_global r{iah}_{cc}".format(iah = iah, cc = cc, pah = pah + "_res"))
                    txt.write("\n")

    if ewktt:
        with open(datacard, 'a') as txt:
            txt.write("\nEWK_yukawa_global param 1 -0.12/+0.11")
            for cc in channels:
                txt.write("\nEWK_yukawa_{cc} param 1 -0.12/+0.11".format(cc = cc))
            txt.write("\n")
            for cc in channels:
                txt.write("\nEWK_yukawa_{cc}_product rateParam {cc} EWK_TT_lin_pos (@0*@1) EWK_yukawa_global EWK_yukawa_{cc}".format(cc = cc))
                txt.write("\nmEWK_yukawa_{cc}_product rateParam {cc} EWK_TT_lin_neg (-@0*@1) EWK_yukawa_global EWK_yukawa_{cc}".format(cc = cc))
                txt.write("\nEWK_yukawa2_{cc}_product rateParam {cc} EWK_TT_quad_pos (@0*@0*@1*@1) EWK_yukawa_global EWK_yukawa_{cc}".format(cc = cc))
                txt.write("\nmEWK_yukawa2_{cc}_product rateParam {cc} EWK_TT_quad_neg (-@0*@0*@1*@1) EWK_yukawa_global EWK_yukawa_{cc}".format(cc = cc))
            txt.write("\n")

    if etat:
        with open(datacard, 'a') as txt:
            txt.write("\CMS_EtaT_norm_13TeV_global extArg 1 [-5,5]")
            for cc in channels:
                txt.write("\nCMS_EtaT_norm_13TeV_{cc} extArg 1 [-5,5]".format(cc = cc))
            txt.write("\n")
            for cc in channels:
                txt.write("\nCMS_EtaT_norm_13TeV_{cc}_product rateParam {cc} EtaT (@0*@1) CMS_EtaT_norm_13TeV_global CMS_EtaT_norm_13TeV_{cc}".format(cc = cc)
            txt.write("\n")

    syscall("combineTool.py -v 0 -M T2W -i {dcd} -o workspace_{wst}.root -m {mmm} {opt} {whs} {ext}".format(
        dcd = datacard,
        wst = "chancomp",
        mmm = mstr,
        opt = "--channel-masks",
        whs = "--X-pack-asympows --optimize-simpdf-constraints=cms --no-wrappers --use-histsum" if ihsum else "",
        ext = extopt
    ))
