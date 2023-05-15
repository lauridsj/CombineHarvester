#!/usr/bin/env python
# utilities containing functions used throughout - combine file

from desalinator import remove_quotes, tokenize_to_list
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

def set_parameter(set_freeze, extopt, masks):
    '''
    problem is, setparameters and freezeparameters may appear only once
    so --extra-option is not usable to study shifting them up if we set g etc
    this method harmonizes the set/freezeParameter options and ignore all others
    '''
    setpar = list(set_freeze[0])
    frzpar = list(set_freeze[1])

    extopt = [] if extopt == "" else extopt.split(' ')
    for option in ['--setParameters', '--freezeParameters']:
        while option in extopt:
            iopt = extopt.index(option)
            parameters = tokenize_to_list(remove_quotes(extopt.pop(iopt + 1))) if iopt + 1 < len(extopt) else []
            extopt.pop(iopt)
            if option == '--setParameters':
                setpar += parameters
            elif option == '--freezeParameters':
                frzpar += parameters

    return '{stp} {frz} {ext}'.format(
        stp = "--setParameters '" + ",".join(setpar + masks) + "'" if len(setpar + masks) > 0 else "",
        frz = "--freezeParameters '" + ",".join(frzpar) + "'" if len(frzpar) > 0 else "",
        ext = ' '.join(extopt)
    )

def nonparametric_option(extopt):
    '''
    removes the parametric part of extopt, and returns the rest as one option string to be passed to combine
    '''
    if extopt == "":
        return ""

    extopt = extopt.split(' ')
    for option in ['--setParameters', '--freezeParameters']:
        while option in extopt:
            iopt = extopt.index(option)
            parameters = tokenize_to_list(remove_quotes(extopt.pop(iopt + 1))) if iopt + 1 < len(extopt) else []
            extopt.pop(iopt)

    return " ".join(extopt)

def make_best_fit(dcdir, card, point, asimov, strategy, poi_range, set_freeze, extopt = "", masks = []):
    fname = point + "_best_fit_" + right_now()

    syscall("combineTool.py -v -1 -M MultiDimFit -d {dcd} -n _{bff} {stg} {prg} {asm} {wsp} {prm} {ext}".format(
        dcd = dcdir + card,
        bff = fname,
        stg = strategy,
        prg = poi_range,
        asm = "-t -1" if asimov else "",
        wsp = "--saveWorkspace --saveSpecifiedNuis=all",
        prm = set_parameter(set_freeze, extopt, masks),
        ext = nonparametric_option(extopt)
    ))
    syscall("mv higgsCombine*{bff}.MultiDimFit*.root {dcd}{bff}.root".format(dcd = dcdir, bff = fname), False)
    return "{dcd}{bff}.root".format(dcd = dcdir, bff = fname)

def read_nuisance(dname, points, qexp_eq_m1 = True):
    dfile = TFile.Open(dname)
    dtree = dfile.Get("limit")

    skip = ["r", "g", "r1", "r2", "g1", "g2", "deltaNLL", "quantileExpected",
            "limit", "limitErr", "mh", "syst",
            "iToy", "iSeed", "iChannel", "t_cpu", "t_real"]

    nuisances = [bb.GetName() for bb in dtree.GetListOfBranches()]
    hasbb = False

    setpar = []
    frzpar = []

    for i in dtree:
        if (dtree.quantileExpected != -1. and qexp_eq_m1) or (dtree.quantileExpected == -1. and not qexp_eq_m1):
            continue

        for nn in nuisances:
            if nn in skip:
                continue

            if "prop_bin" not in nn:
                frzpar.append(nn)
            else:
                hasbb = True

            vv = round(getattr(dtree, nn), 2)
            if abs(vv) > 0.01:
                setpar.append(nn + "=" + str(vv))

        if len(frzpar) > 0:
            break

    if hasbb:
        frzpar.append("rgx{prop_bin.*}")

    return [setpar, frzpar]

def starting_nuisance(point, frz_bb_zero = True, frz_bb_post = False, frz_nuisance_post = False, best_fit_file = ""):
    if frz_bb_zero:
        return [["rgx{prop_bin.*}=0"], ["rgx{prop_bin.*}"]]
    elif frz_bb_post or frz_nuisance_post:
        if best_fit_file == "":
            raise RuntimeError("postfit bb/nuisance freezing is requested, but no best fit file is provided!!!")

        setpar, frzpar = read_nuisance(best_fit_file, point, True)

        if not frz_nuisance_post:
            setpar = [nn for nn in setpar if "prop_bin" in nn]
            frzpar = [nn for nn in frzpar if "prop_bin" in nn]

        return [setpar, frzpar]

    return [[], []]

def fit_strategy(strat, robust = False, tolerance_level = 0):
    fstr = "--X-rtd NO_INITIAL_SNAP --X-rtd FAST_VERTICAL_MORPH --X-rtd CACHINGPDF_NOCLONE --X-rtd MINIMIZER_MaxCalls=9999999"
    fstr += " --cminPreScan --cminDefaultMinimizerAlgo Migrad --cminDefaultMinimizerStrategy {ss} --cminFallbackAlgo Minuit2,Simplex,{ss}".format(ss = strat)

    if tolerance_level > 0:
        fstr += ":{tolerance} --cminDefaultMinimizerTolerance {tolerance} ".format(tolerance = tolerance_level * 0.5)
    if robust:
        fstr += " --robustFit 1 --setRobustFitStrategy {ss} {tt}".format(
            ss = strat,
            tt = "--setRobustFitTolerance {tolerance} --setCrossingTolerance {tolerance}e-3".format(tolerance = tolerance_level * 0.5) if tolerance_level > 0 else ""
        )
    return fstr

def make_datacard_with_args(scriptdir, args):
    syscall("{scr}/make_datacard.py --signal {sig} --background {bkg} --point {pnt} --channel {ch} --year {yr} "
            "{psd} {inj} {tag} {drp} {kee} {kfc} {thr} {lns} {shp} {mcs} {rpr} {prj} {cho} {rep} {rsd}".format(
                scr = scriptdir,
                pnt = ','.join(args.point),
                sig = args.signal,
                bkg = args.background,
                ch = args.channel,
                yr = args.year,
                psd = "--add-pseudodata" if args.asimov else "",
                inj = "--inject-signal " + args.inject if args.inject != "" else "",
                tag = "--tag " + args.tag if args.tag != "" else "",
                drp = "--drop '" + args.drop + "'" if args.drop != "" else "",
                kee = "--keep '" + args.keep + "'" if args.keep != "" else "",
                kfc = "--sushi-kfactor" if args.kfactor else "",
                thr = "--threshold " + args.threshold if args.threshold != "" else "",
                lns = "--lnN-under-threshold" if args.lnNsmall else "",
                shp = "--use-shape-always" if args.alwaysshape else "",
                mcs = "--no-mc-stats" if not args.mcstat else "",
                rpr = "--float-rate '" + args.rateparam + "'" if args.rateparam != "" else "",
                prj = "--projection '" + args.projection + "'" if args.projection != "" else "",
                cho = "--chop-up '" + args.chop + "'" if args.chop != "" else "",
                rep = "--replace-nominal '" + args.repnom + "'" if args.repnom != "" else "",
                rsd = "--seed " + args.seed if args.seed != "" else ""
            ))

def update_mask(masks):
    new_masks = []
    for mask in masks:
        channel, year = mask.split("_")
        if channel == "ll":
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

    return list(set(new_masks))
