#!/usr/bin/env python
# sets the model-independent limit on two A/H signal points

from argparse import ArgumentParser
import os
import sys
import glob
import re
import numpy as np
import itertools

from collections import OrderedDict
import json

from ROOT import TFile, TTree
from array import array

from utilspy import syscall, chunks, elementwise_add, recursive_glob, make_timestamp_dir, directory_to_delete, max_nfile_per_dir
from utilspy import get_point, tuplize, stringify, g_in_filename, floattopm
from utilscombine import max_g, get_best_fit, starting_nuisance, fit_strategy, make_datacard_with_args, set_range, set_parameter, nonparametric_option, update_mask
from utilscombine import list_of_processes, get_fit, is_good_fit, never_gonna_give_you_up, expected_scenario
from utilscombine import channel_compatibility_hackery, channel_compatibility_masking

from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes, remove_quotes
from argumentative import common_point, common_common, common_fit_pure, common_fit, make_datacard_pure, make_datacard_forwarded, common_2D, parse_args
from hilfemir import combine_help_messages

def read_previous_best_fit(gname):
    with open(gname) as ff:
        result = json.load(ff, object_pairs_hook = OrderedDict)
    return tuple(result["best_fit_g1_g2_dnll"])

def read_previous_grid(points, prev_best_fit, gname, epsilon = 2.**-4):
    with open(gname) as ff:
        result = json.load(ff, object_pairs_hook = OrderedDict)

    if result["points"] == points and all([abs(now - prev) < epsilon for now, prev in zip(result["best_fit_g1_g2_dnll"], list(prev_best_fit))]):
        return result["g-grid"]
    else:
        print "\ninconsistent previous grid, ignoring the previous grid..."
        print "previous: ", points, prev_best_fit
        print "current: ", result["points"], result["best_fit_g1_g2_dnll"]
        sys.stdout.flush()

    return OrderedDict()

def points_to_compile(points, expfits, gname):
    gpoints = []
    for ef in expfits:
        gs = ef.split("_")
        g1 = float(gs[ gs.index("g1") + 1 ])
        g2 = float(gs[ gs.index("g2") + 1 ])
        gpoints.append((g1, g2))

    if gname is not None:
        with open(gname) as ff:
            result = json.load(ff, object_pairs_hook = OrderedDict)
        if result["points"] == points:
            for gv in result["g-grid"]:
                gpoints.append(tuplize(gv))
    return sorted(list(set(gpoints)))

def get_toys(toy_name, best_fit, keep_reduced = False, strict_preservation = False, whatever_else = None):
    if not os.path.isfile(toy_name):
        return None

    vname = toy_name.replace("toys.root", "reduced.root")
    syscall("rooteventselector --recreate -e '*' -i '{branches}' -s 'quantileExpected >= 0 && deltaNLL >= 0' {tn}:limit {vn}".format(
        branches = "deltaNLL,quantileExpected",
        tn = toy_name,
        vn = vname
    ), False)

    pval = OrderedDict()
    vfile = TFile.Open(vname)
    vtree = vfile.Get("limit")

    isum = 0
    ipas = 0
    for i in vtree:
        isum += 1
        ipas += 1 if vtree.deltaNLL > best_fit[2] else 0
    vfile.Close()
    if not strict_preservation:
        syscall("cp {vn} {tn}".format(vn = vname, tn = toy_name), False)
    if not keep_reduced:
        syscall("rm {vn}".format(vn = vname), False, True)

    pval["dnll"] = best_fit[2]
    pval["total"] = isum
    pval["pass"] = ipas
    return pval

def sum_up(g1, g2):
    if g1 is None and g2 is not None:
        return g2
    if g1 is not None and g2 is None:
        return g1
    if g1 is None and g2 is None:
        return None

    gs = OrderedDict()
    if g1["dnll"] != g2["dnll"]:
        print '\nWARNING :: incompatible expected/data dnll, when they should be!! Using the first dnll: ', g1["dnll"], ', over the second: ', g2["dnll"]
        sys.stdout.flush()

    gs["total"] = g1["total"] + g2["total"]
    gs["pass"] = g1["pass"] + g2["pass"]
    gs["dnll"] = g1["dnll"]

    return gs

def starting_poi(gvalues, fixpoi, resonly = False):
    if all(float(gg) < 0. for gg in gvalues):
        return [[], []]

    poi = 'r' if resonly else 'g'
    setpar = [poi + str(ii + 1) + '=' + gg for ii, gg in enumerate(gvalues) if (resonly and -20 <= float(gg) <= 20) or (not resonly and float(gg) >= 0)]
    frzpar = [poi + str(ii + 1) for ii, gg in enumerate(gvalues) if (resonly and -20 <= float(gg) <= 20) or (not resonly and float(gg) >= 0)] if fixpoi else []

    return [setpar, frzpar]

def channel_compatibility_set_freeze(ccbase, channels, set_freeze, extopt):
    # takes a list that would be given as the set_freeze arg used by many methods in the code
    # and returns two lists: one is the corresponding list for use in chancomp mode global fits, and the other per-channel fits
    extopt = [] if extopt == "" else extopt.split(' ')
    for option in ['--setParameters', '--freezeParameters']:
        while option in extopt:
            iopt = extopt.index(option)
            parameters = tokenize_to_list(remove_quotes(extopt.pop(iopt + 1))) if iopt + 1 < len(extopt) else []
            extopt.pop(iopt)
            if option == '--setParameters':
                set_freeze[0] += parameters
            elif option == '--freezeParameters':
                set_freeze[1] += parameters
    notcb = [[param for param in set_freeze[0] if param.split('=')[0] not in ccbase], [param for param in set_freeze[1] if param not in ccbase]]

    sf_global = [[], []]
    sf_channel = [[], []]
    for cb in ccbase:
        for cc in channels:
            sf_global[0].append(cb + "_" + cc + "=1")
            sf_global[1].append(cb + "_" + cc)
        sf_channel[0].append(cb + "_global=1")
        sf_channel[1].append(cb + "_global")

        setp = [param for param in set_freeze[0] if param.split('=')[0] == cb]
        if len(setp) == 1:
            value = param.split('=')[1]
            sf_global[0].append(cb + "_global=" + value)
            for cc in channels:
                sf_channel[0].append(cb + "_" + cc + "=" + value)

        if cb in set_freeze[1]:
            sf_global[1].append(cb + "_global")
            for cc in channels:
                sf_channel[1].append(cb + "_" + cc)
    sf_global[0].append("r=1")
    sf_global[1].append("r")
    sf_channel[0].append("r=1")
    sf_channel[1].append("r")

    sf_global = [sorted(list(set(sf_global[0]))), sorted(list(set(sf_global[1])))]
    sf_channel = [sorted(list(set(sf_channel[0]))), sorted(list(set(sf_channel[1])))]
    return [sf_global, sf_channel]

def channel_compatibility_fits(workspace, pois, scenarii, extopt, trkparam, oname, ntoy = 0, tname = ""):
    # computes the NLL for two scenarii and computes the difference in their NLLs
    # https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part3/commonstatsmethods/#channel-compatibility
    # https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/102x-comb2021/src/ChannelCompatibilityCheck.cc

    istoy = ntoy > 0
    results = []
    for idx in range(len(scenarii)):
        scenario, params = scenarii.items()[idx]
        never_gonna_give_you_up(
            command = "combineTool.py -v 0 -M MultiDimFit -d {dcd} -n _{sce} --redefineSignalPOIs '{poi}' "
            "{ext} {asm} {toy} {opd} --saveNLL --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 {stg} {exp} {trk}".format(
                dcd = workspace,
                sce = scenario,
                ext = extopt,
                trk = trkparam,
                poi = ",".join(pois[idx]),
                exp = set_parameter(params, "", []),
                stg = "{fit_strategy}",
                toy = "-s -1",
                asm = "-t {nt}".format(nt = ntoy) if ntoy > 0 else "",
                opd = "--toysFile '" + tname + "'" if ntoy > 0 else "",
            ),

            failure_cleanups = [
                [syscall, "rm higgsCombine_{sce}.MultiDimFit.mH{mmm}*.root".format(sce = scenario, mmm = mstr), False]
            ],

            tolerances = list(range(4)),
            usehesse = args.usehesse,
            first_fit_strategy = args.fitstrat if args.fitstrat > -1 else 0
        )

        syscall("mv higgsCombine_{sce}.MultiDimFit.mH*.root xxtmpxx_{sce}.root".format(sce = scenario), False, True)
        result = get_fit(
            dname = "xxtmpxx_{sce}.root".format(sce = scenario),
            attributes = ["deltaNLL", "nll", "nll0"] + pois[idx] + ['trackedError_' + poi for poi in pois[idx]],
            qexp_eq_m1 = True,
            loop_all = istoy
        )
        if not istoy:
            results.append([result])
        else:
            results.append(result)

    ofile = TFile(oname, "recreate")
    tree = TTree("chancomp", "")
    ddNLL = array('d', [0])
    tree.Branch('ddNLL', ddNLL, 'ddNLL/D')
    dNLLg = array('d', [0])
    tree.Branch('dNLL_global', dNLLg, 'dNLL_global/D')
    dNLLc = array('d', [0])
    tree.Branch('dNLL_channel', dNLLc, 'dNLL_channel/D')
    brpois = [[], []]
    brerrs = [[], []]
    for idx in range(len(pois)):
        for ipoi in range(len(pois[idx])):
            brpois[idx].append(array('d', [0]))
            tree.Branch(pois[idx][ipoi], brpois[idx][-1], pois[idx][ipoi] + '/D')
            brerrs[idx].append(array('d', [0]))
            tree.Branch(pois[idx][ipoi] + "_error", brerrs[idx][-1], pois[idx][ipoi] + '_error/D')

    for rr in range(len(results[0])):
        dNLLg[0] = 2. * sum(results[0][rr][:3])
        dNLLc[0] = 2. * sum(results[1][rr][:3])
        for idx in range(len(pois)):
            for ipoi in range(len(pois[idx])):
                brpois[idx][ipoi][0] = results[idx][rr][ipoi + 3]
                brerrs[idx][ipoi][0] = results[idx][rr][ipoi + len(pois[idx]) + 3]
        ddNLL[0] = dNLLg[0] - dNLLc[0]
        tree.Fill()
    tree.Write()
    ofile.Close()
    syscall("rm xxtmpxx*.root", False, True)
    syscall("rm {tn}".format(tn = tname), False, True)

def hadd_files(dcdir, point_tag, fileexp, direxp):
    '''
    fileexp: a list of file expressions to glob, in order: source, merged
    direxp: a list of directory expressions to glob, in order: source, merged
    source directory is the directory containing source files, ditto for merged
    '''

    fileexp = [ff if ff.endswith(".root") else ff + ".root" for ff in fileexp]
    fsrc, fmrg = fileexp
    dsrc, dmrg = direxp
    files = recursive_glob(dcdir, "{ptg}_*_{src}".format(ptg = point_tag, src = fsrc))

    if len(files) > 0:
        print "\ntwin_point_ahtt :: source files with expression '{src}' detected, merging them...".format(src = fsrc)
        files = set([os.path.basename(re.sub(fsrc.replace('*', '.*'), fmrg, ff)) for ff in files])

        mrgdir = ""
        for ii, ff in enumerate(files):
            ftmp = fmrg.replace(".root", "_x.root")
            if ii % max_nfile_per_dir == 0:
                mrgdir = make_timestamp_dir(base = dcdir, prefix = dmrg)
                directory_to_delete(location = mrgdir)

            tomerge = recursive_glob(dcdir, ff)
            if len(tomerge) > 0:
                for tm in tomerge:
                    directory_to_delete(location = tm)
                syscall("mv {ff} {fg}".format(ff = tomerge[0], fg = mrgdir + ff.replace(fmrg, ftmp)), False, True)

            tomerge = list(set(recursive_glob(dcdir, ff.replace(fmrg, fsrc)) + recursive_glob(dcdir, ff.replace(fmrg, ftmp))))
            for tm in tomerge:
                if dsrc in tm:
                    directory_to_delete(location = tm)

            jj = 0
            while len(tomerge) > 0:
                if len(tomerge) > 1:
                    about_right = 200 # arbitrary, only to prevent commands getting too long
                    tomerge = chunks(tomerge, len(tomerge) // about_right)
                    merged = []

                    for itm, tm in enumerate(tomerge):
                        ftmp = fmrg.replace(".root", "_{jj}-{itm}.root".format(jj = jj, itm = itm))
                        mname = mrgdir + ff.replace(fmrg, ftmp)
                        while os.path.isfile(mname):
                            jj += 1
                            mname = mrgdir + ff.replace(fmrg, ftmp)
                        syscall("hadd -f -k {ff} {fg} && rm {fg}".format(ff = mname, fg = " ".join(tm)))
                        merged.append(mname)

                    jj += 1
                    tomerge = merged
                    continue

                elif len(tomerge) == 1:
                    syscall("mv {fg} {ff}".format(
                        fg = tomerge[0],
                        ff = mrgdir + ff
                    ), False, True)
                    tomerge = []
        directory_to_delete(location = None, flush = True)

def is_ah_res_only(datacard):
    processes = list_of_processes(datacard)
    return all([point + part not in processes for point in points for part in ["_pos", "_neg"]])

if __name__ == '__main__':
    print "twin_point_ahtt :: called with the following arguments"
    print sys.argv[1:]
    print "\n"
    print " ".join(sys.argv)
    print "\n"
    sys.stdout.flush()

    parser = ArgumentParser()
    common_point(parser)
    common_common(parser)
    common_fit_pure(parser)
    common_fit(parser)
    make_datacard_pure(parser)
    make_datacard_forwarded(parser)
    common_2D(parser)

    parser.add_argument("--run-idx", help = combine_help_messages["--run-idx"], default = -1, dest = "runidx", required = False, type = lambda s: int(remove_spaces_quotes(s)))

    args = parse_args(parser)

    points = args.point
    gvalues = args.gvalues
    if len(points) != 2 or len(gvalues) != 2:
        raise RuntimeError("this script is to be used with exactly two A/H points!")
    ptag = "{pnt}{tag}".format(pnt = "__".join(points), tag = args.otag)
    gstr = g_in_filename(gvalues)

    modes = args.mode
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    dcdir = args.basedir + "__".join(points) + args.tag + "/"

    mstr = str(get_point(points[0])[1]).replace(".0", "")
    masks = ["mask_" + mm + "=1" for mm in args.mask]
    print "the following channel x year combinations will be masked:", args.mask

    allmodes = ["datacard", "workspace", "validate",
                "best", "best-fit", "single", "cross",
                "generate", "gof", "fc-scan", "contour", "chancomp",
                "hadd", "merge", "compile",
                "prepost", "corrmat", "psfromws",
                "nll", "likelihood"]
    if (not all([mm in allmodes for mm in modes])):
        print "supported modes:", allmodes
        raise RuntimeError("unxpected mode is given. aborting.")

    # determine what to do with workspace, and do it
    rundc = "datacard" in modes or "workspace" in modes
    runvalid = "validate" in modes
    runbest = "best" in modes or "best-fit" in modes
    runsingle = "single" in modes
    runcross = "cross" in modes
    rungen = "generate" in modes
    rungof = "gof" in modes
    runfc = "fc-scan" in modes or "contour" in modes
    runcc = "chancomp" in modes
    runhadd = "hadd" in modes or "merge" in modes
    runcompile = "compile" in args.mode
    runprepost = "prepost" in modes or "corrmat" in modes
    runpsfromws = "psfromws" in modes
    runnll = "nll" in modes or "likelihood" in modes

    if len(args.fcexp) > 0 and not all([expected_scenario(exp) is not None for exp in args.fcexp]):
        print "given expected scenarii:", args.fcexp
        raise RuntimeError("unexpected expected scenario is given. aborting.")

    runbest = runsingle or runbest or rundc or runcross
    args.keepbest = False if runbest else args.keepbest

    if runsingle:
        args.extopt += " --algo singles --cl=0.68"
    if runcross:
        args.extopt += " --algo cross --cl=0.68"
    if runsingle and runcross:
        raise RuntimeError("too stupid to do cross and single in same run")

    if (rungen or runfc) and any(float(gg) < 0. for gg in gvalues):
        raise RuntimeError("in toy generation or FC scans no g can be negative!!")

    if rundc:
        print "\ntwin_point_ahtt :: making datacard"
        make_datacard_with_args(scriptdir, args)
        ahresonly = is_ah_res_only(
            dcdir + "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else
            dcdir + "ahtt_" + args.channel + '_' + args.year + ".txt"
        )

        print "\ntwin_point_ahtt :: making workspaces"
        for ihsum in [True, False]:
            if ahresonly:
                modelopt = "-P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel "
                modelopt += " ".join(["--PO 'map=.*/{pp}_res:{rr}[1,-5,5]'".format(
                    pp = pp,
                    rr = rr
                ) for pp, rr in zip(points, ["r1", "r2"])])
            else:
                modelopt = "-P CombineHarvester.CombineTools.MultiInterferencePlusFixed:multiInterferencePlusFixed "
                modelopt += "--PO 'signal={pnt}' {pos} {dyt}".format(
                    pnt = ",".join(points),
                    pos = " ".join(["--PO " + stuff for stuff in ["verbose", "no-r"]]),
                    dyt = "--PO yukawa" if "EWK_TT" in args.assignal else ""
                )

            syscall("combineTool.py -v 0 -M T2W -i {dcd} -o workspace_{wst}.root -m {mmm} {mop} {opt} {whs} {ext}".format(
                dcd = dcdir + "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else dcdir + "ahtt_" + args.channel + '_' + args.year + ".txt",
                wst = "twin-g" if ihsum else "fitdiag",
                mmm = mstr,
                mop = modelopt,
                opt = "--channel-masks",
                whs = "--X-pack-asympows --optimize-simpdf-constraints=cms --no-wrappers --use-histsum" if ihsum else "",
                ext = args.extopt
            ))

    if runvalid:
        print "\ntwin_point_ahtt :: validating datacard"
        syscall("ValidateDatacards.py --jsonFile {dcd}{ptg}_validate.json --printLevel 3 {dcd}{crd}".format(
            dcd = dcdir,
            ptg = ptag,
            crd = "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else "ahtt_" + args.channel + '_' + args.year + ".txt"
        ))

    ahresonly = is_ah_res_only(
        dcdir + "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else
        dcdir + "ahtt_" + args.channel + '_' + args.year + ".txt"
    )
    # pois to use in the fit
    poiset = args.poiset if len(args.poiset) else ["r1", "r2"] if ahresonly else ["g1", "g2"]
    poiset = sorted(list(set(poiset)))
    notah = poiset != ["r1", "r2"] if ahresonly else poiset != ["g1", "g2"]
    startpoi = starting_poi(gvalues, args.fixpoi, ahresonly)
    # extra handling since FitDiagnostics s fit is hardcoded to float the first POI
    # therefore postfit plots with H only floating isn't possible otherwise
    if not notah:
        poiset = list(set(poiset) - set(startpoi[1])) if len(startpoi[1]) == 1 else poiset

    # parameter ranges for best fit file
    if ahresonly:
        ranges = ["{rr}: -20, 20".format(rr = rr) for rr in ["r1", "r2"]]
    else:
        ranges = ["{gg}: 0, 5".format(gg = gg) for gg in ["g1", "g2"]]

    if args.experimental:
        ranges += ["rgx{EWK_.*}", "rgx{QCDscale_ME.*}", "tmass"] # veeeery wide hedging for theory ME NPs

    default_workspace = dcdir + "workspace_twin-g.root"
    workspace = get_best_fit(
        dcdir, "__".join(points), [args.otag, args.tag],
        args.defaultwsp, args.keepbest, default_workspace, args.asimov,
        "cross" if runcross else "single" if runsingle else "", '__'.join(poiset) if notah else "",
        "{gvl}{fix}".format(gvl = gstr if gstr != "" else "", fix = "_fixed" if args.fixpoi and gstr != "" else ""),
        poiset,
        set_range(ranges),
        elementwise_add([startpoi, starting_nuisance(args.frzzero, args.frznzro, set())]), args.extopt, masks
    )

    if (rungen or (args.savetoy and (rungof or runfc))) and args.ntoy > 0:
        if args.toyloc == "":
            # no toy location is given, dump the toys in some directory under datacard directory
            args.toyloc = dcdir

    if rungen and args.ntoy > 0:
        print "\ntwin_point_ahtt :: starting toy generation"
        syscall("combine -v 0 -M GenerateOnly -d {dcd} -m {mmm} -n _{snm} {par} {stg} {toy} {poi}".format(
            dcd = workspace,
            mmm = mstr,
            snm = "toygen_" + str(args.runidx) if not args.runidx < 0 else "toygen",
            par = set_parameter(elementwise_add([startpoi, starting_nuisance(args.frzzero, args.frznzro, set())]), args.extopt, masks),
            stg = fit_strategy(strategy = args.fitstrat if args.fitstrat > -1 else 0),
            toy = "-s -1 --toysFrequentist -t " + str(args.ntoy) + " --saveToys",
            poi = "--redefineSignalPOIs '{poi}'".format(poi = ','.join(poiset))
        ))

        syscall("mv higgsCombine_{snm}.GenerateOnly.mH{mmm}*.root {opd}{ptg}_toys_{exp}_{poi}{toy}{idx}.root".format(
            opd = args.toyloc,
            snm = "toygen_" + str(args.runidx) if not args.runidx < 0 else "toygen",
            ptg = ptag,
            exp = "{gvl}{fix}".format(gvl = gstr if gstr != "" else "", fix = "_fixed" if args.fixpoi and gstr != "" else ""),
            poi = '__'.join(poiset) if notah else "",
            toy = "_n" + str(args.ntoy),
            idx = "_" + str(args.runidx) if not args.runidx < 0 else "",
            mmm = mstr,
        ), False)

    if rungof or runfc:
        readtoy = args.toyloc.endswith(".root") and not args.savetoy
        if readtoy:
            for ftoy in reversed(args.toyloc.split("_")):
                if ftoy.startswith("n"):
                    ftoy = int(ftoy.replace("n", "").replace(".root", ""))
                    break

            if ftoy < args.ntoy:
                print "\nWARNING :: file", args.toyloc, "contains less toys than requested in the run!"

    if rungof:
        if args.asimov:
            raise RuntimeError("mode gof is meaningless for asimov dataset!!")

        if args.resdir == "":
            args.resdir = dcdir

        # FIXME test that freezing NPs lead to a very peaky distribution (possibly single value)
        set_freeze = elementwise_add([startpoi, starting_nuisance(args.frzzero, args.frznzro, args.frzpost)])

        if args.gofrundat:
            print "\ntwin_point_ahtt :: starting goodness of fit, saturated model - data fit"
            scan_name = "gof-saturated_obs"

            never_gonna_give_you_up(
                command = "combine -v 0 -M GoodnessOfFit --algo=saturated -d {dcd} -m {mmm} -n _{snm} {stg} {prm} {ext}".format(
                    dcd = workspace,
                    mmm = mstr,
                    snm = scan_name,
                    stg = "{fit_strategy}",
                    msk = ",".join(masks) if len(masks) > 0 else "",
                    prm = set_parameter(set_freeze, args.extopt, masks),
                    ext = nonparametric_option(args.extopt),
                ),
                post_conditions = [
                    [lambda fexp, attrs, qem1: get_fit(glob.glob(fexp)[0], attrs, qem1),
                     "higgsCombine_{snm}.GoodnessOfFit.mH{mmm}*.root".format(snm = scan_name, mmm = mstr),
                     ['limit'],
                     True]
                ],
                failure_cleanups = [
                    [syscall, "rm higgsCombine_{snm}.GoodnessOfFit.mH{mmm}*.root".format(snm = scan_name, mmm = mstr), False]
                ],

                tolerances = list(range(4)),
                usehesse = args.usehesse,
                first_fit_strategy = args.fitstrat if args.fitstrat > -1 else 0
            )

            syscall("mv higgsCombine_{snm}.GoodnessOfFit.mH{mmm}*.root {dcd}{ptg}_{snm}.root".format(
                dcd = args.resdir,
                snm = scan_name,
                mmm = mstr,
                ptg = ptag,
            ), False)

        if args.ntoy > 0:
            print "\ntwin_point_ahtt :: starting goodness of fit, saturated model - toy fits"
            scan_name = "gof-saturated_toys"
            scan_name += "_" + str(args.runidx) if args.runidx > -1 else ""

            syscall("combine -v 0 -M GoodnessOfFit --algo=saturated -d {dcd} -m {mmm} -n _{snm} {stg} {toy} {opd} {svt} {prm} {ext}".format(
                dcd = workspace,
                mmm = mstr,
                snm = scan_name,
                stg = fit_strategy(strategy = args.fitstrat if args.fitstrat > -1 else 0),
                toy = "-s -1 --toysFrequentist -t " + str(args.ntoy),
                opd = "--toysFile '" + args.toyloc + "'" if readtoy else "",
                svt = "--saveToys" if args.savetoy else "",
                prm = set_parameter(set_freeze, args.extopt, masks),
                ext = nonparametric_option(args.extopt),
            ))

            syscall("mv higgsCombine_{snm}.GoodnessOfFit.mH{mmm}*.root {dcd}{ptg}_{snm}.root".format(
                dcd = args.resdir,
                ptg = ptag,
                snm = scan_name,
                mmm = mstr,
            ), False)

            if args.savetoy:
                syscall("cp {dcd}{ptg}_{snm}.root {opd}{ptg}_toys{gvl}{fix}{toy}{idx}.root".format(
                    dcd = args.resdir,
                    opd = args.toyloc,
                    ptg = ptag,
                    snm = scan_name,
                    gvl = "_" + gstr if gstr != "" else "",
                    fix = "_fixed" if args.fixpoi and gstr != "" else "",
                    toy = "_n" + str(args.ntoy),
                    idx = "_" + str(args.runidx) if not args.runidx < 0 else "",
                    mmm = mstr,
                ), False)

    if runfc:
        # FIXME mode fc still can't do frozen NPs. TBC if worth adding this.
        if args.resdir == "":
            args.resdir = dcdir

        scan_name = "pnt_g1_" + gvalues[0] + "_g2_" + gvalues[1]

        if args.fcrundat:
            print "\ntwin_point_ahtt :: finding the best fit point for FC scan"

            for fcexp in args.fcexp:
                scenario = expected_scenario(fcexp, resonly = ahresonly)
                identifier = "_" + scenario[0]
                parameters = [scenario[1]] if scenario[1] != "" else []

                # fit settings should be identical to the one above, since we just want to choose the wsp by fcexp rather than args.asimov
                fcwsp = get_best_fit(
                    dcdir, "__".join(points), [args.otag, args.tag],
                    args.defaultwsp, args.keepbest, default_workspace, fcexp != "obs", "", "",
                    "{gvl}{fix}".format(gvl = gstr if gstr != "" else "", fix = "_fixed" if args.fixpoi and gstr != "" else ""),
                    set_range(ranges),
                    elementwise_add([startpoi, starting_nuisance(args.frzzero, args.frznzro, set())]), args.extopt, masks
                )

                never_gonna_give_you_up(
                    command = "combineTool.py -v 0 -M MultiDimFit -d {dcd} -m {mmm} -n _{snm} --algo fixed --fixedPointPOIs '{par}' "
                    "{exp} {stg} {asm} {toy} {ext}".format(
                        dcd = fcwsp,
                        mmm = mstr,
                        snm = scan_name + identifier,
                        par = "g1=" + gvalues[0] + ",g2=" + gvalues[1],
                        exp = "--setParameters '" + ",".join(parameters + masks) + "'" if len(parameters + masks) > 0 else "",
                        stg = "{fit_strategy}",
                        asm = "-t -1" if fcexp != "obs" else "",
                        toy = "-s -1",
                        #ext = nonparametric_option(args.extopt),
                        ext = args.extopt,
                        #wsp = "--saveWorkspace --saveSpecifiedNuis=all" if False else ""
                    ),

                    post_conditions = [
                        [lambda fexp, attrs, qem1: get_fit(glob.glob(fexp)[0], attrs, qem1),
                         "higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root".format(
                             snm = scan_name + identifier,
                             mmm = mstr),
                         ['g1', 'g2', 'deltaNLL'],
                         ff] for ff in [True, False]
                    ],

                    failure_cleanups = [
                        [syscall, "rm higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root".format(snm = scan_name + identifier, mmm = mstr), False]
                    ],

                    tolerances = list(range(4)),
                    usehesse = args.usehesse,
                    first_fit_strategy = args.fitstrat if args.fitstrat > -1 else 0
                )

                syscall("mv higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root {dcd}{ptg}_fc-scan_{snm}.root".format(
                    dcd = args.resdir,
                    ptg = ptag,
                    snm = scan_name + identifier,
                    mmm = mstr
                ), False)

        if args.ntoy > 0:
            identifier = "_toys_" + str(args.runidx) if args.runidx > -1 else "_toys"
            print "\ntwin_point_ahtt :: performing the FC scan for toys"

            setpar, frzpar = ([], [])
            nus = ""
            nuf = ""
            if len(args.frznzro) > 0:
                nus = "," + ",".join([s.split(":")[0] + "=" + s.split(":")[1] for s in args.frznzro])
                nuf = "--freezeParameters " + ",".join([s.split(":")[0] for s in args.frznzro])
            syscall("combineTool.py -v 0 -M MultiDimFit -d {dcd} -m {mmm} -n _{snm} --algo fixed --fixedPointPOIs '{par}' "
                    "--setParameters '{par}{msk}{nus}' {nuf} {stg} {toy} {ext} {opd} {svt}".format(
                        dcd = workspace,
                        mmm = mstr,
                        snm = scan_name + identifier,
                        par = "g1=" + gvalues[0] + ",g2=" + gvalues[1],
                        msk = "," + ",".join(masks) if len(masks) > 0 else "",
                        nus = nus,
                        nuf = nuf,
                        stg = fit_strategy(strategy = args.fitstrat if args.fitstrat > -1 else 0),
                        toy = "-s -1 --toysFrequentist -t " + str(args.ntoy),
                        #ext = nonparametric_option(args.extopt),
                        ext = args.extopt,
                        opd = "--toysFile '" + args.toyloc + "'" if readtoy else "",
                        #byp = "--bypassFrequentistFit --fastScan" if False else "",
                        svt = "--saveToys" if args.savetoy else ""
                    ))

            syscall("mv higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root {dcd}{ptg}_fc-scan_{snm}.root".format(
                dcd = args.resdir,
                ptg = ptag,
                snm = scan_name + identifier,
                mmm = mstr,
            ), False)

            if args.savetoy:
                syscall("cp {dcd}{ptg}_fc-scan_{snm}.root {opd}{ptg}_toys{gvl}{fix}{toy}{idx}.root".format(
                    dcd = args.resdir,
                    opd = args.toyloc,
                    snm = scan_name + identifier,
                    ptg = ptag,
                    gvl = "_" + gstr if gstr != "" else "",
                    fix = "_fixed" if args.fixpoi and gstr != "" else "",
                    toy = "_n" + str(args.ntoy),
                    idx = "_" + str(args.runidx) if not args.runidx < 0 else "",
                    mmm = mstr,
                ), False)

    if runcc:
        ccbase = ["r1", "r2"] if ahresonly else ["g1", "g2"]
        ccbase += ["EWK_yukawa", "CMS_EtaT_norm_13TeV", "CMS_ChiT_norm_13TeV"]

        if any([cp not in ccbase for cp in poiset]):
            raise RuntimeError("twin_point_ahtt :: mode chancomp only works with any subset of [g1, g2, EWK_yukawa, CMS_EtaT_norm_13TeV, CMS_ChiT_norm_13TeV]")

        if args.asimov:
            raise RuntimeError("mode gof is meaningless for asimov dataset!!")

        print "\ntwin_point_ahtt :: channel compatibility mode"
        cctag = "chancomp{mm}".format(mm = "" if len(args.ccmasks) == 0 else "_" + "_".join(args.ccmasks))
        if not os.path.isfile(dcdir + "ahtt_{cct}.txt".format(cct = cctag)):
            channel_compatibility_hackery(
                dcdir + "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else dcdir + "ahtt_" + args.channel + '_' + args.year + ".txt",
                args.ccmasks
            )
            syscall("combineTool.py -v 0 -M T2W -i {dcd} -o workspace_{wst}.root -m {mmm} {mop} {opt} {whs} {ext}".format(
                dcd = dcdir + "ahtt_{cct}.txt".format(cct = cctag),
                wst = cctag,
                mmm = mstr,
                mop = "",
                opt = "--channel-masks",
                whs = "--X-pack-asympows --optimize-simpdf-constraints=cms --no-wrappers --use-histsum",
                ext = ""
            ))
        channels = channel_compatibility_masking(dcdir + "ahtt_{cct}.txt".format(cct = cctag), args.ccmasks)
        ccpois = [[poi + "_global" for poi in poiset], sorted(list(set([poi + "_" + cc for poi in poiset for cc in channels])))]
        ccprms = [[poi + "_global" for poi in ccbase], [poi + "_" + cc for poi in ccbase for cc in channels]]
        ccprms = [poi + "_product" for poi in ccprms[1] if "g1" not in poi and "g2" not in poi] + sum(ccprms, [])
        ccprms += [] if ahresonly else [tt + cc + "_product" for cc in channels for tt in ["g1_to4_", "g1_to2_", "mg1_to2_", "g2_to4_", "g2_to2_", "mg2_to2_"]]
        ccprms += [tt + cc + "_product" for cc in channels for tt in ["EWK_yukawa2_", "mEWK_yukawa_", "mEWK_yukawa2_"]]
        ccprms = sorted(list(set(ccprms)))

        remove_dupe = lambda x, y: [sorted(list(set(elementwise_add([x, y])[0]))), sorted(list(set(elementwise_add([x, y])[1])))]
        wspprm = channel_compatibility_set_freeze(
            ccbase,
            channels,
            elementwise_add([startpoi, starting_nuisance(args.frzzero, args.frznzro, set())]),
            args.extopt
        )

        trkparam = " --trackParameters '{prm}' --trackErrors '{prm}'".format(prm = ",".join(ccprms))
        chancomp_workspace = get_best_fit(
            dcdir, "__".join(points), [args.otag, args.tag],
            args.defaultwsp, args.keepbest, dcdir + "workspace_{cct}.root".format(cct = cctag), args.asimov,
            "", '__'.join(poiset) + "_{cct}".format(cct = cctag) if notah else "{cct}".format(cct = cctag),
            "{gvl}{fix}".format(gvl = gstr if gstr != "" else "", fix = "_fixed" if args.fixpoi and gstr != "" else ""),
            ccpois[0],
            "",
            wspprm[0], nonparametric_option(args.extopt + trkparam), []
        )

        fitprm = channel_compatibility_set_freeze(
            ccbase,
            channels,
            elementwise_add([startpoi, starting_nuisance(args.frzzero, args.frznzro, args.frzpost)]),
            args.extopt
        )
        scenarii = {"global": fitprm[0], "channel": fitprm[1]}

        if args.ccrundat:
            oname = "{dcd}{ptg}_chancomp_{exp}_{poi}{cct}_obs.root".format(
                dcd = args.resdir,
                ptg = ptag,
                exp = "{gvl}{fix}".format(gvl = gstr if gstr != "" else "", fix = "_fixed" if args.fixpoi and gstr != "" else ""),
                poi = '__'.join(poiset) if notah else "",
                cct = "" if len(args.ccmasks) == 0 else "_" + "_".join(args.ccmasks)
            )
            channel_compatibility_fits(chancomp_workspace, ccpois, scenarii, nonparametric_option(args.extopt), trkparam, oname)

        if args.ntoy > 0:
            # generate toy in global mode - the same toy are used for both fits
            syscall("combine -v 0 -M GenerateOnly -d {dcd} -m {mmm} -n _{snm} {prm} {stg} {toy}".format(
                dcd = chancomp_workspace,
                mmm = mstr,
                snm = "toygen_" + str(args.runidx) if not args.runidx < 0 else "toygen",
                prm = set_parameter(wspprm[0], "", []),
                stg = fit_strategy(strategy = args.fitstrat if args.fitstrat > -1 else 0),
                toy = "-s -1 --toysFrequentist -t " + str(args.ntoy) + " --saveToys"
            ))
            tname = "{ptg}_toys_{cct}{toy}{idx}.root".format(
                ptg = ptag,
                cct = cctag,
                toy = "_n" + str(args.ntoy),
                idx = "_" + str(args.runidx) if not args.runidx < 0 else "",
            )
            syscall("mv higgsCombine_{snm}.GenerateOnly.mH{mmm}*.root {tnm}".format(
                snm = "toygen_" + str(args.runidx) if not args.runidx < 0 else "toygen",
                mmm = mstr,
                tnm = tname
            ), False)

            oname = "{dcd}{ptg}_chancomp_{exp}_{poi}{cct}_toys{idx}.root".format(
                dcd = args.resdir,
                ptg = ptag,
                exp = "{gvl}{fix}".format(gvl = gstr if gstr != "" else "", fix = "_fixed" if args.fixpoi and gstr != "" else ""),
                poi = '__'.join(poiset) if notah else "",
                cct = "" if len(args.ccmasks) == 0 else "_" + "_".join(args.ccmasks),
                idx = "_" + str(args.runidx) if not args.runidx < 0 else "",
            )
            channel_compatibility_fits(chancomp_workspace, ccpois, scenarii, nonparametric_option(args.extopt), trkparam, oname, args.ntoy, tname)

    if runhadd:
        for imode in ["fc", "gof", "chancomp"]:
            hadd_files(dcdir, ptag, ["toys_*.root", "toys.root"], ["{im}-result".format(im = imode), "{im}-merge".format(im = imode)])
        hadd_files(dcdir, ptag, ["reduced.root", "reduced.root"], ["fc-merge", "fc-reduced"])

    if runcompile:
        toys = recursive_glob(dcdir, "{ptg}_fc-scan_*_toys.root".format(ptg = ptag))
        idxs = recursive_glob(dcdir, "{ptg}_fc-scan_*_toys_*.root".format(ptg = ptag))
        if len(toys) == 0 or len(idxs) > 0:
            print "\ntwin_point_ahtt :: either no merged toy files are present, or some indexed ones are."
            raise RuntimeError("run either the fc-scan or hadd modes first before proceeding!")

        print "\ntwin_point_ahtt :: compiling FC scan results..."
        for fcexp in args.fcexp:
            scenario = expected_scenario(fcexp, resonly = ahresonly)
            print "\ntwin_point_ahtt :: compiling scenario {sc}...".format(sc = scenario)
            sys.stdout.flush()

            expfits = recursive_glob(dcdir, "{ptg}_fc-scan_*_{exp}.root".format(ptg = ptag, exp = scenario[0]))
            expfits.sort()

            previous_grids = glob.glob("{dcd}{ptg}_fc-scan_{exp}_*.json".format(
                dcd = dcdir,
                ptg = ptag,
                exp = scenario[0]
            ))
            previous_grids.sort(key = lambda name: int(name.split('_')[-1].split('.')[0]))
            no_previous = args.ignoreprev or len(previous_grids) == 0

            if len(expfits) == 0 and no_previous:
                raise RuntimeError("result compilation can't proceed without the best fit files or previous grids being available!!")
            gpoints = points_to_compile(points, expfits, None if no_previous else previous_grids[-1])
            idx = 0 if len(previous_grids) == 0 else int(previous_grids[-1].split("_")[-1].split(".")[0]) + 1
            best_fit = get_fit(expfits[0], ['g1', 'g2', 'deltaNLL']) if len(expfits) > 0 else read_previous_best_fit(previous_grids[-1])

            grid = OrderedDict()
            grid["points"] = points
            grid["best_fit_g1_g2_dnll"] = best_fit
            grid["g-grid"] = OrderedDict() if idx == 0 or args.ignoreprev else read_previous_grid(points, best_fit, previous_grids[-1])

            for pnt in gpoints:
                gv = stringify(pnt)
                ename = recursive_glob(dcdir, "{ptg}_fc-scan_pnt_g1_{g1}_g2_{g2}_{exp}.root".format(
                    ptg = ptag,
                    g1 = pnt[0],
                    g2 = pnt[1],
                    exp = scenario[0]
                ))
                ename = ename[0] if len(ename) else "{ptg}_fc-scan_pnt_g1_{g1}_g2_{g2}_{exp}.root".format(
                    ptg = ptag,
                    g1 = pnt[0],
                    g2 = pnt[1],
                    exp = scenario[0]
                )
                current_fit = get_fit(ename, ['g1', 'g2', 'deltaNLL'])
                expected_fit = get_fit(ename, ['g1', 'g2', 'deltaNLL'], False)

                if expected_fit is None:
                    if gv in grid["g-grid"]:
                        expected_fit = pnt + (grid["g-grid"][gv]["dnll"],)
                    else:
                        raise RuntimeError("failed getting the expected fit for point " + gv + " from file or grid. aborting.")

                if current_fit is not None and best_fit != current_fit:
                    print '\nWARNING :: incompatible best fit across different g values!! ignoring current, assuming it is due to numerical instability!'
                    print 'this should NOT happen too frequently within a single compilation, and the difference should not be large!!'
                    print "current result ", ename, ": ", current_fit
                    print "first result ", best_fit
                    sys.stdout.flush()

                tname = recursive_glob(dcdir, os.path.basename(ename).replace("{exp}.root".format(exp = scenario[0]), "toys.root"))
                tname = tname[0] if len(tname) else ""

                gg = get_toys(toy_name = tname, best_fit = expected_fit, keep_reduced = args.collecttoy and fcexp == args.fcexp[-1])
                if gg is not None and args.rmroot:
                    directory_to_delete(location = ename)
                    syscall("rm " + ename, False, True)
                    if fcexp == args.fcexp[-1]:
                        directory_to_delete(location = tname)
                        syscall("rm " + tname, False, True)

                if gv in grid["g-grid"]:
                    grid["g-grid"][gv] = sum_up(grid["g-grid"][gv], gg)
                else:
                    grid["g-grid"][gv] = gg

            grid["g-grid"] = OrderedDict(sorted(grid["g-grid"].items()))
            with open("{dcd}{ptg}_fc-scan_{exp}_{idx}.json".format(
                    dcd = dcdir,
                    ptg = ptag,
                    exp = scenario[0],
                    idx = str(idx)
            ), "w") as jj:
                json.dump(grid, jj, indent = 1)

        directory_to_delete(location = None, flush = True)

    if runprepost or runpsfromws:
        gfit = g_in_filename(gvalues)
        set_freeze = elementwise_add([startpoi, starting_nuisance(args.frzzero, args.frznzro, args.frzpost)])
        fitdiag_workspace = get_best_fit(
            dcdir, "__".join(points), [args.otag, args.tag],
            args.defaultwsp, args.keepbest, dcdir + "workspace_fitdiag.root", args.asimov,
            "", '__'.join(poiset) + "_fitdiag" if notah else "fitdiag",
            "{gvl}{fix}".format(gvl = gstr if gstr != "" else "", fix = "_fixed" if args.fixpoi and gstr != "" else ""),
            poiset,
            set_range(ranges),
            elementwise_add([startpoi, starting_nuisance(args.frzzero, args.frznzro, set())]), args.extopt, masks
        )

        fitdiag_result, fitdiag_shape = ["{dcd}{ptg}_fitdiagnostics_{fdo}{poi}{gvl}{fix}_{ftp}.root".format(
            dcd = dcdir,
            ptg = ptag,
            fdo = fdoutput,
            poi = "_" + '__'.join(poiset) if notah else "",
            gvl = "_" + gfit if gfit != "" else "",
            fix = "_fixed" if args.fixpoi and gfit != "" else "",
            ftp = args.prepostfit
        ) for fdoutput in ["result", "shape"]]

    if runprepost:
        print "\ntwin_point_ahtt :: making pre- and postfit plots and covariance matrices"
        stp = startpoi if args.prepostfit == 's' else starting_poi(gvalues if notah else ["0.", "0."], args.fixpoi if notah else True)
        set_freeze = elementwise_add([stp, starting_nuisance(args.frzzero, args.frznzro, args.frzpost)])
        del stp
        fitopt = "--skipBOnlyFit" if args.prepostfit == 's' else '--customStartingPoint --skipSBFit'

        never_gonna_give_you_up(
            command = "combine -v 0 -M FitDiagnostics {dcd} --saveWithUncertainties --saveNormalizations --saveShapes "
            "--saveOverallShapes --plots -m {mmm} -n _prepost {stg} {asm} {poi} {prm} {ext} {fop}".format(
                dcd = fitdiag_workspace,
                mmm = mstr,
                stg = "{fit_strategy}",
                asm = "-t -1" if args.asimov else "",
                poi = "--redefineSignalPOIs '{poi}'".format(poi = ','.join(poiset)),
                prm = set_parameter(set_freeze, args.extopt, masks),
                ext = nonparametric_option(args.extopt),
                fop = fitopt
            ),

            followups = [
                [syscall, "rm *_th1x_*.png", False, True],
                [syscall, "rm covariance_fit_?.png", False, True],
                [syscall, "rm higgsCombine_prepost*.root", False, True],
                [syscall, "rm combine_logger.out", False, True],
                [syscall, "mv fitDiagnostics_prepost.root {fdr}".format(fdr = fitdiag_result), False]
            ],

            fit_result_names = [fitdiag_result, ["fit_{ftp}".format(ftp = args.prepostfit)]],

            optimize = False,
            usehesse = args.usehesse,
            first_fit_strategy = args.fitstrat if args.fitstrat > -1 else 0
        )

    if runpsfromws:
        # from https://cms-talk.web.cern.ch/t/postfit-uncertainty-bands-very-large/20967/9
        prepostmerge = update_mask(args.prepostmerge)
        assert len(prepostmerge) > 1, "--prepost-merge must be longer than 1 in mode psfromws!"
        assert os.path.isfile(dcdir + "ahtt_combined.txt"), "this mode makes no sense with just one channel!"
        assert os.path.isfile(fitdiag_result), "file {fdr} missing! mode prepost must be ran before this one!".format(fdr = fitdiag_result)

        print "\ntwin_point_ahtt :: making channel block workspace"
        nicemerge = len(args.prepostmerge) == 1
        ppmtxt, ppmwsp = "ahtt_prepostmerge.txt", "workspace_prepostmerge.root"

        inputfiles = ["ahtt_input.root"] + ["ahtt_" + ppm + ".txt" for ppm in prepostmerge]
        for inputfile in inputfiles:
            syscall("cp " + dcdir + inputfile + " .")
        syscall("combineCards.py {cards} > {comb}".format(
            cards = " ".join([ppm + "=" + "ahtt_" + ppm + ".txt" for ppm in prepostmerge]),
            comb = ppmtxt
        ))

        syscall("combineTool.py -v 0 -M T2W -i {txt} -o {wsp} -m {mmm} "
                "-P CombineHarvester.CombineTools.MultiInterferencePlusFixed:multiInterferencePlusFixed "
                "--PO 'signal={pnt}' {pos} {dyt} {ext}".format(
            txt = ppmtxt,
            wsp = ppmwsp,
            mmm = mstr,
            pnt = ",".join(points),
            pos = " ".join(["--PO " + stuff for stuff in ["verbose", "no-r"]]),
            dyt = "--PO yukawa" if "EWK_TT" in args.assignal else "",
            ext = args.extopt
        ))
        syscall("rm {inf}".format(inf = " ".join(inputfiles[1:])), False, True)
        inputfiles = [inputfiles[0], ppmtxt, ppmwsp]

        print "\ntwin_point_ahtt :: merging postfit plots as per fit result"
        syscall("PostFitShapesFromWorkspace -d {dcd} -w {fdw} -o {fds} --print --postfit --covariance --sampling --skip-prefit --skip-proc-errs --total-shapes -f {fdr}:fit_{ftp}".format(
            dcd = ppmtxt,
            fdw = ppmwsp,
            fds = fitdiag_shape.replace(
                "fitdiagnostics_shape",
                "psfromws_{ppm}".format(ppm = args.prepostmerge[0] if nicemerge else "-".join(prepostmerge))
            ),
            fdr = fitdiag_result,
            ftp = args.prepostfit
        ))
        syscall("rm {inf}".format(inf = " ".join(inputfiles)), False, True)

    if runnll:
        if len(args.nllparam) < 1:
            raise RuntimeError("what parameter is the nll being scanned against??")

        if len(args.fcexp) != 1:
            raise RuntimeError("in mode nll only one expected scenario is allowed.")

        print "\ntwin_point_ahtt :: evaluating nll as a function of {nlp}".format(nlp = ", ".join(args.nllparam))

        nparam = len(args.nllparam)
        scenario = expected_scenario(args.fcexp[0], gvalues_syntax = True, resonly = ahresonly)
        set_freeze = elementwise_add([starting_poi(scenario[1] if args.fcexp[0] != "obs" else gvalues, args.fixpoi, ahresonly), starting_nuisance(args.frzzero, args.frznzro, args.frzpost)])

        # fit settings should be identical to the one above, since we just want to choose the wsp by fcexp rather than args.asimov
        fcwsp = get_best_fit(
            dcdir, "__".join(points), [args.otag, args.tag],
            args.defaultwsp, args.keepbest, default_workspace, args.fcexp[0] != "obs", "", "",
            "{gvl}{fix}".format(gvl = gstr if gstr != "" else "", fix = "_fixed" if args.fixpoi and gstr != "" else ""),
            set_range(ranges),
            elementwise_add([startpoi, starting_nuisance(args.frzzero, args.frznzro, set())]), args.extopt, masks
        )

        unconstrained = ",".join([param for param in args.nllunconstrained if param not in ["g1", "g2", "r1", "r2", "CMS_EtaT_norm_13TeV", "CMS_ChiT_norm_13TeV"]])

        if len(args.nllwindow) < nparam:
            args.nllwindow += ["0,3" if param in ["g1", "g2"] else "-5,5" for param in args.nllparam[len(args.nllwindow):]]
        minmax = [(float(values.split(",")[0]), float(values.split(",")[1])) for values in args.nllwindow]

        if len(args.nllnpnt) < nparam:
            nsample = 32 # per unit interval
            args.nllnpnt += [nsample * round(minmax[ii][1] - minmax[ii][0]) for ii in range(len(args.nllnpnt), nparam)]
        interval = [list(np.linspace(minmax[ii][0], minmax[ii][1], num = args.nllnpnt[ii if ii < nparam else -1])) for ii in range(nparam)]
        nllparam = " ".join(["-P {param}".format(param = param) for param in args.nllparam])

        nelement = 0
        for element in itertools.product(*interval):
            nelement += 1
            nllpnt = ",".join(["{param}={value}".format(param = param, value = value) for param, value in zip(args.nllparam, element)])
            nllname = args.fcexp[0] + "_".join([""] + ["{param}_{value}".format(param = param, value = floattopm(round(value, 5))) for param, value in zip(args.nllparam, element)])

            never_gonna_give_you_up(
                command = "combineTool.py -v 0 -M MultiDimFit -d {dcd} -m {mmm} -n _{snm} --algo fixed {par} --fixedPointPOIs '{pnt}' {uco} "
                "{exp} {stg} {asm} {ext} --saveNLL".format(
                    dcd = fcwsp,
                    mmm = mstr,
                    snm = nllname,
                    pnt = nllpnt,
                    par = nllparam,
                    uco = "--redefineSignalPOIs '{uco}'".format(uco = unconstrained) if len(unconstrained) else "",
                    exp = set_parameter(set_freeze, args.extopt, masks),
                    stg = "{fit_strategy}",
                    asm = "-t -1" if args.fcexp[0] != "obs" else "",
                    ext = nonparametric_option(args.extopt),
                ),

                failure_cleanups = [
                    [syscall, "rm higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root".format(snm = nllname, mmm = mstr), False]
                ],

                usehesse = args.usehesse,
                first_fit_strategy = args.fitstrat if args.fitstrat > -1 else 0
            )

        if nelement > 1:
            syscall("hadd -f -k {dcd}{ptg}_nll_{exp}_{fit}.root higgsCombine_{exp}_{par}*MultiDimFit.mH{mmm}.root && rm higgsCombine_{exp}_{par}*MultiDimFit.mH{mmm}.root".format(
                dcd = dcdir,
                ptg = ptag,
                exp = args.fcexp[0],
                fit = "_".join(["{pp}_{mi}to{ma}".format(pp = pp, mi = floattopm(mm[0]), ma = floattopm(mm[1])) for pp, mm in zip(args.nllparam, minmax)]),
                par = "*".join(args.nllparam),
                mmm = mstr
            ))
        else:
            syscall("mv higgsCombine_{exp}_{par}*MultiDimFit.mH{mmm}.root {dcd}{ptg}_nll_{exp}_{fit}.root".format(
                dcd = dcdir,
                ptg = ptag,
                exp = args.fcexp[0],
                fit = "_".join(["{pp}_{mi}to{ma}".format(pp = pp, mi = floattopm(mm[0]), ma = floattopm(mm[1])) for pp, mm in zip(args.nllparam, minmax)]),
                par = "*".join(args.nllparam),
                mmm = mstr
            ))

    if args.compress:
        syscall(("tar -czf {dcd}.tar.gz {dcd} && rm -r {dcd}").format(
            dcd = dcdir[:-1]
        ))
