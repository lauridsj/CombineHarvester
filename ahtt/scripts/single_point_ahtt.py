#!/usr/bin/env python
# sets the model-independent limit on a single A/H signal point

from argparse import ArgumentParser
import os
import sys
import numpy as np

from collections import OrderedDict
import json

from ROOT import TFile, TTree

from utilities import syscall, get_point, chunks, min_g, max_g, make_best_fit, starting_nuisance, elementwise_add

def get_limit(lfile):
    lfile = TFile.Open(lfile)
    ltree = lfile.Get("limit")
    limit = OrderedDict()

    for i in ltree:
        qstr = "obs"
        if abs(ltree.quantileExpected - 0.025) < 0.01:
            qstr = "exp-2"
        elif abs(ltree.quantileExpected - 0.16) < 0.01:
            qstr = "exp-1"
        elif abs(ltree.quantileExpected - 0.5) < 0.01:
            qstr = "exp0"
        elif abs(ltree.quantileExpected - 0.84) < 0.01:
            qstr = "exp+1"
        elif abs(ltree.quantileExpected - 0.975) < 0.01:
            qstr = "exp+2"

        limit[qstr] = ltree.limit

    lfile.Close()
    return limit

def get_nll_one_poi(lfile):
    # relevant branches are r, nllo0 (for the minimum NLL value), deltaNLL (wrt nll0), quantileExpected (-1 for data, rest irrelevant)
    lfile = TFile.Open(lfile)
    ltree = lfile.Get("limit")
    nll = OrderedDict()
    nll["obs"] = OrderedDict()
    nll["dnll"] = OrderedDict()

    for i in ltree:
        if ltree.quantileExpected == -1.:
            nll["obs"]["r"] = 1.
            nll["obs"]["g"] = ltree.r
            nll["obs"]["nll0"] = ltree.nll0
        else:
            nll["dnll"][ ltree.r ] = ltree.deltaNLL

    lfile.Close()
    return nll

def get_nll_g_scan(lfile):
    lfile = TFile.Open(lfile)
    ltree = lfile.Get("limit")
    nll = OrderedDict()
    nll["obs"] = OrderedDict()
    nll["dnll"] = OrderedDict()

    nll0 = sys.float_info.max

    for i in ltree:
        if ltree.quantileExpected == -1. and ltree.nll0 < nll0:
            nll["obs"]["r"] = ltree.r
            nll["obs"]["g"] = ltree.g
            nll["obs"]["nll0"] = ltree.nll0

    for i in ltree:
        if ltree.quantileExpected >= 0.:
            nll["dnll"][ ltree.g ] = ltree.nll0 + ltree.deltaNLL - nll["obs"]["nll0"]

    lfile.Close()
    return nll

def single_point_scan(args):
    gval, dcdir, mstr, accuracies, poi_range, strategy, asimov, mcstat = args
    gstr = str(round(gval, 3)).replace('.', 'p')

    epsilon = 2.**-17
    nstep = 1

    syscall("combineTool.py -M AsymptoticLimits -d {dcd}workspace_g-scan.root -m {mmm} -n _limit_g-scan_{gst} "
            "--setParameters g={gvl} --freezeParameters g {acc} --picky {prg} "
            "--singlePoint 1 {stg} {asm} {mcs}".format(
                dcd = dcdir,
                mmm = mstr,
                gst = gstr,
                gvl = gval,
                acc = accuracies,
                prg = poi_range,
                stg = strategy,
                asm = "-t -1" if asimov else "",
                mcs = "--X-rtd MINIMIZER_analytic" if mcstat else ""
            ))

    limit = get_limit("higgsCombine_limit_g-scan_{gstr}.POINT.1.AsymptoticLimits.mH{mmm}.root".format(
        mmm = mstr,
        gstr = gstr,
    ))

    if all([ll >= 0. for qq, ll in limit.items()]):
        return [gval, limit, min([(abs(ll - 0.05), ll) for qq, ll in limit.items()])[1]] # third being the closest cls to 0.05 among the quantiles

    geps = 0.
    syscall("rm higgsCombine_limit_g-scan_{gstr}.POINT.1.*AsymptoticLimits*.root".format(gstr = gstr), False, True)

    for factor in [1., -1.]:
        fgood = False
        for ii in range(1, nstep + 1):
            syscall("combineTool.py -M AsymptoticLimits -d {dcd}workspace_g-scan.root -m {mmm} -n _limit_g-scan_{gst} "
                    "--setParameters g={gvl} --freezeParameters g {acc} --picky {prg} "
                    "--singlePoint 1 {stg} {asm} {mcs}".format(
                        dcd = dcdir,
                        mmm = mstr,
                        gst = gstr + "eps",
                        gvl = gval + (ii * factor * epsilon),
                        acc = accuracies,
                        prg = poi_range,
                        stg = strategy,
                        asm = "-t -1" if asimov else "",
                        mcs = "--X-rtd MINIMIZER_analytic" if mcstat else ""
                    ))

            leps = get_limit("higgsCombine_limit_g-scan_{gstr}.POINT.1.AsymptoticLimits.mH{mmm}.root".format(
                mmm = mstr,
                gstr = gstr + "eps",
            ))
            fgood = all([ll >= 0. for qq, ll in leps.items()])

            if fgood:
                geps = (ii * factor * epsilon)
                for quantile in leps.keys():
                    limit[quantile] = leps[quantile]
                break
        if fgood:
            break

    if all([ll >= 0. for qq, ll in limit.items()]):
        return [gval + geps, limit, min([(abs(ll - 0.05), ll) for qq, ll in limit.items()])[1]] # third being the closest cls to 0.05 among the quantiles

    return None

def dotty_scan(args):
    gvals, dcdir, mstr, accuracies, poi_range, strategy, asimov, mcstat = args
    if len(gvals) < 2:
        return None
    gvals = sorted(gvals)

    results = []
    ii = 0
    while ii < len(gvals):
        result = single_point_scan((gvals[ii], dcdir, mstr, accuracies, poi_range, strategy, asimov, mcstat))

        if result is None:
            ii += 3
            continue

        if (0.025 < result[2] < 0.1):
            ii += 1
        else:
            ii += 2

        results.append(result)
    return results

def starting_poi(onepoi, gvalue, rvalue, fixpoi):
    setpar = []
    frzpar = []

    if onepoi:
        if gvalue >= 0.:
            setpar.append('r=' + str(gvalue))

        if fixpoi:
            frzpar.append('r')
    else:
        if gvalue >= 0.:
            setpar.append('g=' + str(gvalue))

            if fixpoi:
                frzpar.append('g')

        if rvalue >= 0.:
            setpar.append('r=' + str(rvalue))

            if fixpoi:
                frzpar.append('r')

    return [setpar, frzpar]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "desired signal point to run on", default = "", required = True)
    parser.add_argument("--mode", help = "combine mode to run, comma separated", default = "datacard", required = False)

    parser.add_argument("--signal", help = "signal filenames. comma separated", default = "", required = False)
    parser.add_argument("--background", help = "data/background filenames. comma separated", default = "", required = False)

    parser.add_argument("--channel", help = "final state channels considered in the analysis. comma separated", default = "ee,em,mm,e3j,e4pj,m3j,m4pj", required = False)
    parser.add_argument("--year", help = "analysis year determining the correlation model to assume. comma separated", default = "2016pre,2016post,2017,2018", required = False)

    parser.add_argument("--tag", help = "extra tag to be put on datacard names", default = "", required = False)
    parser.add_argument("--drop",
                        help = "comma separated list of nuisances to be dropped in datacard mode. 'XX, YY' means all sources containing XX or YY are dropped. '*' to drop all",
                        default = "", required = False)
    parser.add_argument("--keep",
                        help = "comma separated list of nuisances to be kept in datacard mode. same syntax as --drop. implies everything else is dropped",
                        default = "", required = False)
    parser.add_argument("--sushi-kfactor", help = "apply nnlo kfactors computing using sushi on A/H signals",
                        dest = "kfactor", action = "store_true", required = False)

    parser.add_argument("--threshold", help = "threshold under which nuisances that are better fit by a flat line are dropped/assigned as lnN",
                        default = 0.005, required = False, type = float)
    parser.add_argument("--lnN-under-threshold", help = "assign as lnN nuisances as considered by --threshold",
                        dest = "lnNsmall", action = "store_true", required = False)
    parser.add_argument("--use-shape-always", help = "use lowess-smoothened shapes even if the flat fit chi2 is better",
                        dest = "alwaysshape", action = "store_true", required = False)
    parser.add_argument("--no-mc-stats", help = "don't add nuisances due to limited mc stats (barlow-beeston lite)",
                        dest = "mcstat", action = "store_false", required = False)

    parser.add_argument("--use-pseudodata", help = "don't read the data from file, instead construct pseudodata using poisson-varied sum of backgrounds",
                        dest = "pseudodata", action = "store_true", required = False)
    parser.add_argument("--inject-signal",
                        help = "signal points to inject into the pseudodata, comma separated", dest = "injectsignal", default = "", required = False)
    parser.add_argument("--projection",
                        help = "instruction to project multidimensional histograms, assumed to be unrolled such that dimension d0 is presented "
                        "in slices of d1, which is in turn in slices of d2 and so on. the instruction is in the following syntax:\n"
                        "[instruction 0]:[instruction 1]:...:[instruction n] for n different types of templates.\n"
                        "each instruction has the following syntax: c0,c1,...,cn;b0,b1,...,bn;t0,t1,tm with m < n, where:\n"
                        "ci are the channels the instruction is applicable to, bi are the number of bins along each dimension, ti is the target projection index.\n"
                        "e.g. a channel ll with 3D templates of 20 x 3 x 3 bins, to be projected into the first dimension: ll;20,3,3;0 "
                        "or a projection into 2D templates along 2nd and 3rd dimension: ll;20,3,3;1,2\n"
                        "indices are zero-based, and spaces are ignored. relevant only in datacard/workspace mode.",
                        default = "", required = False)

    parser.add_argument("--seed",
                        help = "random seed to be used for pseudodata generation. give 0 to read from machine, and negative values to use no rng",
                        default = "", required = False)

    parser.add_argument("--unblind", help = "use data when fitting", dest = "asimov", action = "store_false", required = False)
    parser.add_argument("--one-poi", help = "use physics model with only g as poi", dest = "onepoi", action = "store_true", required = False)

    parser.add_argument("--raster-n", help = "number of chunks to split the g raster limit scan into",
                        dest = "nchunk", default = 6, required = False, type = int)
    parser.add_argument("--raster-i", help = "which chunk to process, in doing the raster scan",
                        dest = "ichunk", default = 0, required = False, type = int)

    #parser.add_argument("--impact-sb", help = "do sb pull/impact fit instead of b. "
    #                    "also used in prepost/corrmat/nll for the scenario to consider when finding best fit for freezing nuisances",
    #                    dest = "impactsb", action = "store_true", required = False)
    parser.add_argument("--impact-nuisances", help = "format: grp;n1,n2,...,nN where grp is the name of the group of nuisances, "
                        "and n1,n2,...,nN are the nuisances belonging to that group",
                        dest = "impactnui", default = "", required = False)

    parser.add_argument("--freeze-mc-stats-zero",
                        help = "only in the pull/impact/prepost/corrmat/nll mode, freeze mc stats nuisances to zero",
                        dest = "frzbb0", action = "store_true", required = False)
    parser.add_argument("--freeze-mc-stats-post",
                        help = "only in the pull/impact/prepost/corrmat/nll mode, freeze mc stats nuisances to the postfit values. "
                        "--freeze-mc-stats-zero takes priority over this option",
                        dest = "frzbbp", action = "store_true", required = False)
    parser.add_argument("--freeze-nuisance-post", help = "only in the prepost/corrmat/nll mode, freeze all nuisances to the postfit values.",
                        dest = "frznui", action = "store_true", required = False)

    parser.add_argument("--g-value",
                        help = "g to use when evaluating impacts/fit diagnostics/nll. "
                        "does NOT freeze the value, unless --fix-poi is also used. "
                        "note: semantically sets value of 'r' with --one-poi, as despite the name it plays the role of g.",
                        dest = "setg", default = -1., required = False, type = float)
    parser.add_argument("--r-value",
                        help = "r to use when evaluating impacts/fit diagnostics/nll, if --one-poi is not used."
                        "does NOT freeze the value, unless --fix-poi is also used.",
                        dest = "setr", default = -1., required = False, type = float)
    parser.add_argument("--fix-poi", help = "fix pois in the fit, through --g-value and/or --r-value",
                        dest = "fixpoi", action = "store_true", required = False)

    parser.add_argument("--compress", help = "compress output into a tar file", dest = "compress", action = "store_true", required = False)
    parser.add_argument("--base-directory",
                        help = "in non-datacard modes, this is the location where datacard is searched for, and output written to",
                        dest = "basedir", default = "", required = False)

    args = parser.parse_args()
    print "single_point_ahtt :: called with the following arguments"
    print sys.argv[1:]
    print "\n"
    print " ".join(sys.argv)
    print "\n"
    sys.stdout.flush()

    if (args.tag != "" and not args.tag.startswith("_")):
        args.tag = "_" + args.tag

    if args.injectsignal != "":
        args.pseudodata = True

    modes = args.mode.replace(" ", "").split(',')
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    args.basedir += "" if args.basedir == "" or args.basedir.endswith("/") else "/"
    dcdir = args.basedir + args.point + args.tag + "/"

    point = get_point(args.point)
    mstr = str(point[1]).replace(".0", "")
    poi_range = "--setParameterRanges r=0.,5." if args.onepoi else "--setParameterRanges r=0.,2.:g=0.,5."
    best_fit_file = ""

    allmodes = ["datacard", "workspace", "validate", "limit", "pull", "impact", "prepost", "corrmat", "nll", "likelihood"]
    if (not all([mm in allmodes for mm in modes])):
        print "supported modes:", allmodes
        raise RuntimeError("unxpected mode is given. aborting.")

    # determine what to do with workspace, and do it
    rundc = "datacard" in modes or "workspace" in modes
    runvalid = "validate" in modes
    runlimit = "limit" in modes
    runpull = "pull" in modes or "impact" in modes
    runprepost = "prepost" in modes or "corrmat" in modes
    runnll = "nll" in modes or "likelihood" in modes

    if rundc:
        print "\nsingle_point_ahtt :: making datacard"
        syscall("{scr}/make_datacard.py --signal {sig} --background {bkg} --point {pnt} --channel {ch} --year {yr} "
                "{psd} {inj} {tag} {drp} {kee} {kfc} {thr} {lns} {shp} {mcs} {prj} {rsd}".format(
                    scr = scriptdir,
                    sig = args.signal,
                    bkg = args.background,
                    pnt = args.point,
                    ch = args.channel,
                    yr = args.year,
                    psd = "--use-pseudodata" if args.pseudodata else "",
                    inj = "--inject-signal " + args.injectsignal if args.injectsignal != "" else "",
                    tag = "--tag " + args.tag if args.tag != "" else "",
                    drp = "--drop '" + args.drop + "'" if args.drop != "" else "",
                    kee = "--keep '" + args.keep + "'" if args.keep != "" else "",
                    kfc = "--sushi-kfactor" if args.kfactor else "",
                    thr = "--threshold " + str(args.threshold) if args.threshold != 0.005 else "",
                    lns = "--lnN-under-threshold" if args.lnNsmall else "",
                    shp = "--use-shape-always" if args.alwaysshape else "",
                    mcs = "--no-mc-stats" if not args.mcstat else "",
                    prj = "--projection '" + args.projection + "'" if args.projection != "" else "",
                    rsd = "--seed " + args.seed if args.seed != "" else ""
                ))

        print "\nsingle_point_ahtt :: making workspaces"
        for onepoi in [True, False]:
            syscall("combineTool.py -M T2W -i {dcd} -o workspace_{mod}.root -m {mmm} -P CombineHarvester.CombineTools.{phy}".format(
                dcd = dcdir + "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else dcdir + "ahtt_" + args.channel + '_' + args.year + ".txt",
                mod = "one-poi" if onepoi else "g-scan",
                mmm = mstr,
                phy = "InterferenceModel:interferenceModel" if onepoi else "InterferencePlusFixed:interferencePlusFixed"
            ))

    if runvalid:
        print "\nsingle_point_ahtt :: validating datacard"
        syscall("ValidateDatacards.py --jsonFile {dcd}{pnt}_validate.json --printLevel 3 {dcd}{crd}".format(
            dcd = dcdir,
            pnt = args.point,
            crd = "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else "ahtt_" + args.channel + '_' + args.year + ".txt"
        ))

    if runlimit:
        print "\nsingle_point_ahtt :: computing limit"
        accuracies = '--rRelAcc 0.005 --rAbsAcc 0'
        strategy = "--cminPreScan --cminDefaultMinimizerAlgo Migrad --cminDefaultMinimizerStrategy 1 --cminFallbackAlgo Minuit2,Simplex,1"

        if args.onepoi:
            syscall("rm {dcd}{pnt}_limits_one-poi.root {dcd}{pnt}_limits_one-poi.json".format(dcd = dcdir, pnt = args.point), False, True)
            syscall("combineTool.py -M AsymptoticLimits -d {dcd}workspace_one-poi.root -m {mmm} -n _limit {prg} {acc} {stg} {asm} {mcs}".format(
                dcd = dcdir,
                mmm = mstr,
                maxg = max_g,
                prg = poi_range,
                acc = accuracies,
                stg = strategy,
                asm = "--run blind -t -1" if args.asimov else "",
                mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else ""
            ))

            print "\nsingle_point_ahtt :: collecting limit"
            syscall("combineTool.py -M CollectLimits higgsCombine_limit.AsymptoticLimits.mH*.root -m {mmm} -o {dcd}{pnt}_limits_one-poi.json && "
                    "rm higgsCombine_limit.AsymptoticLimits.mH*.root".format(
                        dcd = dcdir,
                        mmm = mstr,
                        pnt = args.point
                    ))
        else:
            if args.nchunk < 0:
                args.nchunk = 6
            if args.ichunk < 0 or args.ichunk >= args.nchunk:
                args.ichunk = 0

            syscall("rm {dcd}{pnt}_limits_g-scan_{nch}_{idx}.root {dcd}{pnt}_limits_g-scan_{nch}_{idx}.json ".format(
                dcd = dcdir,
                pnt = args.point,
                nch = "n" + str(args.nchunk),
                idx = "i" + str(args.ichunk)), False, True)
            syscall("rm higgsCombine_limit_g-scan_*POINT.1.*AsymptoticLimits*.root", False, True)
            limits = OrderedDict()

            gvals = chunks(list(np.linspace(min_g, max_g, num = 193)), args.nchunk)[args.ichunk]
            lll = dotty_scan((gvals, dcdir, mstr, accuracies, poi_range, strategy, args.asimov, args.mcstat))

            print "\nsingle_point_ahtt :: collecting limit"
            print "\nthe following points have been processed:"
            print gvals

            for ll in lll:
                if ll is not None:
                    limits[ll[0]] = ll[1]
            limits = OrderedDict(sorted(limits.items()))

            syscall("hadd {dcd}{pnt}_limits_g-scan_{nch}_{idx}.root higgsCombine_limit_g-scan_*POINT.1.AsymptoticLimits*.root && "
                    "rm higgsCombine_limit_g-scan_*POINT.1.*AsymptoticLimits*.root".format(
                        dcd = dcdir,
                        pnt = args.point,
                        nch = "n" + str(args.nchunk),
                        idx = "i" + str(args.ichunk)
                    ))
            with open("{dcd}{pnt}_limits_g-scan_{nch}_{idx}.json".format(dcd = dcdir, pnt = args.point, nch = "n" + str(args.nchunk), idx = "i" + str(args.ichunk)), "w") as jj:
                json.dump(limits, jj, indent = 1)

    if runpull:
        syscall("rm {dcd}{pnt}_impacts_{mod}{gvl}{rvl}{fix}*".format(
            dcd = dcdir,
            pnt = args.point,
            mod = "one-poi" if args.onepoi else "g-scan",
            gvl = "_g_" + str(args.setg) if args.setg >= 0. else "",
            rvl = "_r_" + str(args.setr) if args.setr >= 0. and not args.onepoi else "",
            fix = "_fixed" if args.fixpoi else ""
        ), False, True)

        syscall("rm higgsCombine*Fit__pull*.root", False, True)
        syscall("rm combine_logger.out", False, True)

        if not args.onepoi and not (args.setg >= 0. and args.fixpoi):
            raise RuntimeError("it is unknown if impact works correctly with the g-scan model when g is left floating. please freeze it.")

        strategy = "--cminPreScan --cminDefaultMinimizerAlgo Migrad --cminDefaultMinimizerStrategy 1 --cminFallbackAlgo Minuit2,Simplex,1 --robustFit 1 --setRobustFitStrategy 1"

        if args.frzbbp:
            best_fit_file = make_best_fit(dcdir, "workspace_{mod}.root".format(mod = "one-poi" if args.onepoi else "g-scan"), args.point,
                                          args.asimov, args.mcstat, strategy, poi_range,
                                          elementwise_add(starting_poi(args.onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.point, args.frzbb0)))

        args.mcstat = args.mcstat or args.frzbb0 or args.frzbbp
        set_freeze = elementwise_add(starting_poi(args.onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.point, args.frzbb0, args.frzbbp, False, best_fit_file))
        setpar = set_freeze[0]
        frzpar = set_freeze[1]

        print "\nsingle_point_ahtt :: impact initial fit"
        syscall("combineTool.py -M Impacts -d {dcd}workspace_{mod}.root -m {mmm} --doInitialFit -n _pull {stg} {prg} {asm} {mcs} {stp} {frz}".format(
            dcd = dcdir,
            mod = "one-poi" if args.onepoi else "g-scan",
            mmm = mstr,
            prg = poi_range,
            stg = strategy,
            asm = "-t -1" if args.asimov else "",
            mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else "",
            stp = "--setParameters '" + ",".join(setpar) + "'" if len(setpar) > 0 else "",
            frz = "--freezeParameters '" + ",".join(frzpar) + "'" if len(frzpar) > 0 else ""
        ))

        group = ""
        nuisances = ""
        if args.impactnui != "":
            group = "_" + args.impactnui.replace(" ", "").split(';')[0]
            nuisances = args.impactnui.replace(" ", "").split(';')[1]

        print "\nsingle_point_ahtt :: impact remaining fits"
        syscall("combineTool.py -M Impacts -d {dcd}workspace_{mod}.root -m {mmm} --doFits -n _pull {stg} {prg} {asm} {mcs} {nui} {stp} {frz}".format(
            dcd = dcdir,
            mod = "one-poi" if args.onepoi else "g-scan",
            mmm = mstr,
            prg = poi_range,
            stg = strategy,
            asm = "-t -1" if args.asimov else "",
            mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else "",
            nui = "--named '" + nuisances + "'" if args.impactnui != "" else "",
            stp = "--setParameters '" + ",".join(setpar) + "'" if len(setpar) > 0 else "",
            frz = "--freezeParameters '" + ",".join(frzpar) + "'" if len(frzpar) > 0 else ""
        ))

        syscall("rm {dcd}{pnt}_impacts_{mod}{gvl}{rvl}{fix}*".format(
            dcd = dcdir,
            pnt = args.point,
            mod = "one-poi" if args.onepoi else "g-scan",
            gvl = "_g_" + str(args.setg) if args.setg >= 0. else "",
            rvl = "_r_" + str(args.setr) if args.setr >= 0. and not args.onepoi else "",
            fix = "_fixed" if args.fixpoi else ""
        ), False, True)

        print "\nsingle_point_ahtt :: collecting impact results"
        syscall("combineTool.py -M Impacts -d {dcd}workspace_{mod}.root -m {mmm} -n _pull -o {dcd}{pnt}_impacts_{mod}{gvl}{rvl}{fix}{grp}.json".format(
            dcd = dcdir,
            mod = "one-poi" if args.onepoi else "g-scan",
            mmm = mstr,
            pnt = args.point,
            gvl = "_g_" + str(args.setg) if args.setg >= 0. else "",
            rvl = "_r_" + str(args.setr) if args.setr >= 0. and not args.onepoi else "",
            fix = "_fixed" if args.fixpoi else "",
            grp = group
        ))

        syscall("rm higgsCombine*Fit__pull*.root", False, True)
        syscall("rm combine_logger.out", False, True)

        syscall("plotImpacts.py -i {dcd}{pnt}_impacts_{mod}{gvl}{rvl}{fix}{grp}.json -o {dcd}{pnt}_impacts_{mod}{gvl}{rvl}{fix}{grp}".format(
            dcd = dcdir,
            pnt = args.point,
            mod = "one-poi" if args.onepoi else "g-scan",
            gvl = "_g_" + str(args.setg) if args.setg >= 0. else "",
            rvl = "_r_" + str(args.setr) if args.setr >= 0. and not args.onepoi else "",
            fix = "_fixed" if args.fixpoi else "",
            grp = group
        ))

    if runprepost:
        # option '--set/freezeParameters' cannot be specified more than once
        strategy = "--cminPreScan --cminDefaultMinimizerStrategy 2 --cminFallbackAlgo Minuit2,Simplex,2  --robustFit 1 --setRobustFitStrategy 2 --robustHesse 1"

        if args.frzbbp or args.frznui:
            best_fit_file = make_best_fit(dcdir, "workspace_{mod}.root".format(mod = "one-poi" if args.onepoi else "g-scan"), args.point,
                                          args.asimov, args.mcstat, strategy, poi_range,
                                          elementwise_add(starting_poi(args.onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.point, args.frzbb0)))

        args.mcstat = args.mcstat or args.frzbb0 or args.frzbbp
        set_freeze = elementwise_add(starting_poi(args.onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.point, args.frzbb0, args.frzbbp, args.frznui, best_fit_file))
        setpar = set_freeze[0]
        frzpar = set_freeze[1]

        print "\nsingle_point_ahtt :: making pre- and postfit plots and covariance matrices"
        syscall("combine -v -1 -M FitDiagnostics {dcd}workspace_{mod}.root --saveWithUncertainties --saveNormalizations --saveShapes --saveOverallShapes "
                "--plots -m {mmm} -n _prepost {stg} {prg} {asm} {mcs} {stp} {frz}".format(
                    dcd = dcdir,
                    mod = "one-poi" if args.onepoi else "g-scan",
                    mmm = mstr,
                    stg = strategy,
                    prg = poi_range,
                    asm = "-t -1" if args.asimov else "",
                    mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else "",
                    stp = "--setParameters '" + ",".join(setpar) + "'" if len(setpar) > 0 else "",
                    frz = "--freezeParameters '" + ",".join(frzpar) + "'" if len(frzpar) > 0 else "",
        ))

        syscall("rm *_th1x_*.png", False, True)
        syscall("rm covariance_fit_?.png", False, True)
        syscall("rm higgsCombine_prepost*.root", False, True)
        syscall("rm combine_logger.out", False, True)
        syscall("rm robustHesse_*.root", False, True)

        syscall("mv fitDiagnostics_prepost.root {dcd}{pnt}_fitdiagnostics_{mod}{gvl}{rvl}{fix}.root".format(
            dcd = dcdir,
            pnt = args.point,
            gvl = "_g_" + str(args.setg) if args.setg >= 0. else "",
            rvl = "_r_" + str(args.setr) if args.setr >= 0. and not args.onepoi else "",
            fix = "_fixed" if args.fixpoi else "",
        ), False)

    if runnll:
        print "\nsingle_point_ahtt :: calculating nll as a function of gA/H"
        strategy = "--cminPreScan --cminDefaultMinimizerAlgo Migrad --cminDefaultMinimizerStrategy 1 --cminFallbackAlgo Minuit2,Simplex,1 --robustFit 1 --setRobustFitStrategy 1"

        if args.frzbbp or args.frznui:
            best_fit_file = make_best_fit(dcdir, "workspace_{mod}.root".format(mod = "one-poi" if args.onepoi else "g-scan"), args.point,
                                          args.asimov, args.mcstat, strategy, poi_range,
                                          elementwise_add(starting_poi(args.onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.point, args.frzbb0)))

        args.mcstat = args.mcstat or args.frzbb0 or args.frzbbp
        set_freeze = starting_nuisance(args.point, args.frzbb0, args.frzbbp, args.frznui, best_fit_file)
        setpar = set_freeze[0]
        frzpar = set_freeze[1]

        gvalues = [2.**-17, 2.**-15, 2.**-13, 2.**-11, 2.**-9] + list(np.linspace(min_g, max_g, num = 193))
        gvalues.sort()
        scenarii = ['exp-b', 'exp-s', 'obs']
        nlls = OrderedDict()

        syscall("rm {dcd}{pnt}_nll_one-poi.root {dcd}{pnt}_nll_{mod}.json".format(dcd = dcdir, pnt = args.point, mod = "one-poi" if args.onepoi else "g-scan"), False, True)

        for sce in scenarii:
            asimov = "-t -1" if sce != "obs" else ""
            pois = []
            if sce != "obs":
                if args.onepoi:
                    pois = ["r=0"] if sce == "exp-b" else ["r=1"]
                else:
                    pois = ["r=0", "g=0"] if sce == "exp-b" else ["r=1", "g=1"]

            if args.onepoi:
                syscall("combineTool.py -v -1 -M MultiDimFit --algo grid -d {dcd}workspace_one-poi.root -m {mmm} -n _nll {prg} {gvl} "
                        "--saveNLL --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 {stg} {asm} {stp} {frz} {mcs}".format(
                            dcd = dcdir,
                            mmm = mstr,
                            prg = poi_range,
                            gvl = "--points 193 --alignEdges 1",
                            stg = strategy,
                            asm = asimov,
                            stp = "--setParameters '" + ",".join(setpar + pois) + "'" if len(setpar) + len(pois) > 0 else "",
                            frz = "--freezeParameters '" + ",".join(frzpar) + "'" if len(frzpar) > 0 else "",
                            mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else ""
                        ))

                syscall("mv higgsCombine_nll.MultiDimFit.mH*.root {dcd}{pnt}_nll_{sce}_one-poi.root".format(
                    dcd = dcdir,
                    sce = sce,
                    pnt = args.point
                ), False)

                nlls[sce] = get_nll_one_poi("{dcd}{pnt}_nll_{sce}_one-poi.root".format(
                    dcd = dcdir,
                    sce = sce,
                    pnt = args.point
                ))
            else:
                for gval in gvalues:
                    gstr = str(round(gval, 3)).replace('.', 'p')
                    syscall("combineTool.py -v -1 -M MultiDimFit --algo fixed -d {dcd}workspace_g-scan.root -m {mmm} -n _nll_{gst} {prg} "
                            "--saveNLL --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 --fixedPointPOIs r=1,g={gvl} {stg} {asm} {stp} {frz} {mcs}".format(
                                dcd = dcdir,
                                mmm = mstr,
                                gst = "fix-g_" + str(gstr).replace(".", "p"),
                                prg = poi_range,
                                gvl = str(gval),
                                stg = strategy,
                                asm = asimov,
                                stp = "--setParameters '" + ",".join(setpar + pois) + "'" if len(setpar) + len(pois) > 0 else "",
                                frz = "--freezeParameters '" + ",".join(frzpar) + "'" if len(frzpar) > 0 else "",
                                mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else "",
                            ))

                syscall("hadd {dcd}{pnt}_nll_{sce}_g-scan.root higgsCombine_nll_fix-g*.root && "
                        "rm higgsCombine_nll_fix-g*.root".format(
                            dcd = dcdir,
                            sce = sce,
                            pnt = args.point
                        ))

                nlls[sce] = get_nll_g_scan("{dcd}{pnt}_nll_{sce}_g-scan.root".format(
                    dcd = dcdir,
                    sce = sce,
                    pnt = args.point
                ))

        syscall("rm combine_logger.out", False, True)

        with open("{dcd}{pnt}_nll_{mod}.json".format(dcd = dcdir, pnt = args.point, mod = "one-poi" if args.onepoi else "g-scan"), "w") as jj:
            json.dump(nlls, jj, indent = 1)

    if args.compress:
        syscall(("tar -czf {dcd}.tar.gz {dcd} && rm -r {dcd}").format(
            dcd = dcdir[:-1]
        ))
