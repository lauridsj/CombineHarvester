#!/usr/bin/env python
# sets the model-independent limit on a single A/H signal point

from argparse import ArgumentParser
import os
import sys
import numpy as np

from collections import OrderedDict
import json

from ROOT import TFile, TTree

from utilities import syscall
from make_datacard import get_point

max_g = 3.
epsilon = 1e-5
nstep = 5

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

# compares the estimated second derivative using the last 3 points in xys
# and the second derivative using the last 2 points and the current point
# true if these two quantities are similar enough based on an arbitrarily defined criteria
def seemingly_continuous(x, y, xys, tolerance = 0.1):
    if len(xys) < 3:
        return True

    if len(y) != 6:
        return False

    if tolerance <= 0. or tolerance >= 1.:
        tolerance = 0.1

    xs = []
    ys = []
    for ii in range(1, 4):
        xs.append(xys.keys()[-ii])
        ys.append(xys[xs[-1]])

    dx_previous = 0.5 * (xs[0] - xs[2])
    dx_current = 0.5 * (x - xs[1])
    d2ydx2_previous = ys[0]
    d2ydx2_current = ys[0]

    for quantile in d2ydx2_previous.keys():
        if quantile == "obs":
            continue

        d2ydx2_previous[quantile] = (ys[0][quantile] - (2. * ys[1][quantile]) + ys[2][quantile]) / dx_previous**2.
        d2ydx2_current[quantile] = (y[quantile] - (2. * ys[0][quantile]) + ys[1][quantile]) / dx_current**2.

        fom = abs(d2ydx2_current[quantile] / d2ydx2_previous[quantile])
        if fom < (1. / (1. + tolerance)) or fom > (1. / (1. - tolerance)):
            return False

    return True

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "desired signal point to run on", default = "", required = True)
    parser.add_argument("--mode", help = "combine mode to run, comma separated", default = "datacard", required = False)

    parser.add_argument("--signal", help = "signal filenames. comma separated", default = "../input/ll_sig.root", required = False)
    parser.add_argument("--background", help = "data/background filenames. comma separated", default = "../input/ll_bkg.root", required = False)
    parser.add_argument("--channel", help = "final state channels considered in the analysis. comma separated", default = "ll", required = False)
    parser.add_argument("--year", help = "analysis year determining the correlation model to assume. comma separated", default = "2018", required = False)
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

    parser.add_argument("--use-pseudodata", help = "don't read the data from file, instead construct pseudodata using poisson-varied sum of backgrounds",
                        dest = "pseudodata", action = "store_true", required = False)
    parser.add_argument("--inject-signal", help = "signal point to inject into the pseudodata", dest = "injectsignal", default = "", required = False)
    parser.add_argument("--no-mc-stats", help = "don't add nuisances due to limited mc stats (barlow-beeston lite)",
                        dest = "mcstat", action = "store_false", required = False)
    parser.add_argument("--projection",
                        help = "instruction to project multidimensional histograms, assumed to be unrolled such that dimension d0 is presented "
                        "in slices of d1, which is in turn in slices of d2 and so on. the instruction is in the following syntax:\n"
                        "[instruction 0]:[instruction 1]:...:[instruction n] for n different types of templates.\n"
                        "each instruction has the following syntax: c0,c1,...,cn;b0,b1,...,bn;t0,t1,tm with m < n, where:\n"
                        "ci are the channels the instruction is applicable to, bi are the number of bins along each dimension, ti is the target projection index.\n"
                        "e.g. a channel ll with 3D templates of 20 x 3 x 3 bins, to be projected into the first dimension: ll;20,3,3;0 "
                        "or a projection into 2D templates alone 2nd and 3rd dimension: ll;20,3,3;1,2\n"
                        "indices are zero-based, and spaces are ignored. relevant only in datacard/workspace mode.",
                        default = "", required = False)
    parser.add_argument("--freeze-mc-stats-zero", help = "only in the prepost/corrmat mode, freeze mc stats nuisances to zero",
                        dest = "frzbb0", action = "store_true", required = False)
    parser.add_argument("--freeze-mc-stats-post", help = "only in the prepost/corrmat mode, freeze mc stats nuisances to the postfit values. "
                        "requires pull/impact to have been run",
                        dest = "frzbbp", action = "store_true", required = False)
    parser.add_argument("--seed",
                        help = "random seed to be used for pseudodata generation. give 0 to read from machine, and negative values to use no rng",
                        default = "", required = False)

    parser.add_argument("--unblind", help = "use data when fitting", dest = "asimov", action = "store_false", required = False)
    parser.add_argument("--one-poi", help = "use physics model with only g as poi", dest = "onepoi", action = "store_true", required = False)

    parser.add_argument("--impact-sb", help = "do sb pull/impact fit instead of b. "
                        "also used in prepost/corrmat o steer which pull to be read with --freeze-mc-stats-post",
                        dest = "impactsb", action = "store_true", required = False)
    parser.add_argument("--g-value", help = "g value to use when evaluating impacts/fit diagnostics, if one-poi is not used. defaults to 1",
                        dest = "fixg", default = 1, required = False, type = float)

    parser.add_argument("--compress", help = "compress output into a tar file", dest = "compress", action = "store_true", required = False)
    parser.add_argument("--base-directory",
                        help = "in non-datacard modes, this is the location where datacard is searched for, and output written to",
                        dest = "basedir", default = "", required = False)

    args = parser.parse_args()
    print "single_point_ahtt :: called with the following arguments"
    print sys.argv[1:]
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

    allmodes = ["datacard", "workspace", "validate", "limit", "pull", "impact", "prepost", "corrmat"]
    if (not all([mm in allmodes for mm in modes])):
        print "supported modes:", allmodes
        raise RuntimeError("unxpected mode is given. aborting.")

    # determine what to do with workspace, and do it
    rundc = "datacard" in modes or "workspace" in modes
    runvalid = "validate" in modes
    runlimit = "limit" in modes
    runpull = "pull" in modes or "impact" in modes
    runprepost = "prepost" in modes or "corrmat" in modes
    # combine --rMin=0 --rMax=2.5 --cminPreScan -t -1 --saveNLL --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 -M MultiDimFit --algo grid -d A400_relw2p964/workspace_one-poi.root -n lolk # to check absolute NLL curve (seems to work with one-poi only)
    # combine -M MultiDimFit A_m400_w2p5_3D-33_ul_split_smooth_mtt/workspace_g-scan.root --setParameters g=1 --freezeParameters g --redefineSignalPOIs r --setParameterRanges r=0,1.5 --algo grid --points 50 -P r --robustFit 1 --cminPreScan --X-rtd MINIMIZER_analytic --cminDefaultMinimizerStrategy 0

    if rundc:
        print "\nsingle_point_ahtt :: making datacard"
        syscall("{scr}/make_datacard.py --signal {sig} --background {bkg} --point {pnt} --channel {ch} --year {yr} "
                "{psd} {inj} {tag} {drp} {kfc} {thr} {lns} {shp} {mcs} {prj} {rsd}".format(
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
        strategy = "--cminPreScan --cminFallbackAlgo Minuit2,Simplex,1"

        if args.onepoi:
            syscall("rm {dcd}{pnt}_limits_one-poi.root {dcd}{pnt}_limits_one-poi.json".format(dcd = dcdir, pnt = args.point), False, True)
            syscall("combineTool.py -M AsymptoticLimits -d {dcd}workspace_one-poi.root -m {mmm} -n _limit --rMin=0 --rMax={maxg} {acc} {stg} {asm} {mcs}".format(
                        dcd = dcdir,
                        mmm = mstr,
                        maxg = max_g,
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
            syscall("rm {dcd}{pnt}_limits_g-scan.root {dcd}{pnt}_limits_g-scan.json ".format(dcd = dcdir, pnt = args.point), False, True)
            syscall("rm higgsCombine_limit_g-scan_*POINT.1.*AsymptoticLimits*.root", False, True)
            limits = OrderedDict()
            gval = 0.
            r_range = "--rMin=0 --rMax=2"
            consecutive_failure = 0

            while gval < max_g:
                gstr = str(round(gval, 3)).replace('.', 'p')

                syscall("combineTool.py -M AsymptoticLimits -d {dcd}workspace_g-scan.root -m {mmm} -n _limit_g-scan_{gstr} "
                        "--setParameters g={gval} --freezeParameters g {acc} --picky {rrg} "
                        "--singlePoint 1 {stg} {asm} {mcs}".format(
                            dcd = dcdir,
                            mmm = mstr,
                            gstr = gstr,
                            gval = gval,
                            acc = accuracies,
                            rrg = r_range,
                            stg = strategy,
                            asm = "-t -1" if args.asimov else "",
                            mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else ""
                        ))

                limit = get_limit("higgsCombine_limit_g-scan_{gstr}.POINT.1.AsymptoticLimits.mH{mmm}.root".format(
                    mmm = mstr,
                    gstr = gstr,
                ))

                good_cls = all([ll >= 0. and ll <= 1. for qq, ll in limit.items()])
                geps = 0.
                if not good_cls:
                    syscall("rm higgsCombine_limit_g-scan_{gstr}.POINT.1.*AsymptoticLimits*.root".format(gstr = gstr), False, True)

                    for factor in [1., -1.]:
                        fgood = False
                        for ii in range(1, nstep + 1):
                            syscall("combineTool.py -M AsymptoticLimits -d {dcd}workspace_g-scan.root -m {mmm} -n _limit_g-scan_{gstr} "
                                    "--setParameters g={gval} --freezeParameters g {acc} --picky {rrg} "
                                    "--singlePoint 1 {stg} {asm} {mcs}".format(
                                        dcd = dcdir,
                                        mmm = mstr,
                                        gstr = gstr + "eps",
                                        gval = gval + (ii * factor * epsilon),
                                        acc = accuracies,
                                        rrg = r_range,
                                        stg = strategy,
                                        asm = "-t -1" if args.asimov else "",
                                        mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else ""
                                    ))

                            leps = get_limit("higgsCombine_limit_g-scan_{gstr}.POINT.1.AsymptoticLimits.mH{mmm}.root".format(
                                mmm = mstr,
                                gstr = gstr + "eps",
                            ))
                            fgood = all([ll >= 0. and ll <= 1. for qq, ll in leps.items()])

                            if fgood:
                                geps = (ii * factor * epsilon)
                                for quantile in leps.keys():
                                    limit[quantile] = leps[quantile]
                                break
                        if fgood:
                            break

                good_cls = all([ll >= 0. and ll <= 1. for qq, ll in limit.items()])
                if good_cls:
                    limits[gval + geps] = limit
                    consecutive_failure = 0
                    if any([abs(ll - 0.05) < 0.02 for qq, ll in limit.items()]):
                        gval += 0.01
                    else:
                        gval += 0.05
                else:
                    gval += 0.01
                    consecutive_failure += 1
                    print "\nsingle_point_ahtt :: consecutive failure ", consecutive_failure
                    if consecutive_failure > 10:
                        break

            print "\nsingle_point_ahtt :: collecting limit"
            syscall("hadd {dcd}{pnt}_limits_g-scan.root higgsCombine_limit_g-scan_*POINT.1.AsymptoticLimits*.root && "
                    "rm higgsCombine_limit_g-scan_*POINT.1.*AsymptoticLimits*.root".format(
                        dcd = dcdir,
                        pnt = args.point
                    ))
            with open("{dcd}{pnt}_limits_g-scan.json".format(dcd = dcdir, pnt = args.point), "w") as jj: 
                json.dump(limits, jj, indent = 1)

    if runpull:
        syscall("rm {dcd}{pnt}_impacts_{mod}*".format(dcd = dcdir, mod = "one-poi" if args.onepoi else "g-scan", pnt = args.point), False, True)

        r_range = "--rMin=0 --rMax={maxg}".format(maxg = max_g if args.onepoi else "2")
        strategy = "--robustFit 1 --cminPreScan --cminDefaultMinimizerStrategy 0 --cminFallbackAlgo Minuit2,Simplex,0"

        syscall("rm higgsCombine*Fit__pull*.root", False, True)
        syscall("rm robustHesse*Fit__pull*.root", False, True)
        syscall("rm combine_logger.out", False, True)

        print "\nsingle_point_ahtt :: impact initial fit"
        syscall("combineTool.py -M Impacts -d {dcd}workspace_{mod}.root -m {mmm} --doInitialFit -n _pull {stg} {rrg} {poi} {asm} {mcs} {sig}".format(
                    dcd = dcdir,
                    mod = "one-poi" if args.onepoi else "g-scan",
                    mmm = mstr,
                    rrg = r_range,
                    stg = strategy,
                    poi = "" if args.onepoi else "--setParameters g=" + str(args.fixg) + " --freezeParameters g --redefineSignalPOIs r",
                    asm = "-t -1" if args.asimov else "",
                    mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else "",
                    sig = "--expectSignal 1" if args.impactsb else "--expectSignal 0"
                ))

        print "\nsingle_point_ahtt :: impact remaining fits"
        syscall("combineTool.py -M Impacts -d {dcd}workspace_{mod}.root -m {mmm} --doFits --parallel 8 -n _pull {stg} {rrg} {poi} {asm} {mcs} {sig}".format(
                    dcd = dcdir,
                    mod = "one-poi" if args.onepoi else "g-scan",
                    mmm = mstr,
                    rrg = r_range,
                    stg = strategy,
                    poi = "" if args.onepoi else "--setParameters g=" + str(args.fixg) + " --freezeParameters g --redefineSignalPOIs r",
                    asm = "-t -1" if args.asimov else "",
                    mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else "",
                    sig = "--expectSignal 1" if args.impactsb else "--expectSignal 0"
                ))

        print "\nsingle_point_ahtt :: collecting impact results"
        syscall("combineTool.py -M Impacts -d {dcd}workspace_{mod}.root -m {mmm} {poi} -n _pull -o {dcd}{pnt}_impacts_{gvl}_{exp}.json".format(
            dcd = dcdir,
            mod = "one-poi" if args.onepoi else "g-scan",
            mmm = mstr,
            poi = "" if args.onepoi else "--redefineSignalPOIs r",
            pnt = args.point,
            gvl = "one-poi" if args.onepoi else "fix-g_" + str(args.fixg).replace(".", "p"),
            exp = "sig" if args.impactsb else "bkg"
        ))

        syscall("rm higgsCombine*Fit__pull*.root", False, True)
        syscall("rm robustHesse*Fit__pull*.root", False, True)
        syscall("rm combine_logger.out", False, True)

        syscall("plotImpacts.py -i {dcd}{pnt}_impacts_{mod}_{exp}.json -o {dcd}{pnt}_impacts_{mod}_{exp}".format(
            dcd = dcdir,
            gvl = "one-poi" if args.onepoi else "fix-g_" + str(args.fixg).replace(".", "p"),
            pnt = args.point,
            exp = "sig" if args.impactsb else "bkg"
        ))

    if runprepost:
        # option '--set/freezeParameters' cannot be specified more than once
        strategy = "--robustFit 1 --robustHesse 1 --cminPreScan --cminDefaultMinimizerStrategy 2 --cminFallbackAlgo Minuit2,Simplex,2"
        setpar = []
        frzpar = []
        if not args.onepoi:
            setpar.append("g=" + str(args.fixg))
            frzpar.append("g")

        args.mcstat = args.mcstat or args.frzbb0 or args.frzbbp
        if args.frzbb0:
            setpar.append("rgx{prop_bin.*}=0")
            frzpar.append("rgx{prop_bin.*}")

        if args.frzbbp:
            frzpar.append("rgx{prop_bin.*}")
            iname = "{dcd}/{pnt}_impacts_{gvl}_{exp}.json".format(
                dcd = dcdir,
                pnt = args.point,
                gvl = "one-poi" if args.onepoi else "fix-g_" + str(args.fixg).replace(".", "p"),
                exp = "sig" if args.impactsb else "bkg"
            )

            with open(iname) as ff:
                impacts = json.load(ff)

            for pp in impacts["params"]:
                if "prop_bin" in pp["name"]:
                    setpar.append("{par}={val}".format(par = pp["name"], val = str(round(pp["fit"][1], 3) if abs(pp["fit"][1]) > 1e-3 else 0)))

        print "\nsingle_point_ahtt :: making pre- and postfit plots and covariance matrices"
        syscall("combine -v -1 -M FitDiagnostics {dcd}workspace_{mod}.root --saveWithUncertainties --saveNormalizations --saveShapes --saveOverallShapes "
                "--plots -m {mmm} -n _prepost {stg} {asm} {mcs} {frz} {poi}".format(
                    dcd = dcdir,
                    mod = "one-poi" if args.onepoi else "g-scan",
                    mmm = mstr,
                    stg = strategy,
                    asm = "-t -1" if args.asimov else "",
                    mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else "",
                    frz = "--setParameters '" + ",".join(setpar) + "' --freezeParameters '" + ",".join(frzpar) + "'" if len(setpar) > 0 else "",
                    poi = "" if args.onepoi else "--redefineSignalPOIs r"
        ))

        syscall("rm *_th1x_*.png", False, True)
        syscall("rm covariance_fit_?.png", False, True)
        syscall("rm higgsCombine_prepost*.root", False, True)
        syscall("rm combine_logger.out", False, True)
        syscall("mv fitDiagnostics_prepost.root {dcd}{pnt}_fitdiagnostics_{gvl}.root".format(
            dcd = dcdir,
            gvl = "one-poi" if args.onepoi else "fix-g_" + str(args.fixg).replace(".", "p"),
            pnt = args.point
        ), False)

    if args.compress:
        syscall(("tar -czf {dcd}.tar.gz {dcd} && rm -r {dcd}").format(
            dcd = dcdir[:-1]
        ))
