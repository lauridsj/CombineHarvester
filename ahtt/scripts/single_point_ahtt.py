#!/usr/bin/env python
# sets the model-independent limit on a single A/H signal point

from argparse import ArgumentParser
import os
import sys
import numpy as np

from collections import OrderedDict
import json

from ROOT import TFile, TTree

from make_datacard import syscall, get_point

max_g = 3.

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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "desired signal point to run on", default = "", required = True)
    parser.add_argument("--mode", help = "combine mode to run, comma separated", default = "datacard", required = False)

    parser.add_argument("--signal", help = "signal filename. comma separated", default = "../input/ll_sig.root", required = False)
    parser.add_argument("--background", help = "data/background. comma separated", default = "../input/ll_bkg.root", required = False)
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
    parser.add_argument("--freeze-mc-stats-zero", help = "only in the prepost/corrmat mode, freeze mc stats nuisances to zero",
                        dest = "frzbb0", action = "store_true", required = False)
    parser.add_argument("--seed",
                        help = "random seed to be used for pseudodata generation. give 0 to read from machine, and negative values to use no rng",
                        default = "", required = False)

    parser.add_argument("--unblind", help = "use data when fitting", dest = "asimov", action = "store_false", required = False)
    parser.add_argument("--one-poi", help = "use physics model with only g as poi", dest = "onepoi", action = "store_true", required = False)

    parser.add_argument("--impact-sb", help = "do sb impact fit instead of b", dest = "impactsb", action = "store_true", required = False)
    parser.add_argument("--g-value", help = "g value to use when evaluating impacts/fit diagnostics, if one-poi is not used",
                        dest = "fixg", default = 1, required = False, type = float)

    parser.add_argument("--compress", help = "compress output into a tar file", dest = "compress", action = "store_true", required = False)

    args = parser.parse_args()
    if (args.tag != "" and not args.tag.startswith("_")):
        args.tag = "_" + args.tag

    if args.injectsignal != "":
        args.pseudodata = True

    modes = args.mode.strip().split(',')
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    dcdir =  args.point + args.tag + "/"
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

    if rundc:
        print "\nsingle_point_ahtt :: making datacard"
        syscall("{scr}/make_datacard.py --signal {sig} --background {bkg} --point {pnt} --channel {ch} --year {yr} "
                "{psd} {inj} {tag} {drp} {kfc} {thr} {lns} {shp} {mcs} {rsd}".format(
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
        os.chdir(dcdir)
        print "\nsingle_point_ahtt :: validating datacard"
        syscall("ValidateDatacards.py --jsonFile {pnt}_validate.json --printLevel 3 {dcd}".format(
            pnt = args.point,
            dcd = "ahtt_combined.txt" if os.path.isfile("ahtt_combined.txt") else "ahtt_" + args.channel + '_' + args.year + ".txt"
        ))
        os.chdir("..")

    if runlimit:
        print "\nsingle_point_ahtt :: computing limit"
        if args.onepoi:
            syscall("combineTool.py -M AsymptoticLimits -d {dcd}workspace_one-poi.root -m {mmm} --there -n _limit --rMin=0 --rMax={maxg} "
                    "--rRelAcc 0.001 --rAbsAcc 0.001 --cminPreScan {asm} {mcs}".format(
                        dcd = dcdir,
                        mmm = mstr,
                        maxg = max_g,
                        asm = "--run blind -t -1" if args.asimov else "",
                        mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else ""
            ))

            print "\nsingle_point_ahtt :: collecting limit"
            syscall("combineTool.py -M CollectLimits {dcd}higgsCombine_limit.AsymptoticLimits.mH*.root -m {mmm} -o {dcd}{pnt}_limits_one-poi.json && "
                    "rm {dcd}higgsCombine_limit.AsymptoticLimits.mH*.root".format(
                        dcd = dcdir,
                        mmm = mstr,
                        pnt = args.point
            ))
        else:
            limits = OrderedDict()
            gval = 0.

            while gval < max_g:
                gstr = str(round(gval, 3)).replace('.', 'p')

                syscall("combineTool.py -M AsymptoticLimits -d {dcd}workspace_g-scan.root -m {mmm} --there -n _limit_g-scan_{gstr} --rMin=0 --rMax=3 "
                        "--setParameters g={gval} --freezeParameters g --rRelAcc 0.001 --rAbsAcc 0.001 --picky "
                        "--singlePoint 1 --cminPreScan {asm} {mcs}".format(
                            dcd = dcdir,
                            mmm = mstr,
                            gstr = gstr,
                            maxg = max_g,
                            gval = gval,
                            asm = "-t -1" if args.asimov else "",
                            mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else ""
                        ))

                limit = get_limit("{dcd}higgsCombine_limit_g-scan_{gstr}.POINT.1.AsymptoticLimits.mH{mmm}.root".format(
                    dcd = dcdir,
                    mmm = mstr,
                    gstr = gstr,
                ))

                limits[gval] = limit

                if (any([ll > 0.025 and ll < 0.125 for qq, ll in limit.items()])):
                    gval += 0.005
                elif (any([ll < 0. or ll > 1. for qq, ll in limit.items()])):
                    gval += 0.01
                else:
                    gval += 0.04

            print "\nsingle_point_ahtt :: collecting limit"
            syscall("hadd {dcd}{pnt}_limits_g-scan.root {dcd}higgsCombine_limit_g-scan_*POINT.1.AsymptoticLimits*.root && "
                    "rm {dcd}higgsCombine_limit_g-scan_*POINT.1.*AsymptoticLimits*.root".format(
                    dcd = dcdir,
                    pnt = args.point
            ))
            with open("{dcd}{pnt}_limits_g-scan.json".format(dcd = dcdir, pnt = args.point), "w") as jj: 
                json.dump(limits, jj, indent = 1)

    if runpull:
        os.chdir(dcdir)
        r_range = "--rMin=-3 --rMax=3"

        print "\nsingle_point_ahtt :: impact initial fit"
        syscall("combineTool.py -M Impacts -d workspace_{mod}.root -m {mmm} --cminPreScan --doInitialFit --robustFit 1 --robustHesse 1 -n _pull "
                "--cminDefaultMinimizerStrategy 0 {rrg} {com} {asm} {mcs} {sig}".format(
                    mod = "one-poi" if args.onepoi else "g-scan",
                    mmm = mstr,
                    rrg = r_range,
                    com = "" if args.onepoi else "--setParameters g=" + str(args.fixg) + " --freezeParameters g --redefineSignalPOIs r",
                    asm = "-t -1" if args.asimov else "",
                    mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else "",
                    sig = "--expectSignal 1" if args.impactsb else "--expectSignal 0"
                ))

        print "\nsingle_point_ahtt :: impact remaining fits"
        syscall("combineTool.py -M Impacts -d workspace_{mod}.root -m {mmm} --cminPreScan --doFits --parallel 4 --robustFit 1 --robustHesse 1 -n _pull "
                "--cminDefaultMinimizerStrategy 0 {rrg} {com} {asm} {mcs} {sig}".format(
                    mod = "one-poi" if args.onepoi else "g-scan",
                    mmm = mstr,
                    rrg = r_range,
                    com = "" if args.onepoi else "--setParameters g=" + str(args.fixg) + " --freezeParameters g --redefineSignalPOIs r",
                    asm = "-t -1" if args.asimov else "",
                    mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else "",
                    sig = "--expectSignal 1" if args.impactsb else "--expectSignal 0"
                ))

        print "\nsingle_point_ahtt :: collecting impact results"
        syscall("combineTool.py -M Impacts -d workspace_{mod}.root -m {mmm} {com} -n _pull -o {pnt}_impacts_{mod}.json".format(
            mod = "one-poi" if args.onepoi else "g-scan",
            mmm = mstr,
            com = "" if args.onepoi else "--redefineSignalPOIs r",
            pnt = args.point
        ))

        syscall("rm higgsCombine*Fit__pull*.root", False)
        syscall("rm robustHesse*Fit__pull*.root", False)
        syscall("rm combine_logger.out", False)

        syscall("plotImpacts.py -i {pnt}_impacts_{mod}.json -o {pnt}_impacts_{mod}".format(
            mod = "one-poi" if args.onepoi else "g-scan",
            pnt = args.point
        ))

        os.chdir("..")

    if runprepost:
        os.chdir(dcdir)

        setpar = []
        frzpar = []
        if args.onepoi:
            setpar.append("g=" + str(args.fixg))
            frzpar.append("g")

        if args.frzbb0:
            args.mcstat = True
            setpar.append("rgx{prop_bin.*}=0")
            frzpar.append("rgx{prop_bin.*}")

        print "\nsingle_point_ahtt :: making pre- and postfit plots and covariance matrices"
        syscall("combine -M FitDiagnostics workspace_{mod}.root --saveShapes --saveWithUncertainties --saveNormalizations --plots -m {mmm} -n _prepost "
                "--robustFit 1 --robustHesse 1 --cminPreScan --cminDefaultMinimizerStrategy 0 {asm} {mcs} {frz} {com}".format(
                    mod = "one-poi" if args.onepoi else "g-scan",
                    mmm = mstr,
                    asm = "-t -1" if args.asimov else "",
                    mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else "",
                    frz = "--setParameters '" + ",".join(setpar) + "' --freezeParameters '" + ",".join(frzpar) + "'" if len(setpar) > 0 else "",
                    com = "" if args.onepoi else "--redefineSignalPOIs r",
                    # FIXME an example, to be developed more into freezing at postfit if needed (needs impact to be run)
                    # btw option '--setParameters' cannot be specified more than once
        ))

        syscall("rm *_th1x_*.png", False)
        syscall("rm covariance_fit_?.png", False)
        syscall("rm higgsCombine_prepost*.root", False)
        syscall("rm combine_logger.out", False)
        syscall("mv fitDiagnostics_prepost.root {pnt}_fitdiagnostics_{mod}.root".format(
            mod = "one-poi" if args.onepoi else "g-scan",
            pnt = args.point
        ), False)

        os.chdir("..")

    if args.compress:
        syscall(("tar -czf {dcd}.tar.gz {dcd} && rm -r {dcd}").format(
            dcd = dcdir[:-1]
        ))
