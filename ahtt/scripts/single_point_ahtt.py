#!/usr/bin/env python
# sets the model-independent limit on a single A/H signal point

from argparse import ArgumentParser
import os
import sys
import numpy as np

from collections import OrderedDict
import json

from ROOT import TFile, TTree

from utilities import syscall, get_point, chunks, min_g, max_g, make_best_fit, starting_nuisance, elementwise_add, fit_strategy, make_datacard_with_args, set_parameter
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes
from argumentative import common_point, common_common, common_fit_pure, common_fit, make_datacard_pure, make_datacard_forwarded, common_1D
from hilfemir import combine_help_messages

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
    gval, dcdir, mstr, accuracies, poi_range, strategy, asimov, masks = args
    gstr = str(round(gval, 3)).replace('.', 'p')

    epsilon = 2.**-17
    nstep = 1

    syscall("combineTool.py -M AsymptoticLimits -d {dcd}workspace_g-scan.root -m {mmm} -n _limit_g-scan_{gst} "
            "--setParameters g={gvl} --freezeParameters g {acc} --picky {prg} "
            "--singlePoint 1 {stg} {asm} {msk}".format(
                dcd = dcdir,
                mmm = mstr,
                gst = gstr,
                gvl = gval,
                acc = accuracies,
                prg = poi_range,
                stg = strategy,
                asm = "-t -1" if asimov else "",
                msk = "--setParameters '" + ",".join(masks) + "'" if len(masks) > 0 else ""
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
                    "--singlePoint 1 {stg} {asm} {msk}".format(
                        dcd = dcdir,
                        mmm = mstr,
                        gst = gstr + "eps",
                        gvl = gval + (ii * factor * epsilon),
                        acc = accuracies,
                        prg = poi_range,
                        stg = strategy,
                        asm = "-t -1" if asimov else "",
                        msk = "--setParameters '" + ",".join(masks) + "'" if len(masks) > 0 else ""
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
    gvals, dcdir, mstr, accuracies, poi_range, strategy, asimov, masks = args
    if len(gvals) < 2:
        return None
    gvals = sorted(gvals)

    results = []
    ii = 0
    while ii < len(gvals):
        result = single_point_scan((gvals[ii], dcdir, mstr, accuracies, poi_range, strategy, asimov, masks))

        if result is None:
            ii += 3
            continue

        if (0.0125 < result[2] < 0.2):
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
    common_point(parser)
    common_common(parser)
    common_fit_pure(parser)
    common_fit(parser)
    make_datacard_pure(parser)
    make_datacard_forwarded(parser)
    common_1D(parser)

    parser.add_argument("--raster-i", help = combine_help_messages["--raster-i"], dest = "ichunk", default = 0, required = False, type = lambda s: int(remove_spaces_quotes(s)))
    parser.add_argument("--impact-nuisances", help = combine_help_messages["--impact-nuisances"], dest = "impactnui", default = "", required = False,
                        type = lambda s: None if s == "" else tokenize_to_list( remove_spaces_quotes(s), ';' ))

    args = parser.parse_args()
    print "single_point_ahtt :: called with the following arguments"
    print sys.argv[1:]
    print "\n"
    print " ".join(sys.argv)
    print "\n"
    sys.stdout.flush()

    if len(args.point) != 1:
        raise RuntimeError("this script is to be used with exactly one A/H point!")

    modes = args.mode
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    dcdir = args.basedir + args.point[0] + args.tag + "/"

    point = get_point(args.point[0])
    mstr = str(point[1]).replace(".0", "")
    poi_range = "--setParameterRanges 'r=0.,5.'" if args.onepoi else "--setParameterRanges 'r=0.,2.:g=0.,5.'"
    best_fit_file = ""
    masks = ["mask_" + mm + "=1" for mm in args.mask]
    print "the following channel x year combinations will be masked:", args.mask

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

    if args.otag == "":
        args.otag = args.tag

    if rundc:
        print "\nsingle_point_ahtt :: making datacard"
        make_datacard_with_args(scriptdir, args)

        print "\nsingle_point_ahtt :: making workspaces"
        for onepoi in [True, False]:
            syscall("combineTool.py -M T2W -i {dcd} -o workspace_{mod}.root -m {mmm} -P CombineHarvester.CombineTools.{phy} --channel-masks".format(
                dcd = dcdir + "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else dcdir + "ahtt_" + args.channel + '_' + args.year + ".txt",
                mod = "one-poi" if onepoi else "g-scan",
                mmm = mstr,
                phy = "InterferenceModel:interferenceModel" if onepoi else "InterferencePlusFixed:interferencePlusFixed"
            ))

    if runvalid:
        print "\nsingle_point_ahtt :: validating datacard"
        syscall("ValidateDatacards.py --jsonFile {dcd}{pnt}{tag}_validate.json --printLevel 3 {dcd}{crd}".format(
            dcd = dcdir,
            pnt = args.point[0],
            tag = args.otag,
            crd = "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else "ahtt_" + args.channel + '_' + args.year + ".txt"
        ))

    if runlimit:
        print "\nsingle_point_ahtt :: computing limit"
        accuracies = '--rRelAcc 0.005 --rAbsAcc 0'

        if args.onepoi:
            syscall("rm {dcd}{pnt}{tag}_limits_one-poi.root {dcd}{pnt}{tag}_limits_one-poi.json".format(dcd = dcdir, pnt = args.point[0], tag = args.otag), False, True)
            syscall("combineTool.py -M AsymptoticLimits -d {dcd}workspace_one-poi.root -m {mmm} -n _limit {prg} {acc} {stg} {asm} {msk}".format(
                dcd = dcdir,
                mmm = mstr,
                maxg = max_g,
                prg = poi_range,
                acc = accuracies,
                stg = fit_strategy("1"),
                asm = "--run blind -t -1" if args.asimov else "",
                msk = "--setParameters '" + ",".join(masks) + "'" if len(masks) > 0 else ""
            ))

            print "\nsingle_point_ahtt :: collecting limit"
            syscall("combineTool.py -M CollectLimits higgsCombine_limit.AsymptoticLimits.mH*.root -m {mmm} -o {dcd}{pnt}{tag}_limits_one-poi.json && "
                    "rm higgsCombine_limit.AsymptoticLimits.mH*.root".format(
                        dcd = dcdir,
                        mmm = mstr,
                        pnt = args.point[0],
                        tag = args.otag
                    ))
        else:
            if args.nchunk < 0:
                args.nchunk = 6
            if args.ichunk < 0 or args.ichunk >= args.nchunk:
                args.ichunk = 0

            syscall("rm {dcd}{pnt}{tag}_limits_g-scan_{nch}_{idx}.root {dcd}{pnt}{tag}_limits_g-scan_{nch}_{idx}.json ".format(
                dcd = dcdir,
                pnt = args.point[0],
                tag = args.otag,
                nch = "n" + str(args.nchunk),
                idx = "i" + str(args.ichunk)), False, True)
            syscall("rm higgsCombine_limit_g-scan_*POINT.1.*AsymptoticLimits*.root", False, True)
            limits = OrderedDict()

            gvals = chunks(list(np.linspace(min_g, max_g, num = 193)), args.nchunk)[args.ichunk]
            lll = dotty_scan((gvals, dcdir, mstr, accuracies, poi_range, fit_strategy("1"), args.asimov, masks))

            print "\nsingle_point_ahtt :: collecting limit"
            print "\nthe following points have been processed:"
            print gvals

            for ll in lll:
                if ll is not None:
                    limits[ll[0]] = ll[1]
            limits = OrderedDict(sorted(limits.items()))

            syscall("hadd {dcd}{pnt}{tag}_limits_g-scan_{nch}_{idx}.root higgsCombine_limit_g-scan_*POINT.1.AsymptoticLimits*.root && "
                    "rm higgsCombine_limit_g-scan_*POINT.1.*AsymptoticLimits*.root".format(
                        dcd = dcdir,
                        pnt = args.point[0],
                        tag = args.otag,
                        nch = "n" + str(args.nchunk),
                        idx = "i" + str(args.ichunk)
                    ))
            with open("{dcd}{pnt}{tag}_limits_g-scan_{nch}_{idx}.json".format(dcd = dcdir, pnt = args.point[0], tag = args.otag, nch = "n" + str(args.nchunk), idx = "i" + str(args.ichunk)), "w") as jj:
                json.dump(limits, jj, indent = 1)

    if runpull:
        group = ""
        nuisances = ""
        if args.impactnui is not None:
            group = "_" + args.impactnui[0]
            nuisances = args.impactnui[1]

        syscall("rm {dcd}{pnt}{tag}_impacts_{mod}{gvl}{rvl}{fix}{grp}.json".format(
            dcd = dcdir,
            pnt = args.point[0],
            tag = args.otag,
            mod = "one-poi" if args.onepoi else "g-scan",
            gvl = "_g_" + str(args.setg).replace(".", "p") if args.setg >= 0. else "",
            rvl = "_r_" + str(args.setr).replace(".", "p") if args.setr >= 0. and not args.onepoi else "",
            fix = "_fixed" if args.fixpoi and (args.setg >= 0. or args.setr >= 0.) else "",
            grp = group
        ), False, True)

        syscall("rm higgsCombine*Fit__pull*.root", False, True)
        syscall("rm combine_logger.out", False, True)

        if not args.onepoi and not (args.setg >= 0. and args.fixpoi):
            raise RuntimeError("it is unknown if impact works correctly with the g-scan model when g is left floating. please freeze it.")

        if args.frzbbp:
            best_fit_file = make_best_fit(dcdir, "workspace_{mod}.root".format(mod = "one-poi" if args.onepoi else "g-scan"), args.point[0],
                                          args.asimov, fit_strategy("1", True), poi_range,
                                          elementwise_add([starting_poi(args.onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.point[0], args.frzbb0)]),
                                          args.extopt, masks)
        set_freeze = elementwise_add([starting_poi(args.onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.point[0], args.frzbb0, args.frzbbp, False, best_fit_file)])

        print "\nsingle_point_ahtt :: impact initial fit"
        syscall("combineTool.py -M Impacts -d {dcd}workspace_{mod}.root -m {mmm} --doInitialFit -n _pull {stg} {prg} {asm} {prm}".format(
            dcd = dcdir,
            mod = "one-poi" if args.onepoi else "g-scan",
            mmm = mstr,
            prg = poi_range,
            stg = fit_strategy("1", True),
            asm = "-t -1" if args.asimov else "",
            prm = set_parameter(set_freeze, args.extopt, masks)
        ))

        print "\nsingle_point_ahtt :: impact remaining fits"
        syscall("combineTool.py -M Impacts -d {dcd}workspace_{mod}.root -m {mmm} --doFits -n _pull {stg} {prg} {asm} {nui} {prm}".format(
            dcd = dcdir,
            mod = "one-poi" if args.onepoi else "g-scan",
            mmm = mstr,
            prg = poi_range,
            stg = fit_strategy("1", True),
            asm = "-t -1" if args.asimov else "",
            nui = "--named '" + nuisances + "'" if args.impactnui is not None else "",
            prm = set_parameter(set_freeze, args.extopt, masks)
        ))

        print "\nsingle_point_ahtt :: collecting impact results"
        syscall("combineTool.py -M Impacts -d {dcd}workspace_{mod}.root -m {mmm} -n _pull -o {dcd}{pnt}{tag}_impacts_{mod}{gvl}{rvl}{fix}{grp}.json".format(
            dcd = dcdir,
            mod = "one-poi" if args.onepoi else "g-scan",
            mmm = mstr,
            pnt = args.point[0],
            tag = args.otag,
            gvl = "_g_" + str(args.setg).replace(".", "p") if args.setg >= 0. else "",
            rvl = "_r_" + str(args.setr).replace(".", "p") if args.setr >= 0. and not args.onepoi else "",
            fix = "_fixed" if args.fixpoi and (args.setg >= 0. or args.setr >= 0.) else "",
            grp = group
        ))

        syscall("rm higgsCombine*Fit__pull*.root", False, True)
        syscall("rm combine_logger.out", False, True)

    if runprepost:
        if args.frzbbp or args.frznui:
            best_fit_file = make_best_fit(dcdir, "workspace_{mod}.root".format(mod = "one-poi" if args.onepoi else "g-scan"), args.point[0],
                                          args.asimov, fit_strategy("2", True) + " --robustHesse 1", poi_range,
                                          elementwise_add([starting_poi(args.onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.point[0], args.frzbb0)]),
                                          args.extopt, masks)
        set_freeze = elementwise_add([starting_poi(args.onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.point[0], args.frzbb0, args.frzbbp, args.frznui, best_fit_file)])

        print "\nsingle_point_ahtt :: making pre- and postfit plots and covariance matrices"
        syscall("combine -v -1 -M FitDiagnostics {dcd}workspace_{mod}.root --saveWithUncertainties --saveNormalizations --saveShapes --saveOverallShapes "
                "--plots -m {mmm} -n _prepost {stg} {prg} {asm} {prm}".format(
                    dcd = dcdir,
                    mod = "one-poi" if args.onepoi else "g-scan",
                    mmm = mstr,
                    stg = fit_strategy("2", True) + " --robustHesse 1",
                    prg = poi_range,
                    asm = "-t -1" if args.asimov else "",
                    prm = set_parameter(set_freeze, args.extopt, masks)
        ))

        syscall("rm *_th1x_*.png", False, True)
        syscall("rm covariance_fit_?.png", False, True)
        syscall("rm higgsCombine_prepost*.root", False, True)
        syscall("rm combine_logger.out", False, True)
        syscall("rm robustHesse_*.root", False, True)

        syscall("mv fitDiagnostics_prepost.root {dcd}{pnt}{tag}_fitdiagnostics_{mod}{gvl}{rvl}{fix}.root".format(
            dcd = dcdir,
            pnt = args.point[0],
            tag = args.otag,
            mod = "one-poi" if args.onepoi else "g-scan",
            gvl = "_g_" + str(args.setg).replace(".", "p") if args.setg >= 0. else "",
            rvl = "_r_" + str(args.setr).replace(".", "p") if args.setr >= 0. and not args.onepoi else "",
            fix = "_fixed" if args.fixpoi and (args.setg >= 0. or args.setr >= 0.) else "",
        ), False)

    if runnll:
        print "\nsingle_point_ahtt :: calculating nll as a function of gA/H"
        if args.frzbbp or args.frznui:
            best_fit_file = make_best_fit(dcdir, "workspace_{mod}.root".format(mod = "one-poi" if args.onepoi else "g-scan"), args.point[0],
                                          args.asimov, fit_strategy("1", True), poi_range,
                                          elementwise_add([starting_poi(args.onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.point[0], args.frzbb0)]),
                                          "", masks)
        set_freeze = starting_nuisance(args.point[0], args.frzbb0, args.frzbbp, args.frznui, best_fit_file)
        setpar = set_freeze[0]
        frzpar = set_freeze[1]

        gvalues = [2.**-17, 2.**-15, 2.**-13, 2.**-11, 2.**-9] + list(np.linspace(min_g, max_g, num = 193))
        gvalues.sort()
        scenarii = ['exp-b', 'exp-s', 'obs']
        nlls = OrderedDict()

        syscall("rm {dcd}{pnt}{tag}_nll_one-poi.root {dcd}{pnt}{tag}_nll_{mod}.json".format(dcd = dcdir, pnt = args.point[0], tag = args.otag, mod = "one-poi" if args.onepoi else "g-scan"), False, True)

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
                        "--saveNLL --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 {stg} {asm} {stp} {frz}".format(
                            dcd = dcdir,
                            mmm = mstr,
                            prg = poi_range,
                            gvl = "--points 193 --alignEdges 1",
                            stg = fit_strategy("1", True),
                            asm = asimov,
                            stp = "--setParameters '" + ",".join(setpar + pois + masks) + "'" if len(setpar + pois + masks) > 0 else "",
                            frz = "--freezeParameters '" + ",".join(frzpar) + "'" if len(frzpar) > 0 else ""
                        ))

                syscall("mv higgsCombine_nll.MultiDimFit.mH*.root {dcd}{pnt}{tag}_nll_{sce}_one-poi.root".format(
                    dcd = dcdir,
                    sce = sce,
                    pnt = args.point[0],
                    tag = args.otag
                ), False)

                nlls[sce] = get_nll_one_poi("{dcd}{pnt}{pnt}_nll_{sce}_one-poi.root".format(
                    dcd = dcdir,
                    sce = sce,
                    pnt = args.point[0],
                    tag = args.otag,
                ))
            else:
                for gval in gvalues:
                    gstr = str(round(gval, 3)).replace('.', 'p')
                    syscall("combineTool.py -v -1 -M MultiDimFit --algo fixed -d {dcd}workspace_g-scan.root -m {mmm} -n _nll_{gst} {prg} "
                            "--saveNLL --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 --fixedPointPOIs r=1,g={gvl} {stg} {asm} {stp} {frz}".format(
                                dcd = dcdir,
                                mmm = mstr,
                                gst = "fix-g_" + str(gstr).replace(".", "p"),
                                prg = poi_range,
                                gvl = str(gval),
                                stg = fit_strategy("1", True),
                                asm = asimov,
                                stp = "--setParameters '" + ",".join(setpar + pois + masks) + "'" if len(setpar + pois + masks) > 0 else "",
                                frz = "--freezeParameters '" + ",".join(frzpar) + "'" if len(frzpar) > 0 else ""
                            ))

                syscall("hadd {dcd}{pnt}{tag}_nll_{sce}_g-scan.root higgsCombine_nll_fix-g*.root && "
                        "rm higgsCombine_nll_fix-g*.root".format(
                            dcd = dcdir,
                            sce = sce,
                            pnt = args.point[0],
                            tag = args.otag
                        ))

                nlls[sce] = get_nll_g_scan("{dcd}{pnt}{tag}_nll_{sce}_g-scan.root".format(
                    dcd = dcdir,
                    sce = sce,
                    pnt = args.point[0],
                    tag = args.otag
                ))

        syscall("rm combine_logger.out", False, True)

        with open("{dcd}{pnt}{tag}_nll_{mod}.json".format(dcd = dcdir, pnt = args.point[0], tag = args.otag, mod = "one-poi" if args.onepoi else "g-scan"), "w") as jj:
            json.dump(nlls, jj, indent = 1)

    if not os.path.isfile(best_fit_file):
        syscall("rm {bff}".format(bff = best_fit_file), False, True)

    if args.compress:
        syscall(("tar -czf {dcd}.tar.gz {dcd} && rm -r {dcd}").format(
            dcd = dcdir[:-1]
        ))
