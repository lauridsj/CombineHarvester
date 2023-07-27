#!/usr/bin/env python
# sets the model-independent limit on a single A/H signal point

from argparse import ArgumentParser
import os
import sys
import numpy as np
import glob

from collections import OrderedDict
import json

from ROOT import TFile, TTree

from utilspy import syscall, get_point, chunks, elementwise_add
from utilscombine import min_g, max_g, get_best_fit, starting_nuisance, fit_strategy, make_datacard_with_args, set_range, set_parameter, nonparametric_option

from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes
from argumentative import common_point, common_common, common_fit_pure, common_fit, make_datacard_pure, make_datacard_forwarded, common_1D, parse_args
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

def sensible_enough_pull(nuisance, mass):
    # really just the existence of up/down - two-sidedness seems too strong a constraint
    lfile = TFile.Open("higgsCombine_paramFit__pull_{nui}.MultiDimFit.mH{mmm}.root".format(nui = nuisance, mmm = mass))
    ltree = lfile.Get("limit")

    central, band, isgood = False, 0, False
    for i in ltree:
        value = getattr(ltree, nuisance, None)

        if value is not None:
            if ltree.quantileExpected == -1.:
                central = True
            elif ltree.quantileExpected >= 0.:
                band += 1

        if central and band > 1:
            isgood = True
            break

    lfile.Close()
    return isgood

def single_point_scan(args):
    gval, dcdir, workspace, mstr, accuracies, strategy, asimov, masks = args
    gstr = str(round(gval, 3)).replace('.', 'p')

    epsilon = 2.**-17
    nstep = 1

    syscall("combineTool.py -M AsymptoticLimits -d {dcd}workspace_g-scan.root -m {mmm} -n _limit_g-scan_{gst} "
            "--setParameters g={gvl} --freezeParameters g {acc} --picky --singlePoint 1 {stg} {asm} {msk}".format(
                dcd = dcdir,
                mmm = mstr,
                gst = gstr,
                gvl = gval,
                acc = accuracies,
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
                    "--setParameters g={gvl} --freezeParameters g {acc} --picky --singlePoint 1 {stg} {asm} {msk}".format(
                        dcd = dcdir,
                        mmm = mstr,
                        gst = gstr + "eps",
                        gvl = gval + (ii * factor * epsilon),
                        acc = accuracies,
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
    gvals, dcdir, workspace, mstr, accuracies, strategy, asimov, masks = args
    if len(gvals) < 2:
        return None
    gvals = sorted(gvals)

    results = []
    ii = 0
    while ii < len(gvals):
        result = single_point_scan((gvals[ii], dcdir, workspace, mstr, accuracies, strategy, asimov, masks))

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
    print "single_point_ahtt :: called with the following arguments"
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
    common_1D(parser)

    parser.add_argument("--raster-i", help = combine_help_messages["--raster-i"], dest = "ichunk", default = 0, required = False, type = lambda s: int(remove_spaces_quotes(s)))
    parser.add_argument("--impact-nuisances", help = combine_help_messages["--impact-nuisances"], dest = "impactnui", default = "", required = False,
                        type = lambda s: None if s == "" else tokenize_to_list( remove_spaces_quotes(s), ';' ))
    args = parse_args(parser)

    if len(args.point) != 1:
        raise RuntimeError("this script is to be used with exactly one A/H point!")

    modes = args.mode
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    dcdir = args.basedir + args.point[0] + args.tag + "/"

    point = get_point(args.point[0])
    mstr = str(point[1]).replace(".0", "")
    ptag = "{pnt}{tag}".format(pnt = args.point[0], tag = args.otag)

    masks = ["mask_" + mm + "=1" for mm in args.mask]
    print "the following channel x year combinations will be masked:", args.mask

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

    ranges = "--setParameterRanges 'r=0.,5.'" if args.onepoi else "--setParameterRanges 'r=0.,2.:g=0.,5.'"
    # parameter ranges for best fit file
    ranges = ["r: 0, 5"] if args.onepoi else ["r: 0, 2", "g: 0, 5"]
    if args.experimental:
        ranges += ["rgx{EWK_.*}", "rgx{QCDscale_ME.*}", "tmass"] # veeeery wide hedging for theory ME NPs

    if rundc:
        print "\nsingle_point_ahtt :: making datacard"
        make_datacard_with_args(scriptdir, args)

        print "\nsingle_point_ahtt :: making workspaces"
        for onepoi in [True, False]:
            syscall("combineTool.py -M T2W -i {dcd} -o workspace_{mod}.root -m {mmm} -P CombineHarvester.CombineTools.MultiInterferencePlusFixed:multiInterferencePlusFixed "
                    "--PO 'signal={pnt}' {one} {pos} {opt} {ext}".format(
                        dcd = dcdir + "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else dcdir + "ahtt_" + args.channel + '_' + args.year + ".txt",
                        mod = "one-poi" if onepoi else "g-scan",
                        mmm = mstr,
                        pnt = args.point[0],
                        one = "--PO no-r" if onepoi else "",
                        pos = " ".join(["--PO " + stuff for stuff in ["verbose", "yukawa"]]),
                        opt = "--channel-masks --no-wrappers --X-pack-asympows --optimize-simpdf-constraints=cms --use-histsum",
                        ext = args.extopt
                    ))

            #if args.oldmodelnoyukawa:
            #    syscall("combineTool.py -M T2W -i {dcd} -o workspace_{mod}.root -m {mmm} -P CombineHarvester.CombineTools.{phy} {opt} {ext}".format(
            #        dcd = dcdir + "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else dcdir + "ahtt_" + args.channel + '_' + args.year + ".txt",
            #        mod = "one-poi" if onepoi else "g-scan",
            #        mmm = mstr,
            #        phy = "InterferenceModel:interferenceModel" if onepoi else "InterferencePlusFixed:interferencePlusFixed",
            #        opt = "--channel-masks --no-wrappers --X-pack-asympows --optimize-simpdf-constraints=cms --use-histsum",
            #        ext = args.extopt
            #    ))

    if runvalid:
        print "\nsingle_point_ahtt :: validating datacard"
        syscall("ValidateDatacards.py --jsonFile {dcd}{ptg}_validate.json --printLevel 3 {dcd}{crd}".format(
            dcd = dcdir,
            ptg = ptag,
            crd = "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else "ahtt_" + args.channel + '_' + args.year + ".txt"
        ))

    for onepoi in [not args.onepoi, args.onepoi]:
        default_workspace = dcdir + "workspace_{mod}.root".format(mod = "one-poi" if onepoi else "g-scan")
        workspace = get_best_fit(
            dcdir, args.point[0], [args.otag, args.tag],
            args.defaultwsp, args.keepbest, default_workspace, args.asimov, "one-poi" if onepoi else "g-scan",
            "{gvl}{rvl}{fix}".format(
                gvl = "g_" + str(args.setg).replace(".", "p") if args.setg >= 0. else "",
                rvl = "_r_" + str(args.setr).replace(".", "p") if args.setr >= 0. and not onepoi else "",
                fix = "_fixed" if args.fixpoi and (args.setg >= 0. or args.setr >= 0.) else ""
            ),
            fit_strategy(args.fitstrat if args.fitstrat > -1 else 2, True, args.usehesse), set_range(ranges),
            elementwise_add([starting_poi(onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.frzzero, set())]), args.extopt, masks
        )

    if runlimit:
        print "\nsingle_point_ahtt :: computing limit"
        accuracies = '--rRelAcc 0.005 --rAbsAcc 0'

        if args.onepoi:
            syscall("rm {dcd}{ptg}_limits_one-poi.root {dcd}{ptg}_limits_one-poi.json".format(dcd = dcdir, ptg = ptag), False, True)
            syscall("combineTool.py -M AsymptoticLimits -d {dcd} -m {mmm} -n _limit {acc} {stg} {asm} {msk}".format(
                dcd = workspace,
                mmm = mstr,
                maxg = max_g,
                acc = accuracies,
                stg = fit_strategy(args.fitstrat if args.fitstrat > -1 else 1),
                asm = "--run blind -t -1" if args.asimov else "",
                msk = "--setParameters '" + ",".join(masks) + "'" if len(masks) > 0 else ""
            ))

            print "\nsingle_point_ahtt :: collecting limit"
            syscall("combineTool.py -M CollectLimits higgsCombine_limit.AsymptoticLimits.mH*.root -m {mmm} -o {dcd}{ptg}_limits_one-poi.json && "
                    "rm higgsCombine_limit.AsymptoticLimits.mH*.root".format(
                        dcd = dcdir,
                        mmm = mstr,
                        ptg = ptag
                    ))
        else:
            if args.nchunk < 0:
                args.nchunk = 6
            if args.ichunk < 0 or args.ichunk >= args.nchunk:
                args.ichunk = 0

            syscall("rm {dcd}{ptg}_limits_g-scan_{nch}_{idx}.root {dcd}{ptg}_limits_g-scan_{nch}_{idx}.json ".format(
                dcd = dcdir,
                ptg = ptag,
                nch = "n" + str(args.nchunk),
                idx = "i" + str(args.ichunk)), False, True)
            syscall("rm higgsCombine_limit_g-scan_*POINT.1.*AsymptoticLimits*.root", False, True)
            limits = OrderedDict()

            gvals = chunks(list(np.linspace(min_g, max_g, num = 193)), args.nchunk)[args.ichunk]
            lll = dotty_scan((gvals, dcdir, workspace, mstr, accuracies, fit_strategy(args.fitstrat if args.fitstrat > -1 else 1), args.asimov, masks))

            print "\nsingle_point_ahtt :: collecting limit"
            print "\nthe following points have been processed:"
            print gvals

            for ll in lll:
                if ll is not None:
                    limits[ll[0]] = ll[1]
            limits = OrderedDict(sorted(limits.items()))

            syscall("hadd {dcd}{ptg}_limits_g-scan_{nch}_{idx}.root higgsCombine_limit_g-scan_*POINT.1.AsymptoticLimits*.root && "
                    "rm higgsCombine_limit_g-scan_*POINT.1.*AsymptoticLimits*.root".format(
                        dcd = dcdir,
                        ptg = ptag,
                        nch = "n" + str(args.nchunk),
                        idx = "i" + str(args.ichunk)
                    ))
            with open("{dcd}{ptg}_limits_g-scan_{nch}_{idx}.json".format(dcd = dcdir, ptg = ptag, nch = "n" + str(args.nchunk), idx = "i" + str(args.ichunk)), "w") as jj:
                json.dump(limits, jj, indent = 1)

    if runpull:
        group = ""
        nuisances = [""]
        if args.impactnui is not None:
            group = "_" + args.impactnui[0]
            nuisances = tokenize_to_list(args.impactnui[1])

        syscall("rm {dcd}{ptg}_impacts_{mod}{gvl}{rvl}{fix}{grp}.json".format(
            dcd = dcdir,
            ptg = ptag,
            mod = "one-poi" if args.onepoi else "g-scan",
            gvl = "_g_" + str(args.setg).replace(".", "p") if args.setg >= 0. else "",
            rvl = "_r_" + str(args.setr).replace(".", "p") if args.setr >= 0. and not args.onepoi else "",
            fix = "_fixed" if args.fixpoi and (args.setg >= 0. or args.setr >= 0.) else "",
            grp = group
        ), False, True)

        syscall("rm higgsCombine*Fit__pull*.root", False, True)
        syscall("rm combine_logger.out", False, True)
        syscall("rm robustHesse_*.root", False, True)

        if not args.onepoi and not (args.setg >= 0. and args.fixpoi):
            raise RuntimeError("it is unknown if impact works correctly with the g-scan model when g is left floating. please freeze it.")
        set_freeze = elementwise_add([starting_poi(args.onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.frzzero, args.frzpost)])

        print "\nsingle_point_ahtt :: impact initial fit"
        syscall("combineTool.py -M Impacts -d {dcd} -m {mmm} --doInitialFit -n _pull {stg} {asm} {prm} {ext}".format(
            dcd = workspace,
            mmm = mstr,
            stg = fit_strategy(args.fitstrat if args.fitstrat > -1 else 0, True, args.usehesse),
            asm = "-t -1" if args.asimov else "",
            prm = set_parameter(set_freeze, args.extopt, masks),
            ext = nonparametric_option(args.extopt)
        ))
        syscall("rm robustHesse_*.root", False, True)

        print "\nsingle_point_ahtt :: impact remaining fits"
        for nuisance in nuisances:
            for irobust, istrat, itol in [(irobust, istrat, itol) for irobust in [True, False] for istrat in [1, 2, 0] for itol in [0, -1, 1, -2, 2, -3, 3]]:
                syscall("combineTool.py -M Impacts -d {dcd} -m {mmm} --doFits -n _pull {stg} {asm} {nui} {prm} {ext}".format(
                    dcd = workspace,
                    mmm = mstr,
                    stg = fit_strategy(istrat, irobust, irobust and args.usehesse, itol),
                    asm = "-t -1" if args.asimov else "",
                    nui = "--named '" + nuisance + "'" if nuisance != "" else "",
                    prm = set_parameter(set_freeze, args.extopt, masks),
                    ext = nonparametric_option(args.extopt)
                ))
                syscall("rm robustHesse_*.root", False, True)

                if nuisance == "" or sensible_enough_pull(nuisance, mstr):
                    break

        print "\nsingle_point_ahtt :: collecting impact results"
        syscall("combineTool.py -M Impacts -d {wsp} -m {mmm} -n _pull -o {dcd}{ptg}_impacts_{mod}{gvl}{rvl}{fix}{grp}.json {nui}".format(
            dcd = dcdir,
            wsp = workspace,
            mod = "one-poi" if args.onepoi else "g-scan",
            mmm = mstr,
            ptg = ptag,
            gvl = "_g_" + str(args.setg).replace(".", "p") if args.setg >= 0. else "",
            rvl = "_r_" + str(args.setr).replace(".", "p") if args.setr >= 0. and not args.onepoi else "",
            fix = "_fixed" if args.fixpoi and (args.setg >= 0. or args.setr >= 0.) else "",
            grp = group,
            nui = "--named '" + ','.join(nuisances) + "'" if args.impactnui is not None else ""
        ))

        syscall("rm higgsCombine*Fit__pull*.root", False, True)
        syscall("rm combine_logger.out", False, True)
        syscall("rm robustHesse_*.root", False, True)

    if runprepost:
        set_freeze = elementwise_add([starting_poi(args.onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.frzzero, args.frzpost)])

        print "\nsingle_point_ahtt :: making pre- and postfit plots and covariance matrices"
        syscall("combine -v -1 -M FitDiagnostics {dcd} --saveWithUncertainties --saveNormalizations --saveShapes --saveOverallShapes "
                "--plots -m {mmm} -n _prepost {stg} {asm} {prm} {ext}".format(
                    dcd = workspace,
                    mmm = mstr,
                    stg = fit_strategy(args.fitstrat if args.fitstrat > -1 else 2, True, args.usehesse),
                    asm = "-t -1" if args.asimov else "",
                    prm = set_parameter(set_freeze, args.extopt, masks),
                    ext = nonparametric_option(args.extopt)
        ))

        syscall("rm *_th1x_*.png", False, True)
        syscall("rm covariance_fit_?.png", False, True)
        syscall("rm higgsCombine_prepost*.root", False, True)
        syscall("rm combine_logger.out", False, True)
        syscall("rm robustHesse_*.root", False, True)

        syscall("mv fitDiagnostics_prepost.root {dcd}{ptg}_fitdiagnostics_{mod}{gvl}{rvl}{fix}.root".format(
            dcd = dcdir,
            ptg = ptag,
            mod = "one-poi" if args.onepoi else "g-scan",
            gvl = "_g_" + str(args.setg).replace(".", "p") if args.setg >= 0. else "",
            rvl = "_r_" + str(args.setr).replace(".", "p") if args.setr >= 0. and not args.onepoi else "",
            fix = "_fixed" if args.fixpoi and (args.setg >= 0. or args.setr >= 0.) else "",
        ), False)

    if args.compress:
        syscall(("tar -czf {dcd}.tar.gz {dcd} && rm -r {dcd}").format(
            dcd = dcdir[:-1]
        ))
