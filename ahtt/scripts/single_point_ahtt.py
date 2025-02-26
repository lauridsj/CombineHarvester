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

from utilspy import syscall, get_point, chunks, elementwise_add, coinflip
from utilscombine import min_g, max_g, get_best_fit, starting_nuisance, fit_strategy, make_datacard_with_args, set_range, set_parameter, nonparametric_option
from utilscombine import is_good_fit, never_gonna_give_you_up

from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes
from argumentative import common_point, common_common, common_fit_pure, common_fit, make_datacard_pure, make_datacard_forwarded, common_1D, parse_args
from hilfemir import combine_help_messages

def get_limit(lfile):
    if not os.path.isfile(lfile):
        return None

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
    lfile = "higgsCombine_paramFit__pull_{nui}.MultiDimFit.mH{mmm}.root".format(nui = nuisance, mmm = mass)
    if not os.path.isfile(lfile):
        return None

    # really just the existence of up/down - two-sidedness seems too strong a constraint
    lfile = TFile.Open(lfile)
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

def is_acceptable(limit):
    if limit is None:
        return False
    epsilon = 2.**-17
    cls = [ll for qq, ll in limit.items()]
    gte0 = all([cc >= 0. for cc in cls])
    obs0exp1 = limit["obs"] < epsilon and all([abs(ll - 1.) < epsilon for qq, ll in limit.items() if "exp" in qq])
    return gte0 and not obs0exp1

def single_point_scan(args):
    gval, workspace, mstr, first_strategy, accuracies, asimov, masks = args
    gstr = str(round(gval, 3)).replace('.', 'p')
    fname = "higgsCombine_limit_g-scan_{gst}.POINT.1.AsymptoticLimits.mH{mmm}.root".format(mmm = mstr, gst = gstr)
    epsilon = 2.**-13
    nstep = 1
    geps = 0.
    limit = None
    syscall("rm {fname}".format(fname = fname), False, True)

    for factor in [0., 1.]:
        for ii in range(1, nstep + 1 if factor != 0. else 2):
            limit = None
            sign = -1. if gval != 0. and coinflip() else 1.
            never_gonna_give_you_up(
                command = "combineTool.py -v 0 -M AsymptoticLimits -d {wsp} -m {mmm} -n _limit_g-scan_{gst} --strictBounds "
                "--setParameters g={gvl} --freezeParameters g {acc} --picky --redefineSignalPOIs r --singlePoint 1 {stg} {asm} {msk}".format(
                    wsp = workspace,
                    mmm = mstr,
                    gst = gstr,
                    gvl = gval + (ii * sign * factor * epsilon),
                    acc = accuracies,
                    stg = "{fit_strategy}",
                    asm = "-t -1" if asimov else "",
                    msk = "--setParameters '" + ",".join(masks) + "'" if len(masks) > 0 else ""
                ),

                post_conditions = [
                    [lambda fn: is_acceptable(get_limit(fn)), fname]
                ],
                failure_cleanups = [
                    [syscall, "rm {fn}".format(fn = fname), False]
                ],

                all_strategies = [(False, 0, 0), (False, 0, 1), (False, 0, 2), (False, 0, 3), (False, 1, 3), (False, 2, 3)],
                throw_upon_failure = False,
                first_fit_strategy = first_strategy if first_strategy > -1 else 0
            )

            limit = get_limit(fname)
            if is_acceptable(limit):
                geps = (ii * sign * factor * epsilon)
                return [gval + geps, limit, min([(abs(ll - 0.05), ll) for qq, ll in limit.items()])[1]] # third being the closest cls to 0.05 among the quantiles
    return None

def dotty_scan(args):
    gvals, workspace, mstr, first_strategy, accuracies, asimov, masks = args
    if len(gvals) < 2:
        return None
    gvals = sorted(gvals)

    results = []
    ii = 0
    while ii < len(gvals):
        result = single_point_scan((gvals[ii], workspace, mstr, first_strategy, accuracies, asimov, masks))

        if result is not None:
            if (0.0125 < result[2] < 0.2):
                ii += 1
            else:
                ii += 2
        else:
            ii += 3
            continue
        results.append(result)
    return results

def starting_poi(onepoi, gvalue, rvalue, fixpoi):
    setfrzpar = [[], []]
    setpar, frzpar = setfrzpar

    if gvalue >= 0.:
        setpar.append('g=' + str(gvalue))
        if fixpoi:
            frzpar.append('g')

    if not onepoi:
        if rvalue >= 0.:
            setpar.append('r=' + str(rvalue))
            if fixpoi:
                frzpar.append('r')

    return setfrzpar

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

    allmodes = ["datacard", "workspace", "validate", "best", "best-fit", "single", "limit", "pull", "impact"]
    if (not all([mm in allmodes for mm in modes])):
        print "supported modes:", allmodes
        raise RuntimeError("unxpected mode is given. aborting.")

    # determine what to do with workspace, and do it
    rundc = "datacard" in modes or "workspace" in modes
    runvalid = "validate" in modes
    runbest = "best" in modes or "best-fit" in modes
    runsingle = "single" in modes
    runlimit = "limit" in modes
    runpull = "pull" in modes or "impact" in modes

    runbest = runsingle or runbest or rundc
    args.keepbest = False if runbest else args.keepbest

    if runsingle:
        args.extopt += " --algo singles --cl=0.68"

    # pois to use in the fit
    poiset = args.poiset if len(args.poiset) else ["g"] if args.onepoi else ["r", "g"]
    poiset = sorted(list(set(poiset)))
    onepoinotg = len(poiset) == 1 and poiset[0] != "g"
    args.onepoi = args.onepoi or onepoinotg

    # parameter ranges for best fit file
    ranges = ["g: 0, 5"] if args.onepoi else ["r: 0, 2", "g: 0, 5"]

    if args.experimental:
        ranges += ["rgx{EWK_.*}", "rgx{QCDscale_ME.*}", "tmass"] # veeeery wide hedging for theory ME NPs

    if rundc:
        print "\nsingle_point_ahtt :: making datacard"
        make_datacard_with_args(scriptdir, args)

        print "\nsingle_point_ahtt :: making workspaces"
        for onepoi in [True, False]:
            syscall("combineTool.py -v 0 -M T2W -i {dcd} -o workspace_{mod}.root -m {mmm} -P CombineHarvester.CombineTools.MultiInterferencePlusFixed:multiInterferencePlusFixed "
                    "--PO 'signal={pnt}' {one} {vbs} {opt} {dyt} {ext}".format(
                        dcd = dcdir + "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else dcdir + "ahtt_" + args.channel + '_' + args.year + ".txt",
                        mod = "one-poi" if onepoi else "g-scan",
                        mmm = mstr,
                        pnt = args.point[0],
                        one = "--PO no-r" if onepoi else "",
                        vbs = "--PO verbose",
                        dyt = "--PO yukawa" if "EWK_TT" in args.assignal else "",
                        opt = "--channel-masks --no-wrappers --X-pack-asympows --optimize-simpdf-constraints=cms --use-histsum",
                        ext = args.extopt
                    ))

    if runvalid:
        print "\nsingle_point_ahtt :: validating datacard"
        syscall("ValidateDatacards.py --jsonFile {dcd}{ptg}_validate.json --printLevel 3 {dcd}{crd}".format(
            dcd = dcdir,
            ptg = ptag,
            crd = "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else "ahtt_" + args.channel + '_' + args.year + ".txt"
        ))

    for onepoi in [not args.onepoi, args.onepoi]:
        if runsingle and onepoinotg and not onepoi:
            continue

        default_workspace = dcdir + "workspace_{mod}.root".format(mod = "one-poi" if onepoi else "g-scan")
        workspace = get_best_fit(
            dcdir, args.point[0], [args.otag, args.tag],
            args.defaultwsp, args.keepbest, default_workspace, args.asimov,
            "single" if runsingle else "", poiset[0] if onepoinotg else "one-poi" if onepoi else "g-scan",
            "{gvl}{rvl}{fix}".format(
                gvl = "g_" + str(args.setg).replace(".", "p") if args.setg >= 0. else "",
                rvl = "_r_" + str(args.setr).replace(".", "p") if args.setr >= 0. and not onepoi else "",
                fix = "_fixed" if args.fixpoi and (args.setg >= 0. or args.setr >= 0.) else ""
            ),
            ["g"] if onepoi and not onepoinotg else poiset,
            set_range(ranges),
            elementwise_add([starting_poi(onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.frzzero, args.frznzro, set())]), args.extopt, masks
        )

    if runlimit:
        print "\nsingle_point_ahtt :: computing limit"
        accuracies = '--rRelAcc 0.01 --rAbsAcc 0.001'

        if args.onepoi:
            syscall("rm {dcd}{ptg}_limits_one-poi.root {dcd}{ptg}_limits_one-poi.json".format(dcd = dcdir, ptg = ptag), False, True)
            syscall("combineTool.py -v 0 -M AsymptoticLimits -d {dcd} -m {mmm} -n _limit {acc} {stg} {asm} {msk} {poi}".format(
                dcd = workspace,
                mmm = mstr,
                acc = accuracies,
                stg = fit_strategy(strategy = args.fitstrat if args.fitstrat > -1 else 0),
                asm = "--run blind -t -1" if args.asimov else "",
                msk = "--setParameters '" + ",".join(masks) + "'" if len(masks) > 0 else "",
                poi = "--redefineSignalPOIs '{poi}'".format(poi = poiset[0]) if len(poiset) == 1 else ""
            ))

            print "\nsingle_point_ahtt :: collecting limit"
            syscall("combineTool.py -v 0 -M CollectLimits higgsCombine_limit.AsymptoticLimits.mH{mmm}.root -m {mmm} -o {dcd}{ptg}_limits_one-poi.json && "
                    "rm higgsCombine_limit.AsymptoticLimits.mH{mmm}.root".format(
                        dcd = dcdir,
                        mmm = mstr,
                        ptg = ptag
                    ))
        else:
            if args.nchunk < 0:
                args.nchunk = 6
            if args.ichunk < 0 or args.ichunk >= args.nchunk:
                args.ichunk = 0

            fexp = "higgsCombine_limit_g-scan_*POINT.1.AsymptoticLimits.mH{mmm}.root".format(mmm = mstr)
            syscall("rm {dcd}{ptg}_limits_g-scan_{nch}_{idx}.root {dcd}{ptg}_limits_g-scan_{nch}_{idx}.json ".format(
                dcd = dcdir,
                ptg = ptag,
                nch = "n" + str(args.nchunk),
                idx = "i" + str(args.ichunk)), False, True)
            syscall("rm {fexp}".format(fexp = fexp), False, True)
            limits = OrderedDict()

            gvals = chunks(list(np.linspace(min_g, max_g, num = 193)), args.nchunk)[args.ichunk]
            lll = dotty_scan(
                (gvals, workspace, mstr, args.fitstrat, accuracies, args.asimov, masks)
            )
            print "\nsingle_point_ahtt :: collecting limit"
            print "\nthe following points have been processed:"
            print gvals

            for ll in lll:
                if ll is not None:
                    limits[ll[0]] = ll[1]
            limits = OrderedDict(sorted(limits.items()))

            ofiles = glob.glob(fexp)
            if len(ofiles) > 0 and len(limits):
                ofile = "{dcd}{ptg}_limits_g-scan_{nch}_{idx}.root".format(
                    dcd = dcdir,
                    ptg = ptag,
                    nch = "n" + str(args.nchunk),
                    idx = "i" + str(args.ichunk)
                )
                cmd = "hadd {ofile} {fexp} && rm {fexp}" if len(ofiles) > 1 else "mv {fexp} {ofile}"
                syscall(cmd.format(fexp = fexp, ofile = ofile))
                with open(ofile.replace(".root", ".json"), "w") as jj:
                    json.dump(limits, jj, indent = 1)
            else:
                syscall("rm {fexp}".format(fexp = fexp), False, True)

    if runpull:
        group = ""
        nuisances = [""]
        if args.impactnui is not None:
            group = "_" + args.impactnui[0]
            nuisances = tokenize_to_list(args.impactnui[1])

        if onepoinotg:
            nuisances = [nn for nn in nuisances if nn != poiset[0]]

        oname = "{dcd}{ptg}_impacts_{poi}{gvl}{rvl}{fix}{grp}".format(
            dcd = dcdir,
            ptg = ptag,
            poi = poiset[0] if onepoinotg else "one-poi" if args.onepoi else "g-scan",
            gvl = "_g_" + str(args.setg).replace(".", "p") if args.setg >= 0. else "",
            rvl = "_r_" + str(args.setr).replace(".", "p") if args.setr >= 0. and not args.onepoi else "",
            fix = "_fixed" if args.fixpoi and (args.setg >= 0. or args.setr >= 0.) else "",
            grp = group
        )

        syscall("rm {onm}.json".format(onm = oname), False, True)
        syscall("rm higgsCombine*Fit__pull*.root", False, True)
        syscall("rm combine_logger.out", False, True)
        syscall("rm robustHesse_*.root", False, True)

        if not args.onepoi and not (args.setg >= 0. and args.fixpoi):
            raise RuntimeError("impact doesn't work correctly with the g-scan model when g is left floating. please freeze it.")
        set_freeze = elementwise_add([starting_poi(args.onepoi, args.setg, args.setr, args.fixpoi), starting_nuisance(args.frzzero, args.frznzro, args.frzpost)])

        print "\nsingle_point_ahtt :: impact initial fit"
        syscall("combineTool.py -v 0 -M Impacts -d {dcd} -m {mmm} --doInitialFit -n _pull {stg} {asm} {poi} {prm} {ext}".format(
            dcd = workspace,
            mmm = mstr,
            stg = fit_strategy(strategy = args.fitstrat if args.fitstrat > -1 else 0, robust = True, use_hesse = args.usehesse),
            asm = "-t -1" if args.asimov else "",
            poi = "--redefineSignalPOIs '{poi}'".format(poi = poiset[0]),
            prm = set_parameter(set_freeze, args.extopt, masks),
            ext = nonparametric_option(args.extopt)
        ))
        syscall("rm robustHesse_*.root", False, True)

        print "\nsingle_point_ahtt :: impact remaining fits"
        for nuisance in nuisances:
            if nuisance in args.frzzero or nuisance in args.frzpost:
                continue

            never_gonna_give_you_up(
                command = "combineTool.py -v 0 -M Impacts -d {dcd} -m {mmm} --doFits -n _pull {stg} {asm} "
                "{poi} {nui} {prm} {ext}".format(
                    dcd = workspace,
                    mmm = mstr,
                    stg = "{fit_strategy}",
                    asm = "-t -1" if args.asimov else "",
                    poi = "--redefineSignalPOIs '{poi}'".format(poi = poiset[0]),
                    nui = "--named '" + nuisance + "'" if nuisance != "" else "",
                    prm = set_parameter(set_freeze, args.extopt, masks),
                    ext = nonparametric_option(args.extopt)
                ),

                post_conditions = [
                    [lambda nn, mm: nn == "" or sensible_enough_pull(nn, mm), nuisance, mstr]
                ],
                #failure_cleanups = [
                #    [syscall, "rm {fn}".format(fn = fname), False]
                #],

                usehesse = args.usehesse,
                first_fit_strategy = args.fitstrat if args.fitstrat > -1 else 0
            )

        print "\nsingle_point_ahtt :: collecting impact results"
        syscall("combineTool.py -v 0 -M Impacts -d {wsp} -m {mmm} -n _pull -o {onm}.json {nui} {ext}".format(
            wsp = workspace,
            mmm = mstr,
            onm = oname,
            nui = "--named '" + ','.join(nuisances) + "'" if args.impactnui is not None else "",
            ext = nonparametric_option(args.extopt)
        ))

        syscall("rm higgsCombine*Fit__pull*.root", False, True)
        syscall("rm combine_logger.out", False, True)
        syscall("rm robustHesse_*.root", False, True)

    if args.compress:
        syscall(("tar -czf {dcd}.tar.gz {dcd} && rm -r {dcd}").format(
            dcd = dcdir[:-1]
        ))
