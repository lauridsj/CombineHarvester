#!/usr/bin/env python
# sets the model-independent limit on two A/H signal points

from argparse import ArgumentParser
import os
import sys
import glob
import re
import numpy as np

from collections import OrderedDict
import json

from ROOT import TFile, TTree

from utilities import syscall, get_point, right_now
from utilities import read_nuisance, max_g, make_best_fit, starting_nuisance, elementwise_add, stringify, fit_strategy, make_datacard_with_args, set_parameter
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes
from argumentative import common_point, common_common, common_fit_pure, common_fit, make_datacard_pure, make_datacard_forwarded, common_2D
from hilfemir import combine_help_messages

def get_fit(dname, qexp_eq_m1 = True):
    dfile = TFile.Open(dname)
    dtree = dfile.Get("limit")

    bf = None
    for i in dtree:
        if (dtree.quantileExpected == -1. and qexp_eq_m1) or (dtree.quantileExpected != -1. and not qexp_eq_m1):
            bf = (getattr(dtree, "g1"), getattr(dtree, "g2"), dtree.deltaNLL)

        if bf is not None:
            break

    dfile.Close()
    return bf

def read_previous_grid(points, prev_best_fit, gname):
    with open(gname) as ff:
        result = json.load(ff, object_pairs_hook = OrderedDict)

    if result["points"] == points and result["best_fit_g1_g2_dnll"] == list(prev_best_fit):
        return result["g-grid"]
    else:
        print "\ninconsistent previous grid, ignoring the previous grid..."
        print "previous: ", points, prev_best_fit
        print "current: ", result["points"], result["best_fit_g1_g2_dnll"]

    return OrderedDict()

def get_toys(tname, best_fit, whatever_else = None):
    if not os.path.isfile(tname):
        return None

    pval = OrderedDict()

    tfile = TFile.Open(tname)
    ttree = tfile.Get("limit")

    isum = 0
    ipas = 0

    for i in ttree:
        if ttree.quantileExpected != -1. and not ttree.deltaNLL < 0.:
            isum += 1
            if ttree.deltaNLL > best_fit[2]:
                ipas += 1

    tfile.Close()

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
        print '\n WARNING :: incompatible expected/data dnll, when they should be!!'

    gs["total"] = g1["total"] + g2["total"]
    gs["pass"] = g1["pass"] + g2["pass"]

    return gs

def starting_poi(gvalues, fixpoi):
    if all(float(gg) < 0. for gg in gvalues):
        return [[], []]

    setpar = ['g' + str(ii + 1) + '=' + gg for ii, gg in enumerate(gvalues) if float(gg) >= 0.]
    frzpar = ['g' + str(ii + 1) for ii, gg in enumerate(gvalues) if float(gg) >= 0.] if fixpoi else []

    return [setpar, frzpar]

if __name__ == '__main__':
    parser = ArgumentParser()
    common_point(parser)
    common_common(parser)
    common_fit_pure(parser)
    common_fit(parser)
    make_datacard_pure(parser)
    make_datacard_forwarded(parser)
    common_2D(parser)

    parser.add_argument("--run-idx", help = combine_help_messages["--run-idx"], default = -1, dest = "runidx", required = False, type = lambda s: int(remove_spaces_quotes(s)))

    args = parser.parse_args()
    print "twin_point_ahtt :: called with the following arguments"
    print sys.argv[1:]
    print "\n"
    print " ".join(sys.argv)
    print "\n"
    sys.stdout.flush()

    points = args.point
    gvalues = args.gvalues
    if len(points) != 2 or len(gvalues) != 2:
        raise RuntimeError("this script is to be used with exactly two A/H points!")

    modes = args.mode
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    dcdir = args.basedir + "__".join(points) + args.tag + "/"

    mstr = str(get_point(points[0])[1]).replace(".0", "")
    poi_range = "--setParameterRanges '" + ":".join(["g" + str(ii + 1) + "=0.,5." for ii, pp in enumerate(points)]) + "'"
    best_fit_file = ""
    masks = ["mask_" + mm + "=1" for mm in args.mask]
    print "the following channel x year combinations will be masked:", args.mask

    gstr = ""
    for ii, gg in enumerate(gvalues):
        usc = "_" if ii > 0 else ""
        gstr += usc + "g" + str(ii + 1) + "_" + gvalues[ii] if float(gvalues[ii]) >= 0. else ""

    allmodes = ["datacard", "workspace", "validate",
                "generate", "gof", "fc-scan", "contour",
                "hadd", "merge", "compile",
                "prepost", "corrmat"]
    if (not all([mm in allmodes for mm in modes])):
        print "supported modes:", allmodes
        raise RuntimeError("unxpected mode is given. aborting.")

    # determine what to do with workspace, and do it
    rundc = "datacard" in modes or "workspace" in modes
    runvalid = "validate" in modes
    rungen = "generate" in modes
    rungof = "gof" in modes
    runfc = "fc-scan" in modes or "contour" in modes
    runhadd = "hadd" in modes or "merge" in modes
    runcompile = "compile" in args.mode
    runprepost = "prepost" in modes or "corrmat" in modes

    allexp = ["exp-b", "exp-s", "exp-01", "exp-10", "obs"]
    if len(args.fcexp) > 0 and not all([exp in allexp for exp in args.fcexp]):
        print "supported expected scenario:", allexp
        raise RuntimeError("unexpected expected scenario is given. aborting.")

    if args.otag == "":
        args.otag = args.tag

    if rundc:
        print "\ntwin_point_ahtt :: making datacard"
        make_datacard_with_args(scriptdir, args)

        print "\ntwin_point_ahtt :: making workspaces"
        syscall("combineTool.py -M T2W -i {dcd} -o workspace_twin-g.root -m {mmm} -P CombineHarvester.CombineTools.MultiInterferencePlusFixed:multiInterferencePlusFixed "
                "--PO verbose --PO 'signal={pnt}' --PO no-r --channel-masks".format(
                    dcd = dcdir + "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else dcdir + "ahtt_" + args.channel + '_' + args.year + ".txt",
                    mmm = mstr,
                    pnt = ",".join(points)
                ))

    if runvalid:
        print "\ntwin_point_ahtt :: validating datacard"
        syscall("ValidateDatacards.py --jsonFile {dcd}{pnt}_validate.json --printLevel 3 {dcd}{crd}".format(
            dcd = dcdir,
            pnt = "__".join(points),
            crd = "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else "ahtt_" + args.channel + '_' + args.year + ".txt"
        ))

    if (rungen or (args.savetoy and (rungof or runfc))) and args.ntoy > 0:
        if args.toyloc == "":
            # no toy location is given, dump the toys in some directory under datacard directory
            # in gof/fc, files are actually just copied off the files already generated, but this is probably easier to move off to some central storage
            timestamp_dir = os.path.join(dcdir, "toys_" + right_now())
            os.makedirs(timestamp_dir)
            args.toyloc = os.path.abspath(timestamp_dir) + "/"

    if rungen and args.ntoy > 0:
        print "\ntwin_point_ahtt :: starting toy generation"
        syscall("combine -v -1 -M GenerateOnly -d {dcd} -m {mmm} -n _{snm} --setParameters '{par}' {stg} {toy}".format(
                    dcd = dcdir + "workspace_twin-g.root",
                    mmm = mstr,
                    snm = "toygen_" + str(args.runidx) if not args.runidx < 0 else "toygen",
                    par = "g1=" + gvalues[0] + ",g2=" + gvalues[1],
                    stg = fit_strategy("0"),
                    toy = "-s -1 --toysFrequentist -t " + str(args.ntoy) + " --saveToys"
                ))

        syscall("mv higgsCombine_{snm}.GenerateOnly.mH{mmm}*.root {opd}{pnt}{tag}_toys{gvl}{fix}{toy}{idx}.root".format(
            opd = args.toyloc,
            snm = "toygen_" + str(args.runidx) if not args.runidx < 0 else "toygen",
            pnt = "__".join(points),
            tag = args.otag,
            gvl = "_" + gstr.replace(".", "p") if gstr != "" else "",
            fix = "_fixed" if args.fixpoi and gstr != "" else "",
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

        print "\ntwin_point_ahtt :: starting goodness of fit, saturated model - data fit"
        syscall("combine -v -1 -M GoodnessOfFit --algo=saturated -d {dcd} -m {mmm} -n _{snm} {stg}".format(
            dcd = dcdir + "workspace_twin-g.root",
            mmm = mstr,
            snm = "gof-saturated-data",
            stg = fit_strategy("0")
        ))

        print "\ntwin_point_ahtt :: starting goodness of fit, saturated model - toy fits"
        syscall("combine -v -1 -M GoodnessOfFit --algo=saturated -d {dcd} -m {mmm} -n _{snm} {stg} {toy} {opd} {svt}".format(
            dcd = dcdir + "workspace_twin-g.root",
            mmm = mstr,
            snm = "gof-saturated-toys",
            stg = fit_strategy("0"),
            toy = "-s -1 --toysFrequentist -t " + str(args.ntoy),
            opd = "--toysFile '" + args.toyloc + "'" if readtoy else "",
            svt = "--saveToys" if args.savetoy else ""
        ))

        print "\ntwin_point_ahtt :: goodness of fit - collecting results"
        # FIXME continue
        pass

    if runfc:
        if any(float(gg) < 0. for gg in gvalues):
            raise RuntimeError("in FC scans both g can't be negative!!")

        exp_scenario = OrderedDict()
        exp_scenario["exp-b"]  = "g1=0,g2=0"
        exp_scenario["exp-s"]  = "g1=1,g2=1"
        exp_scenario["exp-01"] = "g1=0,g2=1"
        exp_scenario["exp-10"] = "g1=1,g2=0"

        scan_name = "pnt_g1_" + gvalues[0] + "_g2_" + gvalues[1]

        if args.fcrundat:
            print "\ntwin_point_ahtt :: finding the best fit point for FC scan"

            for fcexp in args.fcexp:
                identifier = "_" + fcexp

                for istrat, irobust in [(istrat, irobust) for istrat in ["0", "1", "2"] for irobust in [False, True]]:
                    syscall("combineTool.py -v -1 -M MultiDimFit -d {dcd} -m {mmm} -n _{snm} --algo fixed --fixedPointPOIs '{par}' "
                            "--setParameters '{exp}{msk}' {stg} {asm} {toy} {wsp}".format(
                                dcd = dcdir + "workspace_twin-g.root",
                                mmm = mstr,
                                snm = scan_name + identifier,
                                par = "g1=" + gvalues[0] + ",g2=" + gvalues[1],
                                exp = exp_scenario[fcexp] if fcexp != "obs" else exp_scenario["exp-b"],
                                msk = "," + ",".join(masks) if len(masks) > 0 else "",
                                stg = fit_strategy(istrat, irobust),
                                asm = "-t -1" if fcexp != "obs" else "",
                                toy = "-s -1",
                                wsp = "--saveWorkspace --saveSpecifiedNuis=all" if False else ""
                            ))

                    if all([get_fit(glob.glob("higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root".format(snm = scan_name + identifier, mmm = mstr))[0], ff) is not None for ff in [True, False]]):
                        break
                    else:
                        syscall("rm higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root".format(snm = scan_name + identifier, mmm = mstr), False)

                syscall("mv higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root {dcd}{pnt}{tag}_fc-scan_{snm}.root".format(
                    dcd = dcdir,
                    pnt = "__".join(points),
                    tag = args.otag,
                    snm = scan_name + identifier,
                    mmm = mstr), False)

        if args.ntoy > 0:
            identifier = "_toys_" + str(args.runidx) if args.runidx > -1 else "_toys"
            print "\ntwin_point_ahtt :: performing the FC scan for toys"

            setpar, frzpar = ([], [])
            syscall("combineTool.py -v -1 -M MultiDimFit -d {dcd} -m {mmm} -n _{snm} --algo fixed --fixedPointPOIs '{par}' "
                    "--setParameters '{par}{msk}{nus}' {nuf} {stg} {toy} {opd} {byp} {svt}".format(
                        dcd = dcdir + "workspace_twin-g.root",
                        mmm = mstr,
                        snm = scan_name + identifier,
                        par = "g1=" + gvalues[0] + ",g2=" + gvalues[1],
                        msk = "," + ",".join(masks) if len(masks) > 0 else "",
                        nus = "",
                        nuf = "",
                        stg = fit_strategy("0"),
                        toy = "-s -1 --toysFrequentist -t " + str(args.ntoy),
                        opd = "--toysFile '" + args.toyloc + "'" if readtoy else "",
                        byp = "--bypassFrequentistFit --fastScan" if False else "",
                        svt = "--saveToys" if args.savetoy else ""
                    ))

            syscall("mv higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root {dcd}{pnt}{tag}_fc-scan_{snm}.root".format(
                dcd = dcdir,
                pnt = "__".join(points),
                tag = args.otag,
                snm = scan_name + identifier,
                mmm = mstr,
            ), False)

            if args.savetoy:
                syscall("cp {dcd}{pnt}{tag}_fc-scan_{snm}.root {opd}{pnt}{tag}_toys{gvl}{fix}{toy}{idx}.root".format(
                    opd = args.toyloc,
                    snm = scan_name + identifier,
                    pnt = "__".join(points),
                    tag = args.otag,
                    gvl = "_" + gstr.replace(".", "p") if gstr != "" else "",
                    fix = "_fixed" if args.fixpoi and gstr != "" else "",
                    toy = "_n" + str(args.ntoy),
                    idx = "_" + str(args.runidx) if not args.runidx < 0 else "",
                    mmm = mstr,
                ), False)

    if runhadd:
        idxs = glob.glob("{dcd}{pnt}{tag}_fc-scan_*_toys_*.root".format(dcd = dcdir, pnt = "__".join(points), tag = args.otag))

        if len(idxs) > 0:
            print "\ntwin_point_ahtt :: indexed toy files detected, merging them..."

            toys = glob.glob("{dcd}{pnt}{tag}_fc-scan_*_toys_*.root".format(dcd = dcdir, pnt = "__".join(points), tag = args.otag))
            toys = set([re.sub('toys_.*.root', 'toys.root', toy) for toy in toys])

            for toy in toys:
                syscall("mv {toy} {tox}".format(toy = toy, tox = toy.replace("toys.root", "toys_x.root")), False, True)
                syscall("hadd {toy} {tox} && rm {tox}".format(toy = toy, tox = toy.replace("toys.root", "toys_*.root")))

    if runcompile:
        toys = glob.glob("{dcd}{pnt}{tag}_fc-scan_*_toys.root".format(dcd = dcdir, pnt = "__".join(points), tag = args.otag))
        idxs = glob.glob("{dcd}{pnt}{tag}_fc-scan_*_toys_*.root".format(dcd = dcdir, pnt = "__".join(points), tag = args.otag))
        if len(toys) == 0 or len(idxs) > 0:
            print "\ntwin_point_ahtt :: either no merged toy files are present, or some indexed ones are."
            raise RuntimeError("run either the fc-scan or hadd modes first before proceeding!")

        print "\ntwin_point_ahtt :: compiling FC scan results..."
        for fcexp in args.fcexp:
            gpoints = glob.glob("{dcd}{pnt}{tag}_fc-scan_*{exp}.root".format(dcd = dcdir, pnt = "__".join(points), tag = args.otag, exp = "_" + fcexp))
            if len(gpoints) == 0:
                raise RuntimeError("result compilation can't proceed without the best fit files being available!!")
            gpoints.sort()

            ggrid = glob.glob("{dcd}{pnt}{tag}_fc-scan{exp}_*.json".format(dcd = dcdir, pnt = "__".join(points), tag = args.otag, exp = "_" + fcexp))
            ggrid.sort()
            idx = 0 if len(ggrid) == 0 else int(ggrid[-1].split("_")[-1].split(".")[0]) + 1

            best_fit = get_fit(gpoints[0])

            grid = OrderedDict()
            grid["points"] = points
            grid["best_fit_g1_g2_dnll"] = best_fit
            grid["g-grid"] = OrderedDict() if idx == 0 or args.ignoreprev else read_previous_grid(points, best_fit, ggrid[-1])

            for pnt in gpoints:
                if best_fit != get_fit(pnt):
                    print '\nWARNING :: incompatible best fit across different g values!! ignoring current, assuming it is due to numerical instability!'
                    print 'this should NOT happen too frequently within a single compilation, and the difference should not be large!!'
                    print "current result ", pnt, ": ", get_fit(pnt)
                    print "first result ", gpoints[0], ": ", best_fit

                bf = get_fit(pnt, False)
                if bf is None:
                    raise RuntimeError("failed getting best fit point for file " + pnt + ". aborting.")

                gg = get_toys(pnt.replace("{exp}.root".format(exp = "_" + fcexp), "_toys.root"), bf)
                gv = stringify((bf[0], bf[1]))

                if args.rmroot:
                    syscall("rm " + pnt, False, True)
                    if fcexp == args.fcexp[-1]:
                        syscall("rm " + pnt.replace("{exp}.root".format(exp = "_" + fcexp), "_toys.root"), False, True)

                if gv in grid["g-grid"]:
                    grid["g-grid"][gv] = sum_up(grid["g-grid"][gv], gg)
                else:
                    grid["g-grid"][gv] = gg

            grid["g-grid"] = OrderedDict(sorted(grid["g-grid"].items()))
            with open("{dcd}{pnt}{tag}_fc-scan{exp}_{idx}.json".format(dcd = dcdir, pnt = "__".join(points), tag = args.otag, exp = "_" + fcexp, idx = str(idx)), "w") as jj:
                json.dump(grid, jj, indent = 1)

    if runprepost:
        if args.frzbbp or args.frznui:
            best_fit_file = make_best_fit(dcdir, "workspace_twin-g.root", "__".join(points),
                                          args.asimov, fit_strategy("2", True) + " --robustHesse 1", poi_range,
                                          elementwise_add([starting_poi(gvalues, args.fixpoi), starting_nuisance(points, args.frzbb0)]), args.extopt, masks)
        set_freeze = elementwise_add([starting_poi(gvalues, args.fixpoi), starting_nuisance(points, args.frzbb0, args.frzbbp, args.frznui, best_fit_file)])

        print "\ntwin_point_ahtt :: making pre- and postfit plots and covariance matrices"
        syscall("combine -v -1 -M FitDiagnostics {dcd}workspace_twin-g.root --saveWithUncertainties --saveNormalizations --saveShapes --saveOverallShapes "
                "--plots -m {mmm} -n _prepost {stg} {prg} {asm} {prm}".format(
                    dcd = dcdir,
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

        syscall("mv fitDiagnostics_prepost.root {dcd}{pnt}{tag}_fitdiagnostics{gvl}{fix}.root".format(
            dcd = dcdir,
            pnt = "__".join(points),
            tag = args.otag,
            gvl = "_" + gstr.replace(".", "p") if gstr != "" else "",
            fix = "_fixed" if args.fixpoi and gstr != "" else "",
        ), False)

    if not os.path.isfile(best_fit_file):
        syscall("rm {bff}".format(bff = best_fit_file), False, True)

    if args.compress:
        syscall(("tar -czf {dcd}.tar.gz {dcd} && rm -r {dcd}").format(
            dcd = dcdir[:-1]
        ))
