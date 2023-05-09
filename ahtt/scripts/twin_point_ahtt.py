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

from utilspy import syscall, recursive_glob, make_timestamp_dir, directory_to_delete, max_nfile_per_dir
from utilspy import get_point, elementwise_add, tuplize, stringify
from utilscombine import read_nuisance, max_g, make_best_fit, starting_nuisance, fit_strategy, make_datacard_with_args, set_parameter

from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes
from argumentative import common_point, common_common, common_fit_pure, common_fit, make_datacard_pure, make_datacard_forwarded, common_2D, parse_args
from hilfemir import combine_help_messages

def get_fit(dname, qexp_eq_m1 = True):
    if not os.path.isfile(dname):
        return None

    dfile = TFile.Open(dname)
    dtree = dfile.Get("limit")

    bf = None
    for i in dtree:
        if (dtree.quantileExpected == -1. and qexp_eq_m1) or (dtree.quantileExpected != -1. and not qexp_eq_m1):
            bf = (dtree.g1, dtree.g2, dtree.deltaNLL if dtree.deltaNLL >= 0. else 0.)

        if bf is not None:
            break

    dfile.Close()
    return bf

def read_previous_best_fit(gname):
    with open(gname) as ff:
        result = json.load(ff, object_pairs_hook = OrderedDict)
    return tuple(result["best_fit_g1_g2_dnll"])

def read_previous_grid(points, prev_best_fit, gname):
    with open(gname) as ff:
        result = json.load(ff, object_pairs_hook = OrderedDict)

    if result["points"] == points and result["best_fit_g1_g2_dnll"] == list(prev_best_fit):
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

def get_toys(tname, best_fit, whatever_else = None):
    if not os.path.isfile(tname):
        return None

    pval = OrderedDict()

    tfile = TFile.Open(tname)
    ttree = tfile.Get("limit")

    isum = 0
    ipas = 0

    for i in ttree:
        if ttree.quantileExpected >= 0. and ttree.deltaNLL >= 0.:
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
        print '\nWARNING :: incompatible expected/data dnll, when they should be!! Using the first dnll: ', g1["dnll"], ', over the second: ', g2["dnll"]
        sys.stdout.flush()

    gs["total"] = g1["total"] + g2["total"]
    gs["pass"] = g1["pass"] + g2["pass"]
    gs["dnll"] = g1["dnll"]

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

    args = parse_args(parser)
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
        syscall("ValidateDatacards.py --jsonFile {dcd}{pnt}{tag}_validate.json --printLevel 3 {dcd}{crd}".format(
            dcd = dcdir,
            pnt = "__".join(points),
            tag = args.otag,
            crd = "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else "ahtt_" + args.channel + '_' + args.year + ".txt"
        ))

    if (rungen or (args.savetoy and (rungof or runfc))) and args.ntoy > 0:
        if args.toyloc == "":
            # no toy location is given, dump the toys in some directory under datacard directory
            args.toyloc = dcdir

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

        if args.fcresdir == "":
            args.fcresdir = dcdir

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

                for itol, irobust, istrat in [(itol, irobust, istrat) for itol in [0, 1, 2] for irobust in [False, True] for istrat in [0, 1, 2]]:
                    syscall("combineTool.py -v -1 -M MultiDimFit -d {dcd} -m {mmm} -n _{snm} --algo fixed --fixedPointPOIs '{par}' "
                            "--setParameters '{exp}{msk}' {stg} {asm} {toy} {wsp}".format(
                                dcd = dcdir + "workspace_twin-g.root",
                                mmm = mstr,
                                snm = scan_name + identifier,
                                par = "g1=" + gvalues[0] + ",g2=" + gvalues[1],
                                exp = exp_scenario[fcexp] if fcexp != "obs" else exp_scenario["exp-b"],
                                msk = "," + ",".join(masks) if len(masks) > 0 else "",
                                stg = fit_strategy(istrat, irobust, itol),
                                asm = "-t -1" if fcexp != "obs" else "",
                                toy = "-s -1",
                                wsp = "--saveWorkspace --saveSpecifiedNuis=all" if False else ""
                            ))

                    if all([get_fit(glob.glob("higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root".format(snm = scan_name + identifier, mmm = mstr))[0], ff) is not None for ff in [True, False]]):
                        break
                    else:
                        syscall("rm higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root".format(snm = scan_name + identifier, mmm = mstr), False)

                syscall("mv higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root {dcd}{pnt}{tag}_fc-scan_{snm}.root".format(
                    dcd = args.fcresdir,
                    pnt = "__".join(points),
                    tag = args.otag,
                    snm = scan_name + identifier,
                    mmm = mstr
                ), False)

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
                dcd = args.fcresdir,
                pnt = "__".join(points),
                tag = args.otag,
                snm = scan_name + identifier,
                mmm = mstr,
            ), False)

            if args.savetoy:
                syscall("cp {dcd}{pnt}{tag}_fc-scan_{snm}.root {opd}{pnt}{tag}_toys{gvl}{fix}{toy}{idx}.root".format(
                    dcd = dcdir,
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
        toys = recursive_glob(dcdir, "{pnt}{tag}_fc-scan_*_toys_*.root".format(pnt = "__".join(points), tag = args.otag))

        if len(toys) > 0:
            print "\ntwin_point_ahtt :: indexed toy files detected, merging them..."
            toys = set([os.path.basename(re.sub('toys_.*.root', 'toys.root', toy)) for toy in toys])

            mrgdir = ""
            for ii, toy in enumerate(toys):
                if ii % max_nfile_per_dir == 0:
                    mrgdir = make_timestamp_dir(base = dcdir, prefix = "fc-merge")

                tomerge = recursive_glob(dcdir, toy)
                if len(tomerge) > 0:
                    syscall("mv {toy} {tox}".format(toy = tomerge[0], tox = toy.replace("toys.root", "toys_x.root")), False, True)

                tomerge = recursive_glob(dcdir, toy.replace("toys.root", "toys_*.root"))
                for tm in tomerge:
                    if 'fc-result' in tm:
                        directory_to_delete(location = tm)

                syscall("hadd {toy} {tox} && rm {tox}".format(toy = mrgdir + toy, tox = " ".join(tomerge)))

            directory_to_delete(location = None, flush = True)

    if runcompile:
        toys = glob.glob("{dcd}{pnt}{tag}_fc-scan_*_toys.root".format(dcd = dcdir, pnt = "__".join(points), tag = args.otag))
        idxs = glob.glob("{dcd}{pnt}{tag}_fc-scan_*_toys_*.root".format(dcd = dcdir, pnt = "__".join(points), tag = args.otag))
        if len(toys) == 0 or len(idxs) > 0:
            print "\ntwin_point_ahtt :: either no merged toy files are present, or some indexed ones are."
            raise RuntimeError("run either the fc-scan or hadd modes first before proceeding!")

        print "\ntwin_point_ahtt :: compiling FC scan results..."
        for fcexp in args.fcexp:
            expfits = glob.glob("{dcd}{pnt}{tag}_fc-scan_*{exp}.root".format(
                dcd = dcdir,
                pnt = "__".join(points),
                tag = args.otag,
                exp = "_" + fcexp
            ))
            expfits.sort()

            previous_grids = glob.glob("{dcd}{pnt}{tag}_fc-scan{exp}_*.json".format(
                dcd = dcdir,
                pnt = "__".join(points),
                tag = args.otag,
                exp = "_" + fcexp
            ))
            previous_grids.sort(key = os.path.getmtime)

            if len(expfits) == 0 and (args.ignoreprev or len(previous_grids) == 0):
                raise RuntimeError("result compilation can't proceed without the best fit files or previous grids being available!!")
            gpoints = points_to_compile(points, expfits, None if args.ignoreprev else previous_grids[-1])
            idx = 0 if len(previous_grids) == 0 else int(previous_grids[-1].split("_")[-1].split(".")[0]) + 1
            best_fit = get_fit(expfits[0]) if len(expfits) > 0 else read_previous_best_fit(previous_grids[-1])

            grid = OrderedDict()
            grid["points"] = points
            grid["best_fit_g1_g2_dnll"] = best_fit
            grid["g-grid"] = OrderedDict() if idx == 0 or args.ignoreprev else read_previous_grid(points, best_fit, previous_grids[-1])

            for pnt in gpoints:
                gv = stringify(pnt)
                ename = "{dcd}{pnt}{tag}_fc-scan_pnt_g1_{g1}_g2_{g2}_{exp}.root".format(
                    dcd = dcdir,
                    pnt = "__".join(points),
                    tag = args.otag,
                    g1 = pnt[0],
                    g2 = pnt[1],
                    exp = fcexp
                )
                current_fit = get_fit(ename)
                expected_fit = get_fit(ename, False)

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

                tname = ename.replace("{exp}.root".format(exp = fcexp), "toys.root")
                gg = get_toys(tname, expected_fit)
                if args.rmroot:
                    syscall("rm " + ename, False, True)
                    if fcexp == args.fcexp[-1]:
                        syscall("rm " + tname, False, True)

                if gv in grid["g-grid"]:
                    grid["g-grid"][gv] = sum_up(grid["g-grid"][gv], gg)
                else:
                    grid["g-grid"][gv] = gg

            grid["g-grid"] = OrderedDict(sorted(grid["g-grid"].items()))
            with open("{dcd}{pnt}{tag}_fc-scan_{exp}_{idx}.json".format(
                    dcd = dcdir,
                    pnt = "__".join(points),
                    tag = args.otag,
                    exp = fcexp,
                    idx = str(idx)
            ), "w") as jj:
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
