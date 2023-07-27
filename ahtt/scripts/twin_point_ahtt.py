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

from utilspy import syscall, chunks, elementwise_add, recursive_glob, make_timestamp_dir, directory_to_delete, max_nfile_per_dir
from utilspy import get_point, tuplize, stringify, g_in_filename, floattopm
from utilscombine import max_g, get_best_fit, starting_nuisance, fit_strategy, make_datacard_with_args, set_range, set_parameter, nonparametric_option

from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes
from argumentative import common_point, common_common, common_fit_pure, common_fit, make_datacard_pure, make_datacard_forwarded, common_2D, parse_args
from hilfemir import combine_help_messages

def expected_scenario(exp, gvalues_syntax = False):
    specials = {
        "exp-b":  "g1=0,g2=0",
        "exp-s":  "g1=1,g2=1",
        "exp-01": "g1=0,g2=1",
        "exp-10": "g1=1,g2=0",
        "obs":    ""
    }

    if exp in specials:
        second = [str(float(ss)) for ss in tokenize_to_list(specials[exp].replace("g1=", "").replace("g2=", ""))] if gvalues_syntax else specials[exp]
        return (exp, second)

    if not re.search(r',[eo]', exp):
        gvalues = tokenize_to_list(exp)
        if len(gvalues) != 2 or not all([float(gg) >= 0. for gg in gvalues]):
            return None
        g1, g2 = gvalues
        second = [str(float(ss)) for ss in tokenize_to_list("{g1},{g2}".format(g1 = g1, g2 = g2))] if gvalues_syntax else "g1={g1},g2={g2}".format(g1 = g1, g2 = g2)

        return ("exp-{g1}-{g2}".format(g1 = round(float(g1), 5), g2 = round(float(g2), 5)), second)

    return None

def get_fit(dname, attributes, qexp_eq_m1 = True):
    if not os.path.isfile(dname):
        return None

    dfile = TFile.Open(dname)
    dtree = dfile.Get("limit")

    bf = None
    for i in dtree:
        if (dtree.quantileExpected == -1. and qexp_eq_m1) or (dtree.quantileExpected != -1. and not qexp_eq_m1):
            bf = [getattr(dtree, attr) for attr in attributes]
            if 'deltaNLL' in attributes:
                idx = attributes.index('deltaNLL')
                bf[idx] = max(bf[idx], 0.)
            bf = tuple(bf)

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
                "generate", "gof", "fc-scan", "contour",
                "hadd", "merge", "compile",
                "prepost", "corrmat",
                "nll", "likelihood"]
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
    runnll = "nll" in modes or "likelihood" in modes

    if len(args.fcexp) > 0 and not all([expected_scenario(exp) is not None for exp in args.fcexp]):
        print "given expected scenarii:", args.fcexp
        raise RuntimeError("unexpected expected scenario is given. aborting.")

    if (rungen or runfc) and any(float(gg) < 0. for gg in gvalues):
        raise RuntimeError("in toy generation or FC scans no g can be negative!!")

    # parameter ranges for best fit file
    ranges = ["{gg}: 0, 5".format(gg = gg) for gg in ["g1", "g2"]]
    if args.experimental:
        ranges += ["rgx{EWK_.*}", "rgx{QCDscale_ME.*}", "tmass"] # veeeery wide hedging for theory ME NPs

    if rundc:
        print "\ntwin_point_ahtt :: making datacard"
        make_datacard_with_args(scriptdir, args)

        print "\ntwin_point_ahtt :: making workspaces"
        syscall("combineTool.py -M T2W -i {dcd} -o workspace_twin-g.root -m {mmm} -P CombineHarvester.CombineTools.MultiInterferencePlusFixed:multiInterferencePlusFixed "
                "--PO 'signal={pnt}' {pos} {opt} {ext}".format(
                    dcd = dcdir + "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else dcdir + "ahtt_" + args.channel + '_' + args.year + ".txt",
                    mmm = mstr,
                    pnt = ",".join(points),
                    pos = " ".join(["--PO " + stuff for stuff in ["verbose", "no-r", "yukawa"]]),
                    opt = "--channel-masks --no-wrappers --X-pack-asympows --optimize-simpdf-constraints=cms --use-histsum",
                    ext = args.extopt
        ))

    if runvalid:
        print "\ntwin_point_ahtt :: validating datacard"
        syscall("ValidateDatacards.py --jsonFile {dcd}{ptg}_validate.json --printLevel 3 {dcd}{crd}".format(
            dcd = dcdir,
            ptg = ptag,
            crd = "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else "ahtt_" + args.channel + '_' + args.year + ".txt"
        ))

    default_workspace = dcdir + "workspace_twin-g.root"
    workspace = get_best_fit(
        dcdir, "__".join(points), [args.otag, args.tag],
        args.defaultwsp, args.keepbest, default_workspace, args.asimov, "",
        "{gvl}{fix}".format(gvl = gstr if gstr != "" else "", fix = "_fixed" if args.fixpoi and gstr != "" else ""),
        fit_strategy(args.fitstrat if args.fitstrat > -1 else 2, True, args.usehesse), set_range(ranges),
        elementwise_add([starting_poi(gvalues, args.fixpoi), starting_nuisance(args.frzzero, set())]), args.extopt, masks
    )

    if (rungen or (args.savetoy and (rungof or runfc))) and args.ntoy > 0:
        if args.toyloc == "":
            # no toy location is given, dump the toys in some directory under datacard directory
            args.toyloc = dcdir

    if rungen and args.ntoy > 0:
        print "\ntwin_point_ahtt :: starting toy generation"
        syscall("combine -v -1 -M GenerateOnly -d {dcd} -m {mmm} -n _{snm} --setParameters '{par}' {stg} {toy}".format(
                    dcd = workspace,
                    mmm = mstr,
                    snm = "toygen_" + str(args.runidx) if not args.runidx < 0 else "toygen",
                    par = "g1=" + gvalues[0] + ",g2=" + gvalues[1],
                    stg = fit_strategy(args.fitstrat if args.fitstrat > -1 else 0),
                    toy = "-s -1 --toysFrequentist -t " + str(args.ntoy) + " --saveToys"
                ))

        syscall("mv higgsCombine_{snm}.GenerateOnly.mH{mmm}*.root {opd}{ptg}_toys{gvl}{fix}{toy}{idx}.root".format(
            opd = args.toyloc,
            snm = "toygen_" + str(args.runidx) if not args.runidx < 0 else "toygen",
            ptg = ptag,
            gvl = "_" + gstr if gstr != "" else "",
            fix = "_fixed" if args.fixpoi and gstr != "" else "",
            toy = "_n" + str(args.ntoy),
            idx = "_" + str(args.runidx) if not args.runidx < 0 else "",
            mmm = mstr,
        ), False)

    if rungof or runfc:
        never_gonna_give_you_up = [(irobust, istrat, itol) for irobust in [False, True] for istrat in [0, 1, 2] for itol in range(4)]
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
        set_freeze = elementwise_add([starting_poi(gvalues, args.fixpoi), starting_nuisance(args.frzzero, args.frzpost)])

        if args.gofrundat:
            print "\ntwin_point_ahtt :: starting goodness of fit, saturated model - data fit"
            scan_name = "gof-saturated_obs"

            for irobust, istrat, itol in never_gonna_give_you_up:
                syscall("combine -v -1 -M GoodnessOfFit --algo=saturated -d {dcd} -m {mmm} -n _{snm} {stg} {prm} {ext}".format(
                    dcd = workspace,
                    mmm = mstr,
                    snm = scan_name,
                    stg = fit_strategy(istrat, irobust, irobust and args.usehesse, itol),
                    msk = ",".join(masks) if len(masks) > 0 else "",
                    prm = set_parameter(set_freeze, args.extopt, masks),
                    ext = nonparametric_option(args.extopt),
                ))
                if irobust and args.usehesse:
                    syscall("rm robustHesse_*.root", False, True)

                if get_fit(glob.glob("higgsCombine_{snm}.GoodnessOfFit.mH{mmm}*.root".format(snm = scan_name, mmm = mstr))[0], ['limit'], True):
                    break
                else:
                    syscall("rm higgsCombine_{snm}.GoodnessOfFit.mH{mmm}*.root".format(snm = scan_name, mmm = mstr), False)

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

            syscall("combine -v -1 -M GoodnessOfFit --algo=saturated -d {dcd} -m {mmm} -n _{snm} {stg} {toy} {opd} {svt} {prm} {ext}".format(
                dcd = workspace,
                mmm = mstr,
                snm = scan_name,
                stg = fit_strategy(args.fitstrat if args.fitstrat > -1 else 0),
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
                scenario = expected_scenario(fcexp)
                identifier = "_" + scenario[0]
                parameters = [scenario[1]] if scenario[1] != "" else [] 

                # fit settings should be identical to the one above, since we just want to choose the wsp by fcexp rather than args.asimov
                fcwsp = get_best_fit(
                    dcdir, "__".join(points), [args.otag, args.tag],
                    args.defaultwsp, args.keepbest, default_workspace, fcexp != "obs", "",
                    "{gvl}{fix}".format(gvl = gstr if gstr != "" else "", fix = "_fixed" if args.fixpoi and gstr != "" else ""),
                    fit_strategy(args.fitstrat if args.fitstrat > -1 else 2, True, args.usehesse), set_range(ranges),
                    elementwise_add([starting_poi(gvalues, args.fixpoi), starting_nuisance(args.frzzero, set())]), args.extopt, masks
                )

                for irobust, istrat, itol in never_gonna_give_you_up:
                    syscall("combineTool.py -v -1 -M MultiDimFit -d {dcd} -m {mmm} -n _{snm} --algo fixed --fixedPointPOIs '{par}' "
                            "{exp} {stg} {asm} {toy} {ext}".format(
                                dcd = fcwsp,
                                mmm = mstr,
                                snm = scan_name + identifier,
                                par = "g1=" + gvalues[0] + ",g2=" + gvalues[1],
                                exp = "--setParameters '" + ",".join(parameters + masks) + "'" if len(parameters + masks) > 0 else "",
                                stg = fit_strategy(istrat, irobust, irobust and args.usehesse, itol),
                                asm = "-t -1" if fcexp != "obs" else "",
                                toy = "-s -1",
                                ext = nonparametric_option(args.extopt),
                                #wsp = "--saveWorkspace --saveSpecifiedNuis=all" if False else ""
                            ))
                    if irobust and args.usehesse:
                        syscall("rm robustHesse_*.root", False, True)

                    if all([get_fit(glob.glob("higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root".format(
                            snm = scan_name + identifier,
                            mmm = mstr))[0], ['g1', 'g2', 'deltaNLL'], ff) is not None for ff in [True, False]]):
                        break
                    else:
                        syscall("rm higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root".format(snm = scan_name + identifier, mmm = mstr), False)

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
            syscall("combineTool.py -v -1 -M MultiDimFit -d {dcd} -m {mmm} -n _{snm} --algo fixed --fixedPointPOIs '{par}' "
                    "--setParameters '{par}{msk}{nus}' {nuf} {stg} {toy} {ext} {opd} {svt}".format(
                        dcd = workspace,
                        mmm = mstr,
                        snm = scan_name + identifier,
                        par = "g1=" + gvalues[0] + ",g2=" + gvalues[1],
                        msk = "," + ",".join(masks) if len(masks) > 0 else "",
                        nus = "",
                        nuf = "",
                        stg = fit_strategy(args.fitstrat if args.fitstrat > -1 else 0),
                        toy = "-s -1 --toysFrequentist -t " + str(args.ntoy),
                        ext = nonparametric_option(args.extopt),
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

    if runhadd:
        toys = recursive_glob(dcdir, "{ptg}_*_toys_*.root".format(ptg = ptag))

        if len(toys) > 0:
            print "\ntwin_point_ahtt :: indexed toy files detected, merging them..."
            toys = set([os.path.basename(re.sub('toys_.*.root', 'toys.root', toy)) for toy in toys])

            mrgdir = ""
            for ii, toy in enumerate(toys):
                if ii % max_nfile_per_dir == 0:
                    mrgdir = make_timestamp_dir(base = dcdir, prefix = "{mode}-merge".format(mode = "fc" if "fc-" in toy else "gof"))
                    directory_to_delete(location = mrgdir)

                tomerge = recursive_glob(dcdir, toy)
                if len(tomerge) > 0:
                    syscall("mv {toy} {tox}".format(toy = tomerge[0], tox = toy.replace("toys.root", "toys_x.root")), False, True)
                    for tm in tomerge:
                        directory_to_delete(location = tm)

                tomerge = recursive_glob(dcdir, toy.replace("toys.root", "toys_*.root"))
                for tm in tomerge:
                    if '-result' in tm:
                        directory_to_delete(location = tm)

                jj = 0
                while len(tomerge) > 0:
                    if len(tomerge) > 1:
                        about_right = 200 # arbitrary, only to prevent commands getting too long
                        tomerge = chunks(tomerge, len(tomerge) // about_right)
                        merged = []

                        for itm, tm in enumerate(tomerge):
                            mname = mrgdir + toy.replace("toys.root", "toys_{jj}-{itm}.root".format(jj = jj, itm = itm))
                            while os.path.isfile(mname):
                                jj += 1
                                mname = mrgdir + toy.replace("toys.root", "toys_{jj}-{itm}.root".format(jj = jj, itm = itm))
                            syscall("hadd {toy} {tox} && rm {tox}".format(toy = mname, tox = " ".join(tm)))
                            merged.append(mname)

                        jj += 1
                        tomerge = merged
                        continue

                    elif len(tomerge) == 1:
                        syscall("mv {tox} {toy}".format(
                            tox = tomerge[0],
                            toy = re.sub('toys_.*.root', 'toys.root', tomerge[0])
                        ), False, True)
                        tomerge = []

            directory_to_delete(location = None, flush = True)

    if runcompile:
        toys = recursive_glob(dcdir, "{ptg}_fc-scan_*_toys.root".format(ptg = ptag))
        idxs = recursive_glob(dcdir, "{ptg}_fc-scan_*_toys_*.root".format(ptg = ptag))
        if len(toys) == 0 or len(idxs) > 0:
            print "\ntwin_point_ahtt :: either no merged toy files are present, or some indexed ones are."
            raise RuntimeError("run either the fc-scan or hadd modes first before proceeding!")

        print "\ntwin_point_ahtt :: compiling FC scan results..."
        for fcexp in args.fcexp:
            scenario = expected_scenario(fcexp)
            expfits = recursive_glob(dcdir, "{ptg}_fc-scan_*_{exp}.root".format(ptg = ptag, exp = scenario[0]))
            expfits.sort()

            previous_grids = glob.glob("{dcd}{ptg}_fc-scan_{exp}_*.json".format(
                dcd = dcdir,
                ptg = ptag,
                exp = scenario[0]
            ))
            previous_grids.sort(key = os.path.getmtime)
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
                ))[0]
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

                tname = recursive_glob(dcdir, os.path.basename(ename).replace("{exp}.root".format(exp = scenario[0]), "toys.root"))[0]
                gg = get_toys(tname, expected_fit)
                if args.rmroot:
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

    if runprepost:
        for scenario in [(("0.", "0."), True, "--skipSBFit", "b"), (gvalues, args.fixpoi, "--skipBOnlyFit", "s")]:
            gtofit, fixpoi, fitopt, fittag = scenario
            gfit = g_in_filename(gtofit)
            set_freeze = elementwise_add([starting_poi(gtofit, fixpoi), starting_nuisance(args.frzzero, args.frzpost)])

            print "\ntwin_point_ahtt :: making pre- and postfit plots and covariance matrices (" + fittag + ")"
            syscall("combine -v -1 -M FitDiagnostics {dcd} --saveWithUncertainties --saveNormalizations --saveShapes --saveOverallShapes "
                    "--plots -m {mmm} -n _prepost {stg} {asm} {prm} {ext} {opt}".format(
                        dcd = workspace,
                        mmm = mstr,
                        stg = fit_strategy(args.fitstrat if args.fitstrat > -1 else 2, True, args.usehesse),
                        asm = "-t -1" if args.asimov else "",
                        prm = set_parameter(set_freeze, args.extopt, masks),
                        ext = nonparametric_option(args.extopt),
                        opt = fitopt
                    ))

            syscall("rm *_th1x_*.png", False, True)
            syscall("rm covariance_fit_?.png", False, True)
            syscall("rm higgsCombine_prepost*.root", False, True)
            syscall("rm combine_logger.out", False, True)
            syscall("rm robustHesse_*.root", False, True)

            syscall("mv fitDiagnostics_prepost.root {dcd}{ptg}_fitdiagnostics_{ftg}{gvl}{fix}.root".format(
                dcd = dcdir,
                ptg = ptag,
                ftg = fittag,
                gvl = "_" + gfit if gfit != "" else "",
                fix = "_fixed" if fixpoi and gfit != "" else "",
            ), False)

    if runnll:
        if len(args.nllparam) < 1:
            raise RuntimeError("what parameter is the nll being scanned against??")

        if len(args.fcexp) != 1:
            raise RuntimeError("in mode nll only one expected scenario is allowed.")

        print "\ntwin_point_ahtt :: evaluating nll as a function of {nlp}".format(nlp = ", ".join(args.nllparam))

        nparam = len(args.nllparam)
        scenario = expected_scenario(args.fcexp[0], True)
        set_freeze = elementwise_add([starting_poi(scenario[1] if args.fcexp[0] != "obs" else gvalues, args.fixpoi), starting_nuisance(args.frzzero, args.frzpost)])

        # fit settings should be identical to the one above, since we just want to choose the wsp by fcexp rather than args.asimov
        fcwsp = get_best_fit(
            dcdir, "__".join(points), [args.otag, args.tag],
            args.defaultwsp, args.keepbest, default_workspace, args.fcexp[0] != "obs", "",
            "{gvl}{fix}".format(gvl = gstr if gstr != "" else "", fix = "_fixed" if args.fixpoi and gstr != "" else ""),
            fit_strategy(args.fitstrat if args.fitstrat > -1 else 2, True, args.usehesse), set_range(ranges),
            elementwise_add([starting_poi(gvalues, args.fixpoi), starting_nuisance(args.frzzero, set())]), args.extopt, masks
        )

        isgah = [param in ["g1", "g2"] for param in args.nllparam]
        if len(args.nllwindow) < nparam:
            args.nllwindow += ["0,3" if isg else "-5,5" for isg in isgah[len(args.nllwindow):]]
        minmax = [(float(values.split(",")[0]), float(values.split(",")[1])) for values in args.nllwindow]

        if len(args.nllnpnt) < nparam:
            nsample = 32 # per unit interval
            args.nllnpnt += [nsample * round(minmax[ii][1] - minmax[ii][0]) for ii in range(len(args.nllnpnt), nparam)]
            args.nllnpnt += [2 * args.nllnpnt[ii] if isgah[ii] else args.nllnpnt[ii] for ii in range(len(args.nllnpnt), nparam)]
        interval = [list(np.linspace(minmax[ii][0], minmax[ii][1], num = args.nllnpnt[ii if ii < nparam else -1])) for ii in range(nparam)]

        nelement = 0
        for element in itertools.product(*interval):
            nelement += 1
            nllpnt = ",".join(["{param}={value}".format(param = param, value = value) for param, value in zip(args.nllparam, element)])
            nllname = args.fcexp[0] + "_".join([""] + ["{param}_{value}".format(param = param, value = floattopm(round(value, 5))) for param, value in zip(args.nllparam, element)])

            syscall("combineTool.py -v -1 -M MultiDimFit -d {dcd} -m {mmm} -n _{snm} --algo fixed --fixedPointPOIs '{pnt}' {par} "
                    "{exp} {stg} {asm} {ext} --saveNLL --X-rtd REMOVE_CONSTANT_ZERO_POINT=1".format(
                        dcd = fcwsp,
                        mmm = mstr,
                        snm = nllname,
                        pnt = nllpnt,
                        par = "--redefineSignalPOIs '{param}'".format(param = ",".join(args.nllparam)),
                        exp = set_parameter(set_freeze, args.extopt, masks),
                        stg = fit_strategy(args.fitstrat if args.fitstrat > -1 else 1, True, args.usehesse),
                        asm = "-t -1" if args.fcexp[0] != "obs" else "",
                        ext = nonparametric_option(args.extopt),
                    ))
            if args.usehesse:
                syscall("rm robustHesse_*.root", False, True)

        if nelement > 1:
            syscall("hadd {dcd}{ptg}_nll_{exp}_{fit}.root higgsCombine_{exp}_{par}*MultiDimFit.mH{mmm}.root && rm higgsCombine_{exp}_{par}*MultiDimFit.mH{mmm}.root".format(
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
