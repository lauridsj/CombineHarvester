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

from utilities import syscall, get_point, read_nuisance, max_g, make_best_fit, starting_nuisance, elementwise_add, stringify, fit_strategy

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
    parser.add_argument("--point", help = "desired signal point to run on, comma separated", default = "", required = True)
    parser.add_argument("--mode", help = "combine mode to run, comma separated", default = "datacard", required = False)

    parser.add_argument("--signal", help = "signal filenames. comma separated", default = "../input/ll_sig.root", required = False)
    parser.add_argument("--background", help = "data/background filenames. comma separated", default = "../input/ll_bkg.root", required = False)

    parser.add_argument("--channel", help = "final state channels considered in the analysis. datacard only. comma separated",
                        default = "ee,em,mm,e3j,e4pj,m3j,m4pj", required = False)
    parser.add_argument("--year", help = "analysis year determining the correlation model to assume. datacard only. comma separated",
                        default = "2016pre,2016post,2017,2018", required = False)

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
    parser.add_argument("--no-mc-stats",
                        help = "don't add nuisances due to limited mc stats (barlow-beeston lite) in datacard mode, "
                        "or don't add the bb-lite analytical minimization option in others",
                        dest = "mcstat", action = "store_false", required = False)
    parser.add_argument("--float-rate",
                        help = "semicolon separated list of processes to make the rate floating for, using combine's rateParam directive.\n"
                        "syntax: proc1:min1,max1;proc2:min2,max2; ... procN:minN,maxN. min and max can be omitted, they default to 0,2.\n"
                        "the implementation assumes a single rate parameter across all channels.\n"
                        "it also automatically replaces the now-redundant CMS_[process]_norm_13TeV nuisances.\n"
                        "relevant only in the datacard step.",
                        dest = "rateparam", default = "", required = False)
    parser.add_argument("--mask", help = "channel_year combinations to be masked in statistical analysis commands. comma separated",
                        default = "", required = False)

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

    parser.add_argument("--unblind", help = "use data when fitting", dest = "asimov", action = "store_false", required = False)
    #parser.add_argument("--no-r", help = "use physics model without r accompanying g", dest = "nor", action = "store_true", required = False)

    parser.add_argument("--g-values", help = "the two values of g to e.g. do the FC grid scan for, comma separated",
                        default = "-1., -1.", dest = "gvalues", required = False)
    parser.add_argument("--fix-poi", help = "fix pois in the fit, through --g-values",
                        dest = "fixpoi", action = "store_true", required = False)

    parser.add_argument("--n-toy", help = "number of toys to throw per point when generating or performing FC scans",
                        default = 50, dest = "ntoy", required = False, type = int)
    parser.add_argument("--run-idx", help = "index to append to a given toy generation/FC scan run",
                        default = -1, dest = "runidx", required = False, type = int)
    parser.add_argument("--toy-location", help = "directory to dump the toys in mode generate, "
                        "and file to read them from in mode contour/gof.",
                        dest = "toyloc", default = "", required = False)

    parser.add_argument("--fc-expect", help = "expected scenarios to assume in the FC scan. comma separated.\n"
                        "exp-b -> g1 = g2 = 0; exp-s -> g1 = g2 = 1; exp-01 -> g1 = 0, g2 = 1; exp-10 -> g1 = 1, g2 = 0",
                        default = "exp-b", dest = "fcexp", required = False)
    parser.add_argument("--fc-nuisance-mode", help = "how to handle nuisance parameters in toy generation (see https://arxiv.org/abs/2207.14353)\n"
                        "WARNING: profile mode implementation is incomplete!!",
                        default = "conservative", dest = "fcnui", required = False)
    parser.add_argument("--fc-skip-data", help = "skip running on data/asimov", dest = "fcrundat", action = "store_false", required = False)

    parser.add_argument("--delete-root", help = "delete root files after compiling", dest = "rmroot", action = "store_true", required = False)
    parser.add_argument("--ignore-previous", help = "ignore previous grid when compiling", dest = "ignoreprev", action = "store_true", required = False)

    parser.add_argument("--freeze-mc-stats-zero",
                        help = "only in the prepost/corrmat mode, freeze mc stats nuisances to zero",
                        dest = "frzbb0", action = "store_true", required = False)
    parser.add_argument("--freeze-mc-stats-post",
                        help = "only in the prepost/corrmat mode, freeze mc stats nuisances to the postfit values. "
                        "--freeze-mc-stats-zero takes priority over this option",
                        dest = "frzbbp", action = "store_true", required = False)
    parser.add_argument("--freeze-nuisance-post", help = "only in the prepost/corrmat mode, freeze all nuisances to the postfit values.",
                        dest = "frznui", action = "store_true", required = False)

    parser.add_argument("--seed",
                        help = "random seed to be used for pseudodata generation. give 0 to read from machine, and negative values to use no rng",
                        default = "", required = False)
    parser.add_argument("--compress", help = "compress output into a tar file", dest = "compress", action = "store_true", required = False)
    parser.add_argument("--base-directory",
                        help = "in non-datacard modes, this is the location where datacard is searched for, and output written to",
                        dest = "basedir", default = "", required = False)

    args = parser.parse_args()
    print "twin_point_ahtt :: called with the following arguments"
    print sys.argv[1:]
    print "\n"
    print " ".join(sys.argv)
    print "\n"
    sys.stdout.flush()

    if (args.tag != "" and not args.tag.startswith("_")):
        args.tag = "_" + args.tag

    if args.injectsignal != "":
        args.pseudodata = True

    points = args.point.replace(" ", "").split(',')
    gvalues = args.gvalues.replace(" ", "").split(',')
    if len(points) != 2 or len(gvalues) != 2:
        raise RuntimeError("this script is to be used with exactly two A/H points!")

    modes = args.mode.replace(" ", "").split(',')
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    args.basedir += "" if args.basedir == "" or args.basedir.endswith("/") else "/"
    args.toyloc += "" if args.toyloc == "" or args.toyloc.endswith("/") or args.toyloc.endswith(".root") else "/"
    dcdir = args.basedir + "__".join(points) + args.tag + "/"

    mstr = str(get_point(points[0])[1]).replace(".0", "")
    poi_range = "--setParameterRanges '" + ":".join(["g" + str(ii + 1) + "=0.,5." for ii, pp in enumerate(points)]) + "'"
    best_fit_file = ""
    masks = [] if args.mask == "" else args.mask.replace(" ", "").split(',')
    masks = ["mask_" + mm + "=1" for mm in masks]
    print "the following channel x year combinations will be masked:", masks

    gstr = ""
    for ii, gg in enumerate(gvalues):
        usc = "_" if ii > 0 else ""
        gstr += usc + "g" + str(ii + 1) + "_" + gvalues[ii] if float(gvalues[ii]) >= 0. else ""

    allmodes = ["datacard", "workspace", "validate",
                "generate", "fc-scan", "contour",
                "hadd", "merge", "compile",
                "prepost", "corrmat"]
    if (not all([mm in allmodes for mm in modes])):
        print "supported modes:", allmodes
        raise RuntimeError("unxpected mode is given. aborting.")

    # determine what to do with workspace, and do it
    rundc = "datacard" in modes or "workspace" in modes
    runvalid = "validate" in modes
    rungen = "generate" in modes
    runfc = "fc-scan" in modes or "contour" in modes
    runhadd = "hadd" in modes or "merge" in modes
    runcompile = "compile" in args.mode
    runprepost = "prepost" in modes or "corrmat" in modes

    allexp = ["exp-b", "exp-s", "exp-01", "exp-10"]
    fcexps = args.fcexp.replace(" ", "").split(',')
    if len(fcexps) == 0 or not all([exp in allexp for exp in fcexps]):
        print "supported expected scenario:", allexp
        raise RuntimeError("unexpected expected scenario is given. aborting.")

    allnui = ["conservative", "profile"]
    if not args.fcnui in allnui:
        print "supported nuisance modes:", allnui
        raise RuntimeError("unexpected nuisance mode is given. aborting.")
    elif args.fcnui == "profile":
        raise RuntimeError("nuisance mode profile is no longer pursued at the moment, and so is not allowed.")

    if not args.asimov:
        fcexps.append("obs")

    if rundc:
        print "\ntwin_point_ahtt :: making datacard"
        syscall("{scr}/make_datacard.py --signal {sig} --background {bkg} --point {pnt} --channel {ch} --year {yr} "
                "{psd} {inj} {tag} {drp} {kee} {kfc} {thr} {lns} {shp} {mcs} {rpr} {prj} {rsd}".format(
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
                    rpr = "--float-rate '" + args.rateparam + "'" if args.rateparam != "" else "",
                    prj = "--projection '" + args.projection + "'" if args.projection != "" else "",
                    rsd = "--seed " + args.seed if args.seed != "" else ""
                ))

        print "\ntwin_point_ahtt :: making workspaces"
        syscall("combineTool.py -M T2W -i {dcd} -o workspace_twin-g.root -m {mmm} -P CombineHarvester.CombineTools.MultiInterferencePlusFixed:multiInterferencePlusFixed "
                "--PO verbose --PO 'signal={pnt}' --PO no-r --channel-masks".format(
                    dcd = dcdir + "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else dcdir + "ahtt_" + args.channel + '_' + args.year + ".txt",
                    mmm = mstr,
                    pnt = args.point.replace(" ", "")
                ))

    if runvalid:
        print "\ntwin_point_ahtt :: validating datacard"
        syscall("ValidateDatacards.py --jsonFile {dcd}{pnt}_validate.json --printLevel 3 {dcd}{crd}".format(
            dcd = dcdir,
            pnt = "__".join(points),
            crd = "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else "ahtt_" + args.channel + '_' + args.year + ".txt"
        ))

    if rungen and args.ntoy > 0:
        print "\ntwin_point_ahtt :: starting toy generation"
        syscall("combine -v -1 -M GenerateOnly -d {dcd} -m {mmm} -n _{snm} --setParameters '{par}' {stg} {toy} {mcs}".format(
                    dcd = dcdir + "workspace_twin-g.root",
                    mmm = mstr,
                    snm = "toygen_" + str(args.runidx) if not args.runidx < 0 else "toygen",
                    par = "g1=" + gvalues[0] + ",g2=" + gvalues[1],
                    stg = fit_strategy("2"),
                    toy = "-s -1 --toysFrequentist -t " + str(args.ntoy) + " --saveToys",
                    mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else ""
                ))

        syscall("mv higgsCombine_{snm}.GenerateOnly.mH{mmm}*.root {opd}{pnt}_toys{gvl}{fix}{toy}{idx}.root".format(
            opd = args.toyloc if args.toyloc != "" else dcdir,
            snm = "toygen_" + str(args.runidx) if not args.runidx < 0 else "toygen",
            pnt = "__".join(points),
            gvl = "_" + gstr.replace(".", "p") if gstr != "" else "",
            fix = "_fixed" if args.fixpoi and gstr != "" else "",
            toy = "_n" + str(args.ntoy),
            idx = "_" + str(args.runidx) if not args.runidx < 0 else "",
            mmm = mstr,
        ), False)

    if runfc:
        if any(float(gg) < 0. for gg in gvalues):
            raise RuntimeError("in FC scans both g can't be negative!!")

        exp_scenario = OrderedDict()
        exp_scenario["exp-b"]  = "g1=0,g2=0"
        exp_scenario["exp-s"]  = "g1=1,g2=1"
        exp_scenario["exp-01"] = "g1=0,g2=1"
        exp_scenario["exp-10"] = "g1=1,g2=0"

        scan_name = "pnt_g1_" + gvalues[0] + "_g2_" + gvalues[1]
        snapshot = "fc_scan_snap.root"
        snapexp = "obs" if not args.asimov else "exp-s" if "exp-s" in fcexps else fcexps[0]

        if args.fcrundat or args.fcnui == "profile":
            print "\ntwin_point_ahtt :: finding the best fit point for FC scan"

            for fcexp in fcexps:
                identifier = "_" + fcexp

                for ifit in ["0", "1", "2"]:
                    syscall("combineTool.py -v -1 -M MultiDimFit -d {dcd} -m {mmm} -n _{snm} --algo fixed --fixedPointPOIs '{par}' "
                            "--setParameters '{exp}{msk}' {stg} {asm} {toy} {mcs} {wsp}".format(
                                dcd = dcdir + "workspace_twin-g.root",
                                mmm = mstr,
                                snm = scan_name + identifier,
                                par = "g1=" + gvalues[0] + ",g2=" + gvalues[1],
                                exp = exp_scenario[fcexp] if fcexp != "obs" else exp_scenario["exp-b"],
                                msk = "," + ",".join(masks) if len(masks) > 0 else "",
                                stg = fit_strategy(ifit),
                                asm = "-t -1" if fcexp != "obs" else "",
                                toy = "-s -1",
                                mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else "",
                                wsp = "--saveWorkspace --saveSpecifiedNuis=all" if args.fcnui == "profile" and fcexp == snapexp else ""
                            ))

                    if all([get_fit(glob.glob("higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root".format(snm = scan_name + identifier, mmm = mstr))[0], ff) is not None for ff in [True, False]]):
                        break
                    else:
                        syscall("rm higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root".format(snm = scan_name + identifier, mmm = mstr), False)

                if args.fcnui == "profile" and fcexp == snapexp:
                    syscall("cp higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root {snp}".format(
                        snm = scan_name + identifier,
                        mmm = mstr,
                        snp = snapshot), False)

                if args.fcrundat:
                    syscall("mv higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root {dcd}fc_scan_{snm}.root".format(
                        snm = scan_name + identifier,
                        mmm = mstr,
                        dcd = dcdir), False)
                else:
                    syscall("rm higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root".format(snm = scan_name + identifier, mmm = mstr), False)

        if args.ntoy > 0:
            identifier = "_toys_" + str(args.runidx) if args.runidx > -1 else "_toys"
            print "\ntwin_point_ahtt :: performing the FC scan for toys"

            readtoy = args.toyloc.endswith(".root")
            if readtoy:
                for ftoy in args.toyloc.split("_"):
                    if ftoy.startswith("n"):
                        ftoy = int(ftoy.replace("n", "").replace(".root", ""))
                        break

                if ftoy < args.ntoy:
                    print "\nWARNING :: file", args.toyloc, "contains less toys than requested in the run!"

            setpar, frzpar = read_nuisance(snapshot, points, False) if args.fcnui == "profile" else ([], [])
            syscall("combineTool.py -v -1 -M MultiDimFit -d {dcd} -m {mmm} -n _{snm} --algo fixed --fixedPointPOIs '{par}' "
                    "--setParameters '{par}{msk}{nus}' {nuf} {stg} {toy} {opd} {mcs} {byp}".format(
                        dcd = snapshot if args.fcnui == "profile" else dcdir + "workspace_twin-g.root",
                        mmm = mstr,
                        snm = scan_name + identifier,
                        par = "g1=" + gvalues[0] + ",g2=" + gvalues[1],
                        msk = "," + ",".join(masks) if len(masks) > 0 else "",
                        #nus = "," + ",".join(setpar) if args.fcnui == "profile" and len(setpar) > 0 else "",
                        nus = "",
                        nuf = "",
                        #nuf = "--freezeParameters '" + ",".join(frzpar) + "'" if args.fcnui == "profile" and len(frzpar) > 0 else "",
                        stg = fit_strategy("0"),
                        toy = "-s -1 --toysFrequentist -t " + str(args.ntoy),
                        opd = "--toysFile '" + args.toyloc + "'" if readtoy else "",
                        mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else "",
                        byp = "--bypassFrequentistFit --fastScan" if args.fcnui == "profile" else "",
                    ))

            syscall("mv higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root {dcd}fc_scan_{snm}.root".format(
                dcd = dcdir,
                snm = scan_name + identifier,
                mmm = mstr,
            ), False)
            syscall("rm {snp}".format(snp = snapshot), False, True)

    if runhadd:
        idxs = glob.glob("{dcd}fc_scan_*_toys_*.root".format(dcd = dcdir))

        if len(idxs) > 0:
            print "\ntwin_point_ahtt :: indexed toy files detected, merging them..."

            toys = glob.glob("{dcd}fc_scan_*_toys_*.root".format(dcd = dcdir))
            toys = set([re.sub('toys_.*.root', 'toys.root', toy) for toy in toys])

            for toy in toys:
                syscall("mv {toy} {tox}".format(toy = toy, tox = toy.replace("toys.root", "toys_x.root")), False, True)
                syscall("hadd {toy} {tox} && rm {tox}".format(toy = toy, tox = toy.replace("toys.root", "toys_*.root")))

    if runcompile:
        toys = glob.glob("{dcd}fc_scan_*_toys.root".format(dcd = dcdir))
        idxs = glob.glob("{dcd}fc_scan_*_toys_*.root".format(dcd = dcdir))
        if len(toys) == 0 or len(idxs) > 0:
            print "\ntwin_point_ahtt :: either no merged toy files are present, or some indexed ones are."
            raise RuntimeError("run either the fc-scan or hadd modes first before proceeding!")

        print "\ntwin_point_ahtt :: compiling FC scan results..."
        for fcexp in fcexps:
            gpoints = glob.glob("{dcd}fc_scan_*{exp}.root".format(dcd = dcdir, exp = "_" + fcexp))
            if len(gpoints) == 0:
                raise RuntimeError("result compilation can't proceed without the best fit files being available!!")
            gpoints.sort()

            ggrid = glob.glob("{dcd}{pnt}_fc_scan{exp}_*.json".format(dcd = dcdir, pnt = "__".join(points), exp = "_" + fcexp))
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
                    if fcexp == fcexps[-1]:
                        syscall("rm " + pnt.replace("{exp}.root".format(exp = "_" + fcexp), "_toys.root"), False, True)

                if gv in grid["g-grid"]:
                    grid["g-grid"][gv] = sum_up(grid["g-grid"][gv], gg)
                else:
                    grid["g-grid"][gv] = gg

            grid["g-grid"] = OrderedDict(sorted(grid["g-grid"].items()))
            with open("{dcd}{pnt}_fc_scan{exp}_{idx}.json".format(dcd = dcdir, pnt = "__".join(points), exp = "_" + fcexp, idx = str(idx)), "w") as jj:
                json.dump(grid, jj, indent = 1)

    if runprepost:
        if args.frzbbp or args.frznui:
            best_fit_file = make_best_fit(dcdir, "workspace_twin-g.root", "__".join(points),
                                          args.asimov, args.mcstat, strategy, poi_range,
                                          elementwise_add([starting_poi(gvalues, args.fixpoi), starting_nuisance(points, args.frzbb0)]))

        args.mcstat = args.mcstat or args.frzbb0 or args.frzbbp
        set_freeze = elementwise_add([starting_poi(gvalues, args.fixpoi), starting_nuisance(points, args.frzbb0, args.frzbbp, args.frznui, best_fit_file)])
        setpar = set_freeze[0]
        frzpar = set_freeze[1]

        print "\ntwin_point_ahtt :: making pre- and postfit plots and covariance matrices"
        syscall("combine -v -1 -M FitDiagnostics {dcd}workspace_twin-g.root --saveWithUncertainties --saveNormalizations --saveShapes --saveOverallShapes "
                "--plots -m {mmm} -n _prepost {stg} {prg} {asm} {mcs} {stp} {frz}".format(
                    dcd = dcdir,
                    mmm = mstr,
                    stg = fit_strategy("2") + " --robustFit 1 --setRobustFitStrategy 2 --robustHesse 1",
                    prg = poi_range,
                    asm = "-t -1" if args.asimov else "",
                    mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else "",
                    stp = "--setParameters '" + ",".join(setpar + masks) + "'" if len(setpar + masks) > 0 else "",
                    frz = "--freezeParameters '" + ",".join(frzpar) + "'" if len(frzpar) > 0 else "",
        ))

        syscall("rm *_th1x_*.png", False, True)
        syscall("rm covariance_fit_?.png", False, True)
        syscall("rm higgsCombine_prepost*.root", False, True)
        syscall("rm combine_logger.out", False, True)
        syscall("rm robustHesse_*.root", False, True)

        syscall("mv fitDiagnostics_prepost.root {dcd}{pnt}_fitdiagnostics{gvl}{fix}.root".format(
            dcd = dcdir,
            pnt = "__".join(points),
            gvl = "_" + gstr.replace(".", "p") if gstr != "" else "",
            fix = "_fixed" if args.fixpoi and gstr != "" else "",
        ), False)

    if args.compress:
        syscall(("tar -czf {dcd}.tar.gz {dcd} && rm -r {dcd}").format(
            dcd = dcdir[:-1]
        ))
