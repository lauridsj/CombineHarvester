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

from utilities import syscall
from make_datacard import get_point

def get_fit(dname, points, qexp_eq_m1 = True):
    dfile = TFile.Open(dname)
    dtree = dfile.Get("limit")

    bf = None
    for i in dtree:
        if (dtree.quantileExpected == -1. and qexp_eq_m1) or (dtree.quantileExpected != -1. and not qexp_eq_m1):
            bf = (getattr(dtree, "g_" + points[0]), getattr(dtree, "g_" + points[1]), dtree.deltaNLL)

        if bf is not None:
            break

    dfile.Close()
    return bf

def read_previous_grid(gpoints, gname):
    with open(gname) as ff:
        result = json.load(ff)

    if result["points"] == gpoints:
        return result["g-grid"]

    return OrderedDict()

def get_toys(tname, best_fit, whatever_else = None):
    tfile = TFile.Open(tname)
    ttree = tfile.Get("limit")

    pval = OrderedDict()

    isum = 0
    ipas = 0

    for i in ttree:
        if ttree.quantileExpected != -1.:
            isum += 1
            if ttree.deltaNLL > best_fit[2]:
                ipas += 1

    tfile.Close()

    pval["total"] = isum
    pval["pass"] = ipas

    return pval

def sum_up(g1, g2):
    gs = OrderedDict()
    gs["total"] = g1["total"] + g2["total"]
    gs["pass"] = g1["pass"] + g2["pass"]

    return gs

def stringify(gtuple):
    return str(gtuple)[1: -1]

max_g = 3.
epsilon = 1e-5
nstep = 5

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "desired signal point to run on, comma separated", default = "", required = True)
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
                        "or a projection into 2D templates along 2nd and 3rd dimension: ll;20,3,3;1,2\n"
                        "indices are zero-based, and spaces are ignored. relevant only in datacard/workspace mode.",
                        default = "", required = False)

    parser.add_argument("--unblind", help = "use data when fitting", dest = "asimov", action = "store_false", required = False)
    #parser.add_argument("--no-r", help = "use physics model without r accompanying g", dest = "nor", action = "store_true", required = False)

    parser.add_argument("--fc-g-values", help = "the two values of g to do the FC grid scan for, comma separated",
                        default = "0.0, 0.0", dest = "fcgvl", required = False)
    parser.add_argument("--fc-expect", help = "expected scenarios to assume in the scan, if --unblind isn't used\n"
                        "exp-b -> g1 = g2 = 0; exp-s -> g1 = g2 = 1; exp-01 -> g1 = 0, g2 = 1; exp-10 -> g1 = 1, g2 = 0",
                        default = "exp-b", dest = "fcexp", required = False)
    parser.add_argument("--fc-n-toy", help = "number of toys to throw per FC grid scan",
                        default = 100, dest = "fctoy", required = False, type = int)
    parser.add_argument("--fc-skip-data", help = "skip running on data/asimov", dest = "fcrundat", action = "store_false", required = False)
    parser.add_argument("--fc-idx", help = "index to append to FC grid scan",
                        default = -1, dest = "fcidx", required = False, type = int)

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
    sys.stdout.flush()

    if (args.tag != "" and not args.tag.startswith("_")):
        args.tag = "_" + args.tag

    if args.injectsignal != "":
        args.pseudodata = True

    points = args.point.replace(" ", "").split(',')
    if len(points) != 2:
        raise RuntimeError("this script is to be used with exactly two A/H points!")

    modes = args.mode.replace(" ", "").split(',')
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    args.basedir += "" if args.basedir == "" or args.basedir.endswith("/") else "/"
    dcdir = args.basedir + "__".join(points) + args.tag + "/"

    mstr = str(get_point(points[0])[1]).replace(".0", "")

    allmodes = ["datacard", "workspace", "validate", "fc-scan", "contour", "hadd", "merge", "compile"]
    if (not all([mm in allmodes for mm in modes])):
        print "supported modes:", allmodes
        raise RuntimeError("unxpected mode is given. aborting.")

    # determine what to do with workspace, and do it
    rundc = "datacard" in modes or "workspace" in modes
    runvalid = "validate" in modes
    runfc = "fc-scan" in modes or "contour" in modes
    runhadd = "hadd" in modes or "merge" in modes
    runcompile = "compile" in args.mode

    if rundc:
        print "\ntwin_point_ahtt :: making datacard"
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

        print "\ntwin_point_ahtt :: making workspaces"
        syscall("combineTool.py -M T2W -i {dcd} -o workspace_twin-g.root -m {mmm} -P CombineHarvester.CombineTools.MultiInterferencePlusFixed:multiInterferencePlusFixed "
                "--PO verbose --PO 'signal={pnt}' --PO no-r".format(
                    dcd = dcdir + "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else dcdir + "ahtt_" + args.channel + '_' + args.year + ".txt",
                    mmm = mstr,
                    pnt = args.point.replace(" ", "")
                ))

    if runvalid:
        print "\ntwin_point_ahtt :: validating datacard"
        syscall("ValidateDatacards.py --jsonFile {dcd}{pnt}_validate.json --printLevel 3 {dcd}{crd}".format(
            dcd = dcdir,
            pnt = "__".join(points) + args.tag,
            crd = "ahtt_combined.txt" if os.path.isfile(dcdir + "ahtt_combined.txt") else "ahtt_" + args.channel + '_' + args.year + ".txt"
        ))

    if runfc:
        allexp = ["exp-b", "exp-s", "exp-01", "exp-10"]
        if args.fcexp not in allexp:
            print "supported expected scenario:", allexp
            raise RuntimeError("unexpected expected scenario is given. aborting.")

        exp_scenario = OrderedDict()
        exp_scenario["exp-b"]  = "g_" + points[0] + "=0" + ",g_" + points[1] + "=0"
        exp_scenario["exp-s"]  = "g_" + points[0] + "=1" + ",g_" + points[1] + "=1"
        exp_scenario["exp-01"] = "g_" + points[0] + "=0" + ",g_" + points[1] + "=1"
        exp_scenario["exp-10"] = "g_" + points[0] + "=1" + ",g_" + points[1] + "=0"

        strategy = "--cminPreScan --cminDefaultMinimizerAlgo Migrad --cminDefaultMinimizerStrategy {fstr} --cminFallbackAlgo Minuit2,Simplex,{fstr}"
        #strategy += " --robustFit 1 --setRobustFitStrategy 2 --robustHesse 1" # slow!

        fcgvl = args.fcgvl.replace(" ", "").split(',')
        scan_name = "pnt_g1_" + fcgvl[0] + "_g2_" + fcgvl[1]
        identifier = "_" + args.fcexp if args.asimov else "_data"

        if args.fcrundat:
            print "\ntwin_point_ahtt :: finding the best fit point for FC scan"

            for ifit in ["2", "1", "0"]:
                syscall("combineTool.py -v -1 -M MultiDimFit --algo fixed -d {dcd}workspace_twin-g.root -m {mmm} -n _{snm} "
                        "--fixedPointPOIs '{par}' --setParameters '{exp}' {stg} {asm} {toy} {mcs}".format(
                            dcd = dcdir,
                            mmm = mstr,
                            snm = scan_name + identifier,
                            par = "g_" + points[0] + "=" + fcgvl[0] + ",g_" + points[1] + "=" + fcgvl[1],
                            exp = exp_scenario[args.fcexp],
                            stg = strategy.format(fstr = ifit),
                            asm = "-t -1" if args.asimov else "",
                            toy = "-s -1",
                            mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else ""
                        ))

                if all([get_fit(glob.glob("higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root".format(snm = scan_name + identifier, mmm = mstr))[0], points, ff) is not None for ff in [True, False]]):
                    break
                else:
                    syscall("rm higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root".format(snm = scan_name + identifier, mmm = mstr), False)

            syscall("mv higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root {dcd}fc_scan_{snm}.root".format(snm = scan_name + identifier, mmm = mstr, dcd = dcdir), False)

        if args.fctoy > 0:
            identifier = "_toys_" + str(args.fcidx) if args.fcidx > -1 else "_toys"
            print "\ntwin_point_ahtt :: performing the FC scan for toys"
            syscall("combineTool.py -v -1 -M MultiDimFit --algo fixed -d {dcd}workspace_twin-g.root -m {mmm} -n _{snm} "
                    "--fixedPointPOIs '{par}' --setParameters '{par}' {stg} {toy} {mcs}".format(
                        dcd = dcdir,
                        mmm = mstr,
                        snm = scan_name + identifier,
                        par = "g_" + points[0] + "=" + fcgvl[0] + ",g_" + points[1] + "=" + fcgvl[1],
                        stg = strategy,
                        toy = "-s -1 --toysFrequentist -t " + str(args.fctoy),
                        mcs = "--X-rtd MINIMIZER_analytic" if args.mcstat else ""
                    ))

            syscall("mv higgsCombine_{snm}.MultiDimFit.mH{mmm}*.root {dcd}fc_scan_{snm}.root".format(
                dcd = dcdir,
                snm = scan_name + identifier,
                mmm = mstr,
            ), False)

    if runhadd:
        toys = glob.glob("{dcd}fc_scan_*_toys.root".format(dcd = dcdir))
        idxs = glob.glob("{dcd}fc_scan_*_toys_*.root".format(dcd = dcdir))

        if len(idxs) > 0:
            print "\ntwin_point_ahtt :: indexed toy files detected, merging them..."

            if len(toys) == 0:
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

        best = glob.glob("{dcd}fc_scan_*{exp}.root".format(dcd = dcdir, exp = "_" + args.fcexp if args.asimov else "_data"))
        if len(best) == 0:
            raise RuntimeError("result compilation can't proceed without the best fit files being available!!")

        print "\ntwin_point_ahtt :: compiling FC scan results..."
        ggrid = glob.glob("{dcd}fc_scan{exp}_*.json".format(dcd = dcdir, exp = "_" + args.fcexp if args.asimov else "_data"))
        ggrid.sort()
        idx = 0 if len(ggrid) == 0 else int(ggrid[-1].split("_")[-1].split(".")[0]) + 1

        grid = OrderedDict()
        grid["points"] = points
        grid["g-grid"] = OrderedDict() if idx == 0 else read_previous_grid(grid["points"], ggrid[-1])

        for bb in best:
            bf = get_fit(bb, points)
            gg = get_toys(bb.replace("{exp}.root".format(exp = "_" + args.fcexp if args.asimov else "_data"), "_toys.root"), bf)
            gv = get_fit(bb, points, False)
            gv = (gv[0], gv[1])

            if gv in grid["g-grid"]:
                grid["g-grid"][stringify(gv)] = sum_up(grid["g-grid"][stringify(gv)], gg)
            else:
                grid["g-grid"][stringify(gv)] = gg

        with open("{dcd}fc_scan{exp}_{idx}.json".format(dcd = dcdir, exp = "_" + args.fcexp if args.asimov else "_data", idx = str(idx)), "w") as jj:
            json.dump(grid, jj, indent = 1)

    if args.compress:
        syscall(("tar -czf {dcd}.tar.gz {dcd} && rm -r {dcd}").format(
            dcd = dcdir[:-1]
        ))
