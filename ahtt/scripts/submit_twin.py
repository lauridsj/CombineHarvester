#!/usr/bin/env python
# submit the two-point jobs and compiles its results

from argparse import ArgumentParser
import os
import sys
import glob
import subprocess
import numpy as np
import copy

from collections import OrderedDict
import json
import math

from utilities import syscall, submit_job, aggregate_submit

min_g = 0.
max_g = 3.

def tuplize(gstring):
    return tuple([float(gg) for gg in gstring.replace(" ", "").split(",")])

sqd = lambda p1, p2: sum([(pp1 - pp2)**2. for pp1, pp2 in zip(p1, p2)], 0.)

halfway = lambda p1, p2: tuple([(pp1 + pp2) / 2. for pp1, pp2 in zip(p1, p2)])

def generate_g_grid(pair, ggrids = "", gmode = "", propersig = False, ndivision = 7):
    if not hasattr(generate_g_grid, "alphas"):
        generate_g_grid.alphas = [0.6827, 0.9545, 0.9973, 0.999937, 0.9999997] if propersig else [0.68, 0.95, 0.9973, 0.999937, 0.9999997]
        generate_g_grid.alphas = [1. - pval for pval in generate_g_grid.alphas]

    g_grid = []

    if ggrids != "":
        ggrids = ggrids.replace(" ", "").split(",")

        for ggrid in ggrids:
            if not os.path.isfile(ggrid):
                print "g grid file " + ggrid + " is not accessible, skipping..."
                return g_grid

            with open(ggrid) as ff:
                cc = json.load(ff, object_pairs_hook = OrderedDict)

            if cc["points"] != pair:
                print "given pair in grid is inconsistent with currently expected pair, skipping..."
                print "in grid: ", cc["points"]
                print "currently expected: ", pair
                return g_grid

            if gmode == "add":
                for gv in cc["g-grid"].keys():
                    gt = tuplize(gv)
                    if gt not in g_grid:
                        g_grid.append(gt)

            if gmode == "refine":
                mintoy = sys.maxsize
                for gv in cc["g-grid"].keys():
                    mintoy = min(mintoy, cc["g-grid"][gv]["total"])

                cuts = [mintoy > (4.5 / alpha) for alpha in generate_g_grid.alphas]
                gts = [tuplize(gv) for gv in cc["g-grid"].keys()]
                effs = [float(cc["g-grid"][gv]["pass"]) / float(cc["g-grid"][gv]["total"]) for gv in cc["g-grid"].keys()]

                for gt, eff in zip(gts, effs):
                    unary_sqd = lambda pp: sqd(pp[0], gt)

                    gx = [(gg, ee) for gg, ee in zip(gts, effs) if gg[1] == gt[1] and gg[0] > gt[0]]
                    gy = [(gg, ee) for gg, ee in zip(gts, effs) if gg[0] == gt[0] and gg[1] > gt[1]]
                    gxy = [(gg, ee) for gg, ee in zip(gts, effs) if gg[1] > gt[1] and gg[0] > gt[0]]

                    gx = min(gx, key = unary_sqd) if len(gx) > 0 else None
                    gy = min(gy, key = unary_sqd) if len(gy) > 0 else None
                    gxy = min(gxy, key = unary_sqd) if len(gxy) > 0 else None

                    for cut, alpha in zip(cuts, generate_g_grid.alphas):
                        if cut:
                            differences = [gg is not None and ((gg[1] > alpha and eff < alpha) or (gg[1] < alpha and eff > alpha)) for gg in [gx, gy, gxy]]
                            if any(differences):
                                halfsies = []
                                for g1 in [gx, gy, gxy]:
                                    if g1 is None:
                                        continue

                                    for g2 in [gx, gy, gxy]:
                                        if g2 is None or g2 == g1:
                                            continue

                                        halfsies.append((g1[0], gt))
                                        halfsies.append((g1[0], g2[0]))
                                halfsies = [halfway(p1, p2) for p1, p2 in halfsies]

                                for half in halfsies:
                                    if half not in g_grid:
                                        g_grid.append(half)

        return g_grid

    # default LO case
    gvls = [list(np.linspace(min_g, max_g, num = ndivision)), list(np.linspace(min_g, max_g, num = ndivision))]
    for ig1 in gvls[0]:
        for ig2 in gvls[1]:
            g_grid.append( (ig1,ig2) )

    return g_grid

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "desired pairs of signal points to run on, comma (between points) and semicolon (between pairs) separated", default = "", required = True)
    parser.add_argument("--mode", help = "combine mode to run, comma separated", default = "datacard,validate", required = False)

    parser.add_argument("--signal", help = "signal filenames. comma separated", default = "", required = False)
    parser.add_argument("--background", help = "data/background filenames. comma separated", default = "", required = False)

    parser.add_argument("--channel", help = "final state channels considered in the analysis. comma separated", default = "ee,em,mm,e3j,e4pj,m3j,m4pj", required = False)
    parser.add_argument("--year", help = "analysis year determining the correlation model to assume. comma separated", default = "2016pre,2016post,2017,2018", required = False)

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

    parser.add_argument("--fc-g-grid", help = "comma (between files) and semicolon (between pairs) separated json files generated by compile mode to read points from",
                        default = "", dest = "fcgrid", required = False)
    parser.add_argument("--fc-mode",
                        help = "what to do with the grid read from --fc-g-grid, can be 'add' for submitting more toys of the same points,\n"
                        "or 'refine', for refining the contour that can be drawn using the grid",
                        default = "", dest = "fcmode", required = False)
    parser.add_argument("--fc-idxs", help = "can be one or more comma separated non-negative integers, or something of the form A...B where A < B and A, B non-negative\n"
                        "where the comma separated version is plainly the list of indices to be given to --fc-idx, if --fc-n-toy > 0\n"
                        "and the A...B version builds a list of indices from [A, B). If A is omitted, it is assumed to be 0\n"
                        "mixing of both syntaxes are not allowed.",
                        default = "-1", dest = "fcidxs", required = False)

    parser.add_argument("--fc-expect", help = "expected scenarios to assume in the scan. comma separated.\n"
                        "exp-b -> g1 = g2 = 0; exp-s -> g1 = g2 = 1; exp-01 -> g1 = 0, g2 = 1; exp-10 -> g1 = 1, g2 = 0",
                        default = "exp-b", dest = "fcexp", required = False)
    parser.add_argument("--fc-nuisance-mode", help = "how to handle nuisance parameters in toy generation (see https://arxiv.org/abs/2207.14353)",
                        default = "profile", dest = "fcnui", required = False)
    parser.add_argument("--fc-n-toy", help = "number of toys to throw per FC grid scan",
                        default = 175, dest = "fctoy", required = False, type = int)
    parser.add_argument("--fc-skip-data", help = "skip running on data/asimov", dest = "fcrundat", action = "store_false", required = False)

    parser.add_argument("--fc-initial-distance", help = "initial distance between g grid points for FC scans",
                        default = 0.5, dest = "fcinit", required = False, type = float)

    parser.add_argument("--delete-root", help = "delete root files after compiling", dest = "rmroot", action = "store_true", required = False)
    parser.add_argument("--ignore-previous", help = "ignore previous grid when compiling", dest = "ignoreprev", action = "store_true", required = False)

    parser.add_argument("--proper-sigma", help = "use proper 1 or 2 sigma CLs instead of 68% and 95% in alphas",
                        dest = "propersig", action = "store_true", required = False)

    parser.add_argument("--job-time", help = "time to assign to each job", default = "", dest = "jobtime", required = False)
    args = parser.parse_args()
    if (args.tag != "" and not args.tag.startswith("_")):
        args.tag = "_" + args.tag
    scriptdir = os.path.dirname(os.path.abspath(__file__))

    # FIXME something to allow --point to be non-default
    pairs = args.point.replace(" ", "").split(';')

    ggrids = None
    if args.fcgrid != "" and args.fcmode != "":
        ggrids = args.fcgrid.replace(" ", "").split(';')
    else:
        ggrids = ["" for pair in pairs]

    if ggrids is not None:
        if len(ggrids) != len(pairs):
            raise RuntimeError("there needs to be as many json file groups as there are pairs")
    else:
        ggrids = ["" for pair in pairs]

    if args.injectsignal != "":
        args.pseudodata = True

    if args.jobtime != "":
        args.jobtime = "-t " + args.jobtime

    rundc = "datacard" in args.mode or "workspace" in args.mode
    runfc = "fc-scan" in args.mode or "contour" in args.mode
    runhadd = "hadd" in args.mode or "merge" in args.mode
    runcompile = "compile" in args.mode

    if runcompile and (rundc or runfc or runhadd):
        raise RuntimeError("compile mode must be ran on its own!")

    # backgrounds if not given
    backgrounds = []
    if args.background == "":
        if any(cc in args.channel for cc in ["ee", "em", "mm"]):
            backgrounds.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/bkg_templates_3D-33.root")
        if any(cc in args.channel for cc in ["e3j", "e4pj", "m3j", "m4pj"]):
            backgrounds.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/templates_lj_bkg_rename.root")
        background = ','.join(backgrounds)
    else:
        background = args.background

    siglj = []
    if args.signal == "":
        if any(cc in args.channel for cc in ["e3j", "e4pj", "m3j", "m4pj"]):
            if "2016pre" in args.year:
                siglj.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/templates_lj_sig_2016pre.root")
            if "2016post" in args.year:
                siglj.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/templates_lj_sig_2016post.root")
            if "2017" in args.year:
                siglj.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/templates_lj_sig_2017.root")
            if "2018" in args.year:
                siglj.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/templates_lj_sig_2018.root")

    for pair, ggrid in zip(pairs, ggrids):
        # generate an aggregate submission file name
        agg = aggregate_submit()

        points = pair.split(',')
        pstr = '__'.join(points)

        if args.signal == "":
            signals = copy.deepcopy(siglj)
            for pnt in points:
                if "_m3" in pnt or "_m1000" in pnt or "_m3" in args.injectsignal or "_m1000" in args.injectsignal:
                    if any(cc in args.channel for cc in ["ee", "em", "mm"]):
                        signals.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/nonsense_timestamp/sig_ll_3D-33_m3xx_and_m1000.root")
                for im in ["_m4", "_m5", "_m6", "_m7", "_m8", "_m9"]:
                    if im in pnt or im in args.injectsignal:
                        if any(cc in args.channel for cc in ["ee", "em", "mm"]):
                            signals.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/nonsense_timestamp/sig_ll_3D-33" + im + "xx.root")
            signal = ','.join(set(signals))
        else:
            signal = args.signal

        if not rundc and not os.path.isdir(pstr + args.tag) and os.path.isfile(pstr + args.tag + ".tar.gz"):
            syscall("tar xf {ttt} && rm {ttt}".format(ttt = pstr + args.tag + ".tar.gz"))

        hasworkspace = os.path.isfile(pstr + args.tag + "/workspace_twin-g.root")
        if not rundc and not hasworkspace:
            syscall("rm -r {ddd}".format(ddd = pstr + args.tag))
            rundc = True
            args.mode = "datacard," + args.mode

        if os.path.isdir(pstr + args.tag):
            logs = glob.glob("twin_point_" + pstr + args.tag + "_*.o*")
            for ll in logs:
                syscall("mv {lll} {ddd}".format(lll = ll, ddd = pstr + args.tag))

        if rundc and os.path.isdir(pstr + args.tag):
            args.mode = args.mode.replace("datacard,", "").replace("datacard", "").replace("workspace,", "").replace("workspace", "")

            if args.mode != "":
                rundc = False
            else:
                continue

        job_name = "twin_point_" + pstr + args.tag + "_" + "_".join(args.mode.replace(" ", "").split(","))
        job_arg = ("--point {pnt} --mode {mmm} {sus} {psd} {inj} {tag} {drp} {kee} {sig} {bkg} {cha} {yyy} {thr} {lns} "
                   "{shp} {mcs} {prj} {asm} {com} {rmr} {igp} {exp} {bsd}").format(
            pnt = pair,
            mmm = args.mode,
            sus = "--sushi-kfactor" if args.kfactor else "",
            psd = "--use-pseudodata" if args.pseudodata else "",
            inj = "--inject-signal " + args.injectsignal if args.injectsignal != "" else "",
            tag = "--tag " + args.tag if args.tag != "" else "",
            drp = "--drop '" + args.drop + "'" if args.drop != "" else "",
            kee = "--keep '" + args.keep + "'" if args.keep != "" else "",
            sig = "--signal " + signal,
            bkg = "--background " + background,
            cha = "--channel " + args.channel,
            yyy = "--year " + args.year,
            thr = "--threshold " + str(args.threshold),
            lns = "--lnN-under-threshold" if args.lnNsmall else "",
            shp = "--use-shape-always" if args.alwaysshape else "",
            mcs = "--no-mc-stats" if not args.mcstat else "",
            prj = "--projection '" + args.projection + "'" if rundc and args.projection != "" else "",
            asm = "--unblind" if not args.asimov else "",
            com = "--compress" if rundc else "",
            rmr = "--delete-root" if args.rmroot else "",
            igp = "--ignore-previous" if args.ignoreprev else "",
            exp = "--fc-expect " + args.fcexp if runfc or runcompile else "",
            bsd = "" if rundc else "--base-directory " + os.path.abspath("./")
        )

        if runfc:
            if args.fcmode != "" and ggrid == "":
                print "checking last grids"
                fcexps = args.fcexp.replace(" ", "").split(',')
                for fcexp in fcexps:
                    ggg = glob.glob(pstr + args.tag + "/fc_scan_" + fcexp + "_*.json")
                    ggg.sort(key = os.path.getmtime)
                    ggrid += ggg[-1] if ggrid == "" else "," + ggg[-1]
                print "using the following as last grids:"
                print ggrid

            gvalues = generate_g_grid(points, ggrid, args.fcmode, args.propersig, int(math.ceil((max_g - min_g) / args.fcinit)) + 1 if min_g < args.fcinit < max_g else 7)
            idxs = []
            if args.fctoy > 0:
                if "," in args.fcidxs and "..." in args.fcidxs:
                    raise RuntimeError("it is said that mixing syntaxes is not allowed smh.")
                elif "," in args.fcidxs:
                    idxs = [int(ii) for ii in args.fcidxs.replace(" ", "").split(",")]
                elif "..." in args.fcidxs:
                    idxs = args.fcidxs.replace(" ", "").split("...")
                    idxs = range(int(idxs[0]), int(idxs[1])) if idxs[0] != "" else range(int(idxs[1]))
                else:
                    idxs = [int(args.fcidxs.replace(" ", ""))]
            else:
                idxs = [-1]

            for ig1, ig2 in gvalues:
                scan_name = "pnt_g1_" + str(ig1) + "_g2_" + str(ig2)

                for idx in idxs:
                    jname = job_name + scan_name.replace("pnt", "")
                    jname += '_' + str(idx) if idx != -1 else ''
                    logs = glob.glob(pstr + args.tag + "/" + jname + ".o*")

                    if len(logs) > 0:
                        continue

                    fcrundat = args.fcrundat and idx == idxs[0]

                    jarg = job_arg
                    jarg += " {gvl} {nui} {toy} {dat} {idx}".format(
                        gvl = "--fc-g-values '" + str(ig1) + "," + str(ig2) + "'",
                        nui = "--fc-nuisance-mode " + args.fcnui,
                        toy = "--fc-n-toy " + str(args.fctoy) if args.fctoy > 0 else "",
                        dat = "--fc-skip-data " if not fcrundat else "",
                        idx = "--fc-idx " + str(idx) if idx > -1 else ""
                    )

                    submit_job(agg, jname, jarg, args.jobtime, 1, "",
                               "" if rundc else "-l $(readlink -f " + pstr + args.tag + ")", scriptdir + "/twin_point_ahtt.py", True)
        else:
            logs = glob.glob(pstr + args.tag + "/" + job_name + ".o*")

            if len(logs) > 0:
                continue

            submit_job(agg, job_name, job_arg, args.jobtime, 1, "",
                       "" if rundc else "-l $(readlink -f " + pstr + args.tag + ")", scriptdir + "/twin_point_ahtt.py", True, runhadd or runcompile)

        if os.path.isfile(agg):
            syscall('condor_submit {agg}'.format(agg = agg), False)
            syscall('rm {agg}'.format(agg = agg), False)
