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

from utilities import syscall, submit_job, aggregate_submit, input_bkg, input_sig, min_g, max_g, tuplize

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
                contour = json.load(ff, object_pairs_hook = OrderedDict)

            if contour["points"] != pair:
                print "given pair in grid is inconsistent with currently expected pair, skipping..."
                print "in grid: ", contour["points"]
                print "currently expected: ", pair
                return g_grid

            if gmode == "add":
                for gv in contour["g-grid"].keys():
                    if contour["g-grid"][gv] is not None:
                        gt = tuplize(gv)
                        if gt not in g_grid:
                            g_grid.append(gt)

            if gmode == "refine":
                mintoy = sys.maxsize
                for gv in contour["g-grid"].keys():
                    mintoy = min(mintoy, contour["g-grid"][gv]["total"] if contour["g-grid"][gv] is not None else sys.maxsize)

                cuts = [mintoy > (4.5 / alpha) for alpha in generate_g_grid.alphas]
                if sum([1 if cut else 0 for cut in cuts]) < 2:
                    print "minimum toy count: " + str(mintoy)
                    raise RuntimeError("minimum toys insufficient to determine 2 sigma contour!! likely unintended, recheck!!")

                gts = [tuplize(gv) for gv in contour["g-grid"].keys() if contour["g-grid"][gv] is not None]
                effs = [float(contour["g-grid"][gv]["pass"]) / float(contour["g-grid"][gv]["total"]) for gv in contour["g-grid"].keys() if contour["g-grid"][gv] is not None]

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
    parser.add_argument("--point",
                        help = "desired pairs of signal points to run on, comma (between points) and semicolon (between pairs) separated\n"
                        "another syntax is: m1,m2,...,mN;w1,w2,...,wN;m1,m2,...,mN;w1,w2,...,wN, where:\n"
                        "the first mass and width strings refer to the A grid, and the second to the H grid.\n"
                        "both mass and width strings must include their m and w prefix, and for width, their p0 suffix.\n"
                        "e.g. m400,m450;w5p0;m600,m750;w10p0 expands to A_m400_w5p0,H_m600_w10p0;A_m450_w5p0,H_m600_w10p0;A_m400_w5p0,H_m750_w10p0;A_m450_w5p0,H_m750_w10p0",
                        default = "", required = True)
    parser.add_argument("--mode", help = "combine mode to run, comma separated", default = "datacard,validate", required = False)

    parser.add_argument("--signal", help = "signal filenames. comma separated", default = "", required = False)
    parser.add_argument("--background", help = "data/background filenames. comma separated", default = "", required = False)

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

    parser.add_argument("--seed",
                        help = "random seed to be used for pseudodata generation. give 0 to read from machine, and negative values to use no rng",
                        default = "", required = False)

    parser.add_argument("--unblind", help = "use data when fitting", dest = "asimov", action = "store_false", required = False)
    #parser.add_argument("--no-r", help = "use physics model without r accompanying g", dest = "nor", action = "store_true", required = False)

    parser.add_argument("--g-values", help = "the two values of g to e.g. do fit diagnostics for, comma separated",
                        default = "-1., -1.", dest = "gvl", required = False)
    parser.add_argument("--fix-poi", help = "fix pois in the fit, through --g-values",
                        dest = "fixpoi", action = "store_true", required = False)

    parser.add_argument("--n-toy", help = "number of toys to throw per point when generating or performing FC scans",
                        default = 50, dest = "ntoy", required = False, type = int)
    parser.add_argument("--run-idxs", help = "can be one or more comma separated non-negative integers, or something of the form A...B where A < B and A, B non-negative\n"
                        "where the comma separated version is plainly the list of indices to be given to --run-idx, if --n-toy > 0\n"
                        "and the A...B version builds a list of indices from [A, B). If A is omitted, it is assumed to be 0\n"
                        "mixing of both syntaxes are not allowed.",
                        default = "-1", dest = "runidxs", required = False)
    parser.add_argument("--toy-location", help = "directory to dump the toys in mode generate, "
                        "and file to read them from in mode contour (only for g1 = g2 = 0) or gof.\n"
                        "comma separated when reading multiple files, and must be as long as --run-idxs.",
                        dest = "toyloc", default = "", required = False)

    parser.add_argument("--fc-g-grid", help = "comma (between files) and semicolon (between pairs) separated json files generated by compile mode to read points from",
                        default = "", dest = "fcgrid", required = False)
    parser.add_argument("--fc-mode",
                        help = "what to do with the grid read from --fc-g-grid, can be 'add' for submitting more toys of the same points,\n"
                        "or 'refine', for refining the contour that can be drawn using the grid",
                        default = "", dest = "fcmode", required = False)
    parser.add_argument("--fc-single-point", help = "run FC scan only on a single point given by --g-values",
                        dest = "fcsinglepnt", action = "store_true", required = False)

    parser.add_argument("--fc-expect", help = "expected scenarios to assume in the FC scan. comma separated.\n"
                        "exp-b -> g1 = g2 = 0; exp-s -> g1 = g2 = 1; exp-01 -> g1 = 0, g2 = 1; exp-10 -> g1 = 1, g2 = 0",
                        default = "exp-b", dest = "fcexp", required = False)
    parser.add_argument("--fc-nuisance-mode", help = "how to handle nuisance parameters in toy generation (see https://arxiv.org/abs/2207.14353)\n"
                        "WARNING: profile mode implementation is incomplete!!",
                        default = "conservative", dest = "fcnui", required = False)
    parser.add_argument("--fc-skip-data", help = "skip running on data/asimov", dest = "fcrundat", action = "store_false", required = False)

    parser.add_argument("--fc-initial-distance", help = "initial distance between g grid points for FC scans",
                        default = 0.5, dest = "fcinit", required = False, type = float)

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

    parser.add_argument("--proper-sigma", help = "use proper 1 or 2 sigma CLs instead of 68% and 95% in FC scan alphas",
                        dest = "propersig", action = "store_true", required = False)

    parser.add_argument("--job-time", help = "time to assign to each job", default = "", dest = "jobtime", required = False)
    parser.add_argument("--local", help = "run jobs locally, do not submit to HTC", dest = "runlocal", action = "store_true", required = False)
    parser.add_argument("--force", help = "force local jobs to run, even if a job log already exists", dest = "forcelocal", action = "store_true", required = False)

    args = parser.parse_args()
    if (args.tag != "" and not args.tag.startswith("_")):
        args.tag = "_" + args.tag
    scriptdir = os.path.dirname(os.path.abspath(__file__))

    pairs = args.point.replace(" ", "").split(';')

    # handle the case of gridding pairs
    if len(pairs) == 4:
        pairgrid = [pp.split(",") for pp in pairs]
        if all([mm.startswith("m") for mm in pairgrid[0] + pairgrid[2]]) and all([ww.startswith("w") for ww in pairgrid[1] + pairgrid[3]]):
            alla = []
            for mm in pairgrid[0]:
                for ww in pairgrid[1]:
                    alla.append("_".join(["A", mm, ww]))

            allh = []
            for mm in pairgrid[2]:
                for ww in pairgrid[3]:
                    allh.append("_".join(["H", mm, ww]))

            pairs = []
            for aa in alla:
                for hh in allh:
                    pairs.append(aa + "," + hh)

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
    rungen = "generate" in modes
    runfc = "fc-scan" in args.mode or "contour" in args.mode
    runhadd = "hadd" in args.mode or "merge" in args.mode
    runcompile = "compile" in args.mode
    runprepost = "prepost" in args.mode or "corrmat" in args.mode
    runclean = "clean" in args.mode

    if runcompile and (rundc or runfc or runhadd):
        raise RuntimeError("compile mode must be ran on its own!")

    for pair, ggrid in zip(pairs, ggrids):
        # generate an aggregate submission file name
        agg = aggregate_submit()

        points = pair.split(',')
        pstr = '__'.join(points)

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
                   "{shp} {mcs} {msk} {prj} {frz} {asm} {rsd} {com} {rmr} {igp} {gvl} {fix} {exp} {bsd}").format(
            pnt = pair,
            mmm = args.mode if not "clean" in args.mode else ','.join([mm for mm in args.mode.replace(" ", "").split(",") if "clean" not in mm]),
            sus = "--sushi-kfactor" if args.kfactor else "",
            psd = "--use-pseudodata" if args.pseudodata else "",
            inj = "--inject-signal " + args.injectsignal if args.injectsignal != "" else "",
            tag = "--tag " + args.tag if args.tag != "" else "",
            drp = "--drop '" + args.drop + "'" if args.drop != "" else "",
            kee = "--keep '" + args.keep + "'" if args.keep != "" else "",
            sig = "--signal " + input_sig(args.signal, pair, args.injectsignal, args.channel, args.year),
            bkg = "--background " + input_bkg(args.background, args.channel),
            cha = "--channel " + args.channel,
            yyy = "--year " + args.year,
            thr = "--threshold " + str(args.threshold),
            lns = "--lnN-under-threshold" if args.lnNsmall else "",
            shp = "--use-shape-always" if args.alwaysshape else "",
            mcs = "--no-mc-stats" if not args.mcstat else "",
            msk = "--mask '" + args.mask + "'" if args.mask != "" else "",
            prj = "--projection '" + args.projection + "'" if rundc and args.projection != "" else "",
            frz = "--freeze-mc-stats-zero" if args.frzbb0 else "--freeze-mc-stats-post" if args.frzbbp else "--freeze-nuisance-post" if args.frznui else "",
            asm = "--unblind" if not args.asimov else "",
            rsd = "--seed " + args.seed if args.seed != "" else "",
            com = "--compress" if rundc else "",
            rmr = "--delete-root" if args.rmroot else "",
            igp = "--ignore-previous" if args.ignoreprev else "",
            gvl = "--g-values '" + args.gvl + "'" if not runfc and any(float(gg) >= 0. for gg in args.gvl.replace(" ", "").split(',')) else "",
            fix = "--fix-poi" if args.fixpoi and any(float(gg) >= 0. for gg in args.gvl.replace(" ", "").split(',')) else "",
            exp = "--fc-expect " + args.fcexp if runfc or runcompile else "",
            bsd = "" if rundc else "--base-directory " + os.path.abspath("./")
        )

        if rungen or runfc:
            idxs = []
            if args.ntoy > 0:
                if "," in args.runidxs and "..." in args.runidxs:
                    raise RuntimeError("it is said that mixing syntaxes is not allowed smh.")
                elif "," in args.runidxs:
                    idxs = [int(ii) for ii in args.runidxs.replace(" ", "").split(",")]
                elif "..." in args.runidxs:
                    idxs = args.runidxs.replace(" ", "").split("...")
                    idxs = range(int(idxs[0]), int(idxs[1])) if idxs[0] != "" else range(int(idxs[1]))
                else:
                    idxs = [int(args.runidxs.replace(" ", ""))]
            else:
                idxs = [-1]

            if rungen:
                for ii, idx in enumerate(idxs):
                    jname = job_name
                    jname += '_' + str(idx) if idx != -1 else ''
                    logs = glob.glob(pstr + args.tag + "/" + jname + ".o*")

                    if not (args.runlocal and args.forcelocal):
                        if len(logs) > 0:
                            continue

                    jarg = job_arg
                    jarg += " {toy} {idx} {opd}".format(
                        toy = "--n-toy " + str(args.ntoy) if args.ntoy > 0 else "",
                        idx = "--run-idx " + str(idx) if idx > -1 else "",
                        opd = "--toy-location '" + os.path.abspath(args.toyloc) + "'" if args.toyloc != "" else ""
                    )

                    submit_job(agg, jname, jarg, args.jobtime, 1, "",
                               "." if rundc else "$(readlink -f " + pstr + args.tag + ")", scriptdir + "/twin_point_ahtt.py", True, args.runlocal)

            if runfc:
                if args.fcmode != "" and ggrid == "":
                    print "checking last grids"
                    fcexps = args.fcexp.replace(" ", "").split(',')
                    if not args.asimov:
                        fcexps.append("obs")

                    for fcexp in fcexps:
                        ggg = glob.glob(pstr + args.tag + "/" + pstr + "_fc_scan_" + fcexp + "_*.json")
                        ggg.sort(key = os.path.getmtime)
                        ggrid += ggg[-1] if ggrid == "" else "," + ggg[-1]
                print "using the following grids:"
                print ggrid

                if args.fcsinglepnt:
                    gvalues = [gg in args.gvl.replace(" ", "").split(',')]
                    gvalues = [tuple(gvalues)]
                else:
                    gvalues = generate_g_grid(points, ggrid, args.fcmode, args.propersig, int(math.ceil((max_g - min_g) / args.fcinit)) + 1 if min_g < args.fcinit < max_g else 7)

                toylocs = []
                if args.toyloc != "":
                    toylocs = glob.glob("{opd}**/*_toys_*_n*.root".format(opd = args.toyloc))
                    shuffle(ftoys)
                if len(toylocs) != 0 and len(toylocs) < len(idxs):
                        raise RuntimeError("expecting at least as many toy files as there are run indices!!")

                for ig1, ig2 in gvalues:
                    scan_name = "g1_" + str(ig1) + "_g2_" + str(ig2)

                    for idx in idxs:
                        jname = job_name + scan_name
                        jname += '_' + str(idx) if idx != -1 else ''
                        logs = glob.glob(pstr + args.tag + "/" + jname + ".o*")

                        if not (args.runlocal and args.forcelocal):
                            if len(logs) > 0:
                                continue

                        fcrundat = args.fcmode != "add" and args.fcrundat and idx == idxs[0]

                        jarg = job_arg
                        jarg += " {gvl} {nui} {toy} {dat} {idx}".format(
                            gvl = "--g-values '" + str(ig1) + "," + str(ig2) + "'",
                            nui = "--fc-nuisance-mode " + args.fcnui,
                            toy = "--n-toy " + str(args.ntoy) if args.ntoy > 0 else "",
                            dat = "--fc-skip-data " if not fcrundat else "",
                            idx = "--run-idx " + str(idx) if idx > -1 else ""
                        )

                        if ig1 == 0. and ig2 == 0. and len(toylocs) > 0:
                            jarg += " --toy-location " + toylocs[ii]

                        submit_job(agg, jname, jarg, args.jobtime, 1, "",
                                   "." if rundc else "$(readlink -f " + pstr + args.tag + ")", scriptdir + "/twin_point_ahtt.py", True, args.runlocal)
        else:
            logs = glob.glob(pstr + args.tag + "/" + job_name + ".o*")

            if not (args.runlocal and args.forcelocal):
                if len(logs) > 0:
                    continue

            if runclean:
                syscall("find {dcd} -type f -name 'twin_point_{dcd}_contour_g1_*_g2_*.o*.*' | xargs rm".format(dcd = pstr + args.tag), True, True)
                syscall("find {dcd} -type f -name 'twin_point_{dcd}_fc-scan_g1_*_g2_*.o*.*' | xargs rm".format(dcd = pstr + args.tag), True, True)
                syscall("find {dcd} -type f -name 'twin_point_{dcd}_merge.o*.*' | xargs rm".format(dcd = pstr + args.tag), True, True)
                syscall("find {dcd} -type f -name 'twin_point_{dcd}_hadd.o*.*' | xargs rm".format(dcd = pstr + args.tag), True, True)
                syscall("find {dcd} -type f -name 'twin_point_{dcd}_compile.o*.*' | xargs rm".format(dcd = pstr + args.tag), True, True)

            #job_mem = "12 GB" if runprepost and not (args.frzbb0 or args.frzbbp or args.frznui) else ""
            job_mem = ""
            submit_job(agg, job_name, job_arg, args.jobtime, 1, job_mem,
                       "." if rundc else "$(readlink -f " + pstr + args.tag + ")", scriptdir + "/twin_point_ahtt.py",
                       True, runcompile or args.runlocal)

        if os.path.isfile(agg):
            syscall('condor_submit {agg}'.format(agg = agg), False)
            syscall('rm {agg}'.format(agg = agg), False)
