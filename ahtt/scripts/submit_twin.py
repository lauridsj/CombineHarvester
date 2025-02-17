#!/usr/bin/env python
# submit the two-point jobs and compiles its results

from argparse import ArgumentParser
import os
import sys
import glob
import numpy as np
import copy
import re

from random import shuffle
from collections import OrderedDict
import json
import math
from datetime import datetime

from utilspy import syscall, tuplize, g_in_filename, recursive_glob, index_list, floattopm, uniform, coinflip
from utilspy import make_timestamp_dir, directory_to_delete, max_nfile_per_dir
from utilslab import input_base, input_sig, remove_mjf, masses, widths
from utilscombine import problematic_datacard_log, min_g, max_g, expected_scenario
from utilshtc import submit_job, flush_jobs, common_job

from desalinator import prepend_if_not_empty, tokenize_to_list, remove_quotes, remove_spaces_quotes, clamp_with_quote
from argumentative import common_common, common_fit_pure, common_fit_forwarded, make_datacard_pure, make_datacard_forwarded, common_2D
from argumentative import common_submit, parse_args
from hilfemir import combine_help_messages, submit_help_messages

sqd = lambda p1, p2: sum([(pp2 - pp1)**2. for pp1, pp2 in zip(p1, p2)], 0.)
halfway = lambda p1, p2: tuple([(pp1 + pp2) / 2. for pp1, pp2 in zip(p1, p2)])
angle = lambda p1, p2: math.atan2(p2[1] - p1[1], p2[0] - p1[0])
q1sqd = lambda p1, p2: sqd(p1, p2) if 0. <= angle(p1, p2) / math.pi <= 0.5 else sys.float_info.max
gstr_precision = 3

def make_initial_grid(grange, spacing):
    grid = []
    gvls = [list(np.linspace(grange[0], grange[1], num = math.ceil((grange[1] - grange[0]) / spacing[0]) + 1)),
            list(np.linspace(grange[2], grange[3], num = math.ceil((grange[3] - grange[2]) / spacing[1]) + 1))]
    for ig1 in gvls[0]:
        for ig2 in gvls[1]:
            grid.append( (ig1, ig2, 0) )
    return grid

def default_nminmax(arg = ""):
    result = list(map(float, tokenize_to_list( remove_spaces_quotes(arg))))
    defaults = [32, 2.**-9, 2.**-2]
    while len(result) < len(defaults):
        result.append(defaults[len(result)])
    return result

def default_initial_distance(arg = ""):
    result = [] if arg == "" else tokenize_to_list(remove_spaces_quotes(arg), ",", astype = float)
    if len(result) >= 2:
        result = result[:2]
    elif len(result) == 1:
        result = [result[0], result[0]]
    if not (len(result) == 2 and all([0. < rr <= 1. for rr in result])):
        result = (1., 1.)
    return tuple(result)

def default_g_range(arg = ""):
    result = [] if arg == "" else tokenize_to_list(remove_spaces_quotes(arg), ",", astype = float)
    if len(result) >= 4:
        result = result[:4]
    elif len(result) >= 2:
        result = result[:2]

    onerange = len(result) == 2 and sorted(result) and all([min_g <= rr <= max_g for rr in result])
    tworange = len(result) == 4 and sorted(result[:2]) and sorted(result[2:4]) and all([min_g <= rr <= max_g for rr in result[:2]]) and all([min_g <= rr <= max_g for rr in result[2:4]])

    if not (onerange or tworange):
        result = [min_g, max_g, min_g, max_g]
    elif onerange:
        result += result
    return result

def generate_g_grid(pair, ggrids = "", gmode = "", propersig = False, grange = default_g_range(), initial_distance = (1., 1.), randaround = (-1., -1.), randnminmax = default_nminmax()):
    if not hasattr(generate_g_grid, "alphas"):
        generate_g_grid.alphas = [0.6827, 0.9545, 0.9973, 0.999937, 0.9999997] if propersig else [0.68, 0.95, 0.9973, 0.999937, 0.9999997]
        generate_g_grid.alphas = [1. - pval for pval in generate_g_grid.alphas]
        generate_g_grid.nomore = 25 # n passing toys above which we declare the point is 'confidently' inside
        generate_g_grid.atleast = 4 # n total toys (divided by alphas) below which we consider it unreliable to say a point is in/outside

    g_grid = []

    if ggrids != "":
        ggrids = ggrids.replace(" ", "").split(",")

        for ggrid in ggrids:
            if not os.path.isfile(ggrid):
                print("g grid file " + ggrid + " is not accessible, skipping...")
                return g_grid

            with open(ggrid) as ff:
                contour = json.load(ff, object_pairs_hook = OrderedDict)

            if contour["points"] != pair:
                print("given pair in grid is inconsistent with currently expected pair, skipping...")
                print("in grid: ", contour["points"])
                print("currently expected: ", pair)
                return g_grid

            best_fit = contour["best_fit_g1_g2_dnll"]
            galready = [tuplize(gv) for gv in contour["g-grid"].keys()]

            if gmode == "random":
                around = [best_fit[0] if randaround[0] < 0. else randaround[0], best_fit[1] if randaround[1] < 0. else randaround[1]]
                npoint = int(randnminmax[0]) if randnminmax[0] > 0 else 32
                imin, imax = randnminmax[1], randnminmax[2]
                if imax <= imin:
                    imin, imax = default_nminmax()[1:]
                ipoint = 0

                while ipoint < npoint:
                    deltas = [uniform(imin, imax), uniform(imin, imax)]
                    signs = [1. if coinflip() else -1., 1. if coinflip() else -1.]
                    gtorun = [max(grange[0], min(round(gvalue + (delta * sign), gstr_precision), grange[1])) for gvalue, delta, sign in zip(around, deltas, signs)]
                    if tuple(gtorun) not in galready:
                        g_grid.append( tuple(gtorun) + (0,) )
                        ipoint += 1

            if gmode == "add" or gmode == "brim":
                for gv in contour["g-grid"].keys():
                    if contour["g-grid"][gv] is not None:
                        ntoy = (contour["g-grid"][gv]["total"],) if gmode == "brim" else (0,)
                        gt = tuplize(gv)
                        if not any([gt == (ggt[0], ggt[1]) for ggt in g_grid]):
                            npass = contour["g-grid"][gv]["pass"]
                            if npass < generate_g_grid.nomore:
                                g_grid.append(gt + ntoy)

            if gmode == "refine":
                mintoy = sys.maxsize
                for gv in contour["g-grid"].keys():
                    mintoy = min(mintoy, contour["g-grid"][gv]["total"] if contour["g-grid"][gv] is not None else sys.maxsize)

                cuts = [mintoy > (generate_g_grid.atleast / alpha) for alpha in generate_g_grid.alphas]
                if sum([1 if cut else 0 for cut in cuts]) < 1:
                    print("minimum toy count: " + str(mintoy))
                    raise RuntimeError("minimum toys insufficient to determine 1 sigma contour!! likely unintended, recheck!!")

                gts = [tuplize(gv) for gv in contour["g-grid"].keys() if contour["g-grid"][gv] is not None]
                effs = [float(contour["g-grid"][gv]["pass"]) / float(contour["g-grid"][gv]["total"]) for gv in contour["g-grid"].keys() if contour["g-grid"][gv] is not None]

                # add the best fit point into list of grid points, by construction the 0 sigma point
                gts.append((round(best_fit[0], gstr_precision), round(best_fit[1], gstr_precision)))
                effs.append(1.)

                tmpgrid = []
                nnearest = 3
                minsqd = 2.**-9
                for gt, eff in zip(gts, effs):
                    unary_q1sqd = lambda pp: q1sqd(gt, pp[0])
                    gxy = sorted([(gg, ee) for gg, ee in zip(gts, effs)], key = unary_q1sqd)
                    q1nearest = [gxy[1] if len(gxy) > 1 else None]

                    for igxy in range(2, len(gxy)):
                        if all([sqd(gxy[igxy][0], gg[0]) > minsqd and abs(angle(gxy[igxy][0], gg[0])) > math.pi / 8. for gg in q1nearest]):
                            q1nearest.append(gxy[igxy])
                        if len(q1nearest) >= nnearest:
                            break
                    while len(q1nearest) < nnearest:
                        q1nearest.append(None)

                    for cut, alpha in zip(cuts, generate_g_grid.alphas):
                        if cut:
                            differences = [gg is not None and ((gg[1] > alpha and eff < alpha) or (gg[1] < alpha and eff > alpha)) for gg in q1nearest]
                            if any(differences):
                                halfsies = []
                                for ig1, g1 in enumerate(q1nearest):
                                    if g1 is None:
                                        continue

                                    for ig2, g2 in enumerate(q1nearest):
                                        if ig2 <= ig1 or g2 is None:
                                            continue

                                        halfsies.append((g1[0], gt))
                                        halfsies.append((g1[0], g2[0]))
                                halfsies = [halfway(p1, p2) for p1, p2 in halfsies]
                                halfsies = [half for half in halfsies if not any([half == (ggt[0], ggt[1]) for ggt in tmpgrid])]

                                for half in halfsies:
                                    tmpgrid.append(half + (0,))

                if len(tmpgrid) == 0 and any([initdist >= 0.125 for initdist in initial_distance]):
                    # if we cant refine the grid, it can only mean nothing belongs to the contour
                    # the bane of good sensitivity - can only regenerate LO, but with a finer comb
                    # 31 is NNLO, if we have nothing within max sigma at 0.125 granularity uh oh that's a lot of points to scan
                    for gv in contour["g-grid"].keys():
                        if contour["g-grid"][gv] is not None:
                            gt = tuplize(gv)
                            tmpgrid.append(gt + (0,))
                    tmpgrid = list(set(make_initial_grid(grange, (max(initial_distance[0] / 2., 0.125), max(initial_distance[1] / 2., 0.125)))) - set(tmpgrid) - set(g_grid))
                g_grid += tmpgrid

        return g_grid

    # default LO case
    return make_initial_grid(grange, initial_distance)

def toy_locations(base, savetoy, gvalues, indices, max_per_dir = max_nfile_per_dir):
    toylocs = []
    if savetoy:
        for ii, idx in enumerate(idxs):
            if ii % max_per_dir == 0:
                toyloc = make_timestamp_dir(base = base, prefix = "toys") if base != "" else ""
            toylocs.append(toyloc)
    elif base != "" and not savetoy:
        gstr = g_in_filename(gvalues)
        toylocs = recursive_glob("{opd}".format(opd = base), "*_toys_{gstr}_n*.root".format(gstr = gstr if gstr != "" else "*"))
        if len(toylocs) < len(idxs):
            raise RuntimeError("expecting at least as many toy files as there are run indices in {opd}!!".format(opd = args.toyloc))
        shuffle(toylocs)
    return toylocs

if __name__ == '__main__':
    print("submit_twin :: called with the following arguments")
    print(sys.argv[1:])
    print("\n")
    print(" ".join(sys.argv))
    print("\n")
    sys.stdout.flush()

    parser = ArgumentParser()
    common_common(parser)
    common_fit_pure(parser)
    common_fit_forwarded(parser)
    make_datacard_pure(parser)
    make_datacard_forwarded(parser)
    common_2D(parser)
    common_submit(parser)

    parser.add_argument("--point", help = submit_help_messages["--point"], default = "", required = True,
                        type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))

    parser.add_argument("--run-idxs", help = submit_help_messages["--run-idxs"], default = "-1", dest = "runidxs", required = False, type = remove_spaces_quotes)
    parser.add_argument("--fc-single-point", help = submit_help_messages["--fc-single-point"], dest = "fcsinglepnt", action = "store_true", required = False)
    parser.add_argument("--fc-mode", help = submit_help_messages["--fc-mode"], default = "", dest = "fcmode", required = False, type = remove_spaces_quotes)

    parser.add_argument("--fc-g-grid", help = submit_help_messages["--fc-g-grid"], default = "", dest = "fcgrid", required = False,
                        type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))
    parser.add_argument("--fc-initial-distance", help = submit_help_messages["--fc-initial-distance"], default = default_initial_distance(), dest = "fcinit", required = False,
                        type = default_initial_distance)
    parser.add_argument("--fc-submit-also", help = submit_help_messages["--fc-submit-also"], default = "", dest = "fcsubalso", required = False,
                        type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' if ';' in s or re.search(r',[^eo]', remove_spaces_quotes(s)) else ',' ))

    parser.add_argument("--fc-g-min-max", help = submit_help_messages["--fc-g-min-max"], default = default_g_range(),
                        dest = "fcgrange", required = False, type = default_g_range)

    parser.add_argument("--fc-random-around", help = submit_help_messages["--fc-random-around"], default = "-1., -1.", dest = "fcrandaround",
                        required = False, type = lambda s: tuple([float(ss) for ss in tokenize_to_list( remove_spaces_quotes(s) )]))
    parser.add_argument("--fc-random-n-min-max", help = submit_help_messages["--fc-random-n-min-max"], default = default_nminmax(),
                        dest = "fcrandnminmax", required = False, type = default_nminmax)

    parser.add_argument("--nll-full-range", help = submit_help_messages["--nll-full-range"],
                        dest = "nllfullrange", default = "", required = False,
                        type = lambda s: [] if s == "" else tokenize_to_list(remove_spaces_quotes(s), ";"))
    parser.add_argument("--nll-njob", help = submit_help_messages["--nll-njob"], dest = "nllnjob",
                        default = "", required = False,
                        type = lambda s: [] if s == "" else [int(njob) for njob in tokenize_to_list(remove_spaces_quotes(s))])

    parser.add_argument("--proper-sigma", help = submit_help_messages["--proper-sigma"], dest = "propersig", action = "store_true", required = False)

    args = parse_args(parser)
    remove_mjf()
    scriptdir = os.path.dirname(os.path.abspath(__file__))

    pairs = args.point

    # handle the case of gridding pairs
    if len(pairs) == 4:
        pairgrid = [pp.split(",") for pp in pairs]
        if all([mm.startswith("m") for mm in pairgrid[0] + pairgrid[2]]) and all([ww.startswith("w") for ww in pairgrid[1] + pairgrid[3]]):
            for ii in range(4):
                if len(pairgrid[ii]) == 1 and '*' in pairgrid[ii][0]:
                    pairgrid[ii] = masses if ii % 2 == 0 else widths

            alla = []
            for mm in pairgrid[0]:
                for ww in pairgrid[1]:
                    if mm == "m343" and ww != "w2p0":
                        continue
                    alla.append("_".join(["A", mm, ww]))

            allh = []
            for mm in pairgrid[2]:
                for ww in pairgrid[3]:
                    if mm == "m343" and ww != "w2p0":
                        continue
                    allh.append("_".join(["H", mm, ww]))

            pairs = []
            for aa in alla:
                for hh in allh:
                    pairs.append(aa + "," + hh)

    ggrids = None
    if args.fcgrid != [] and args.fcmode != "":
        ggrids = args.fcgrid
    else:
        ggrids = ["" for pair in pairs]

    if ggrids is not None:
        if len(ggrids) != len(pairs):
            raise RuntimeError("there needs to be as many json file groups as there are pairs")
    else:
        ggrids = ["" for pair in pairs]

    rundc = "datacard" in args.mode or "workspace" in args.mode
    runclean = "clean" in args.mode
    rungen = "generate" in args.mode
    rungof = "gof" in args.mode
    runfc = "fc-scan" in args.mode or "contour" in args.mode
    runcc = "chancomp" in args.mode
    runhadd = "hadd" in args.mode or "merge" in args.mode
    runcompile = "compile" in args.mode
    runprepost = "prepost" in args.mode or "corrmat" in args.mode
    runpsfromws = "psfromws" in args.mode
    runnll = "nll" in args.mode or "likelihood" in args.mode

    if runcompile and (rundc or runfc or runhadd):
        raise RuntimeError("compile mode must be ran on its own!")

    if int(rungof) + int(runfc) + int(runcc) + int(runnll) > 1:
        raise RuntimeError("these are the worst modes in this suite, and you wanna run them together? lmao. edit this raise out by hand, then.")

    if (runfc or runcompile) and not args.asimov and "obs" not in args.fcexp:
        args.fcexp.append("obs")

    for pair, ggrid in zip(pairs, ggrids):
        dorundc = rundc
        flush_jobs()
        points = pair.split(',')
        pstr = '__'.join(points)

        if not dorundc and not os.path.isdir(pstr + args.tag) and os.path.isfile(pstr + args.tag + ".tar.gz"):
            syscall("tar xf {ttt} && rm {ttt}".format(ttt = pstr + args.tag + ".tar.gz"))

        mode = ""
        hasworkspace = os.path.isfile(pstr + args.tag + "/workspace_twin-g.root")
        if not dorundc and not hasworkspace:
            syscall("rm -r {ddd}".format(ddd = pstr + args.tag), True, True)
            dorundc = True
            mode = "datacard," + args.mode

        if os.path.isdir(pstr + args.tag):
            logs = glob.glob("twin_point_" + pstr + args.tag + "_*.o*")
            for ll in logs:
                if 'validate' in ll and problematic_datacard_log(ll):
                    print(("WARNING :: datacard of point {pstr} is tagged as problematic by problematic_datacard_log!!!\n\n\n".format(pstr = pstr + args.tag)))
                syscall("mv {lll} {ddd}".format(lll = ll, ddd = pstr + args.tag))

        if dorundc and os.path.isdir(pstr + args.tag):
            mode = args.mode.replace("datacard,", "").replace("datacard", "").replace("workspace,", "").replace("workspace", "")

            if mode != "":
                dorundc = False
            else:
                continue

        if runclean:
            for job in ["generate", "gof", "contour_g1_*_g2_*", "fc-scan_g1_*_g2*", "merge", "hadd", "compile"]:
                syscall("find {pnt}{tag} -type f -name 'twin_point_{pnt}{otg}_*{job}*.o*.*' | xargs rm".format(
                    pnt = pstr,
                    tag = args.tag,
                    otg = args.otag,
                    job = job), False, True)
            for tmps in ["fc-result", "gof-result", "chancomp-result", "toys"]:
                tmp = glob.glob(pstr + args.tag + "/" + tmps + "_*")
                for tm in tmp:
                    directory_to_delete(location = tm)
            directory_to_delete(location = None, flush = True)

        mode = mode if mode != "" else args.mode
        valid_g = any(float(gg) >= 0. for gg in args.gvalues)
        job_name = "twin_point_" + pstr + args.otag + "_" + "_".join(tokenize_to_list( remove_spaces_quotes(mode) ))
        job_name += "{poi}{gvl}{fix}".format(
            poi = "_" + args.poiset.replace(",", "__") if args.poiset != "" else "",
            gvl = "_" + g_in_filename(args.gvalues) if g_in_filename(args.gvalues) != "" else "",
            fix = "_fixed" if args.fixpoi and g_in_filename(args.gvalues) != "" else ""
        )
        job_arg = "--point {pnt} --mode {mmm} {sig} {rmr} {clt} {igp} {gvl} {fix} {exp}".format(
            pnt = pair,
            mmm = mode if not "clean" in mode else ','.join([mm for mm in mode.replace(" ", "").split(",") if "clean" not in mm]),
            sig = "--signal " + input_sig(args.signal, pair, args.inject, args.channel, args.year) if dorundc else "",
            rmr = "--delete-root" if args.rmroot else "",
            clt = "--collect-toy" if args.collecttoy else "",
            igp = "--ignore-previous" if args.ignoreprev else "",
            gvl = clamp_with_quote(
                string = ','.join(args.gvalues),
                prefix = "--g-values{s}".format(s = '=' if args.gvalues[0][0] == "-" else " ")
            ) if valid_g and not runfc else "",
            fix = "--fix-poi" if valid_g and args.fixpoi else "",
            exp = clamp_with_quote(
                string = ";".join(args.fcexp + args.fcsubalso) if runfc or runcompile else ";".join(args.fcexp),
                prefix = "--{fn}-expect{s}".format(
                    fn = "fc" if runfc or runcompile else "nll",
                    s = '=' if args.fcexp[0][0] == "-" else " "
                )
            ) if runfc or runnll or runcompile else ""
        )
        args.rundc = dorundc
        job_arg += common_job(args)

        if rungen or rungof or runfc or runcc:
            idxs = index_list(args.runidxs) if args.ntoy > 0 else [-1]

            if rungen:
                for ii, idx in enumerate(idxs):
                    toylocs = toy_locations(base = args.toyloc if args.toyloc != "" else pstr + args.tag if rungen else "",
                                            savetoy = rungen or args.savetoy, gvalues = args.gvalues, indices = idxs)
                    writelog = ii < 1 and args.writelog # write logs only for first toy job, unless explicitly disabled
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
                        opd = "--toy-location " + toylocs[ii] if toylocs[ii] != "" else ""
                    )

                    submit_job(jname, jarg, args.jobtime, 1, args.memory,
                               "." if dorundc else pstr + args.tag, scriptdir + "/twin_point_ahtt.py", True, args.runlocal, writelog)

            if rungof:
                toylocs = [""] + toy_locations(base = args.toyloc, savetoy = args.savetoy, gvalues = [-1, -1], indices = idxs)
                resdir = make_timestamp_dir(base = pstr + args.tag, prefix = "gof-result")
                expnres = 0

                for ii, idx in enumerate([-1] + idxs):
                    if expnres > max_nfile_per_dir:
                        resdir = make_timestamp_dir(base = pstr + args.tag, prefix = "gof-result")
                        expnres = 0

                    firstjob = ii == 0
                    writelog = ii < 2 and args.writelog # write logs only for best fit and first toy job unless explicitly disabled
                    sdx = '_' + str(idx) if idx != -1 else ''
                    jname = job_name + sdx
                    logs = glob.glob(pstr + args.tag + "/" + jname + ".o*")
                    roots = recursive_glob(pstr + args.tag, pstr + args.otag + "_gof-saturated_toys" + sdx + ".root")
                    gofrundat = args.gofrundat and firstjob

                    if not (args.runlocal and args.forcelocal):
                        if len(logs) > 0 or len(roots) > 0:
                            continue

                    jarg = job_arg
                    jarg += " {toy} {dat} {rsd} {idx}".format(
                        toy = "--n-toy " + str(args.ntoy) if args.ntoy > 0 and not firstjob else "--n-toy 0",
                        dat = "--gof-skip-data " if not gofrundat else "",
                        rsd = "--result-dir " + resdir,
                        idx = "--run-idx " + str(idx) if idx > -1 else ""
                    )

                    if len(toylocs) > 1 and not firstjob:
                        jarg += " --toy-location " + toylocs[ii]
                        if args.savetoy:
                            jarg += " --save-toy"

                    if not ("--gof-skip-data" in jarg and "--n-toy 0" in jarg):
                        expnres += 2 if firstjob and gofrundat else 2 if writelog else 1
                        submit_job(jname, jarg, args.jobtime, 1, args.memory,
                                   "." if dorundc else pstr + args.tag, scriptdir + "/twin_point_ahtt.py", True, args.runlocal, writelog)

            if runfc:
                if args.fcmode != "" and ggrid == "":
                    print("checking last grids")
                    for fcexp in args.fcexp:
                        ggg = glob.glob(pstr + args.tag + "/" + pstr + args.otag + "_fc-scan_" + expected_scenario(fcexp)[0] + "_*.json")
                        ggg.sort(key = os.path.getmtime)
                        ggrid += ggg[-1] if ggrid == "" else "," + ggg[-1]
                print("using the following grids:")
                print(ggrid)
                print()
                sys.stdout.flush()

                if args.fcsinglepnt:
                    gvalues = [tuple([float(gg) for gg in args.gvalues]) + (0,)]
                    toylocs = [""] + toy_locations(base = args.toyloc, savetoy = args.savetoy, gvalues = args.gvalues, indices = idxs)
                else:
                    gvalues = generate_g_grid(points, ggrid, args.fcmode, args.propersig, args.fcgrange, args.fcinit, args.fcrandaround, args.fcrandnminmax)

                sumtoy = args.ntoy * len(idxs)
                resdir = make_timestamp_dir(base = pstr + args.tag, prefix = "fc-result")
                expnres = 0
                for ig1, ig2, ntotal in gvalues:
                    ig1 = round(ig1, 8)
                    ig2 = round(ig2, 8)
                    scan_name = "_g1_" + str(ig1) + "_g2_" + str(ig2)
                    ndiff = max(0, sumtoy - ntotal) if args.fcmode == "brim" else 0
                    ndiff = int(math.ceil(float(ndiff) / args.ntoy)) if args.ntoy > 0 else 0

                    for ii, idx in enumerate([-1] + idxs):
                        if expnres > max_nfile_per_dir:
                            resdir = make_timestamp_dir(base = pstr + args.tag, prefix = "fc-result")
                            expnres = 0

                        if args.fcmode == "brim" and (ndiff == 0 or ii - 1 >= ndiff):
                            continue

                        firstjob = ii == 0
                        writelog = ii < 2 and args.writelog # write logs only for best fit and first toy job unless explicitly disabled
                        sdx = '_' + str(idx) if idx != -1 else ''
                        jname = job_name + scan_name + sdx
                        logs = glob.glob(pstr + args.tag + "/" + jname + ".o*")
                        fcrundat = args.fcrundat and firstjob
                        fmatches = ["_" + expected_scenario(fcexp)[0] for fcexp in args.fcexp + args.fcsubalso] if fcrundat else ["_toys" + sdx]
                        fmatches = ["{ptg}_fc-scan_pnt{snm}{sfx}.root".format(
                            ptg = pstr + args.otag,
                            snm = scan_name,
                            sfx = suffix
                        ) for suffix in fmatches]
                        roots = []
                        for fmatch in fmatches:
                            roots += recursive_glob(pstr + args.tag, fmatch)
                        if not (args.runlocal and args.forcelocal):
                            hasroots = len(roots) >= len(args.fcexp + args.fcsubalso) if fcrundat else len(roots) > 0
                            if len(logs) > 0 or hasroots:
                                continue

                        jarg = job_arg
                        jarg += " {gvl} {toy} {dat} {rsd} {idx}".format(
                            gvl = "--g-values '" + str(ig1) + "," + str(ig2) + "'",
                            toy = "--n-toy " + str(args.ntoy) if args.ntoy > 0 and not firstjob else "--n-toy 0",
                            dat = "--fc-skip-data " if not fcrundat else "",
                            rsd = "--result-dir " + resdir,
                            idx = "--run-idx " + str(idx) if idx > -1 else ""
                        )

                        if args.fcsinglepnt and len(toylocs) > 1 and not firstjob:
                            jarg += " --toy-location " + toylocs[ii]
                            if args.savetoy:
                                jarg += " --save-toy"

                        if not ("--fc-skip-data" in jarg and "--n-toy 0" in jarg):
                            expnres += 2 * len(args.fcexp + args.fcsubalso) if firstjob and fcrundat else 2 if writelog else 1
                            submit_job(jname, jarg, args.jobtime, 1, args.memory,
                                       "." if dorundc else pstr + args.tag, scriptdir + "/twin_point_ahtt.py", True, args.runlocal, writelog)

            if runcc:
                resdir = make_timestamp_dir(base = pstr + args.tag, prefix = "chancomp-result")
                cctag = "chancomp{mm}".format(mm = "" if len(args.ccmasks) == 0 else "_" + "_".join(args.ccmasks))
                expnres = 0

                for ii, idx in enumerate([-1] + idxs):
                    if expnres > max_nfile_per_dir:
                        resdir = make_timestamp_dir(base = pstr + args.tag, prefix = "chancomp-result")
                        expnres = 0

                    firstjob = ii == 0
                    writelog = ii < 2 and args.writelog # write logs only for best fit and first toy job unless explicitly disabled
                    sdx = '_' + str(idx) if idx != -1 else ''
                    jname = job_name + "{mm}".format(mm = "" if len(args.ccmasks) == 0 else "_" + "_".join(args.ccmasks))  + sdx
                    logs = glob.glob(pstr + args.tag + "/" + jname + ".o*")
                    roots = recursive_glob(pstr + args.tag, pstr + args.otag + "_" + cctag + "_toys" + sdx + ".root")
                    ccrundat = args.ccrundat and firstjob

                    if not (args.runlocal and args.forcelocal):
                        if len(logs) > 0 or len(roots) > 0:
                            continue

                    jarg = job_arg
                    jarg += " {toy} {dat} {msk} {rsd} {idx}".format(
                        toy = "--n-toy " + str(args.ntoy) if args.ntoy > 0 and not firstjob else "--n-toy 0",
                        dat = "--cc-skip-data " if not ccrundat else "",
                        msk = "--cc-mask '{ccm}'".format(ccm = ",".join(args.ccmasks)) if len(args.ccmasks) else "",
                        rsd = "--result-dir " + resdir,
                        idx = "--run-idx " + str(idx) if idx > -1 else ""
                    )

                    if not ("--cc-skip-data" in jarg and "--n-toy 0" in jarg):
                        expnres += 2 if firstjob and ccrundat else 2 if writelog else 1
                        submit_job(jname, jarg, args.jobtime, 1, args.memory,
                                   "." if dorundc else pstr + args.tag, scriptdir + "/twin_point_ahtt.py", True, args.runlocal, writelog)

        elif runnll:
            for nllwindow in args.nllfullrange:
                minmax = [(values.split(",")[0], values.split(",")[1]) for values in nllwindow]
                jname = job_name + "_" + args.fcexp[0]
                jname += "_" + "_".join(["{pp}_{mi}to{ma}".format(pp = pp, mi = floattopm(mm[0]), ma = floattopm(mm[1])) for pp, mm in zip(args.nllparam[:len(minmax)], minmax)])
                if len(minmax) < len(args.nllparam):
                    jname += "_" + "_".join(args.nllparam[len(minmax):])
                logs = glob.glob(pstr + args.tag + "/" + jname + ".o*")

                if not (args.runlocal and args.forcelocal):
                    if len(logs) > 0:
                        continue

                jarg = job_arg
                jarg += " {par} {win} {pnt} {uco}".format(
                    par = clamp_with_quote(string = ",".join(args.nllparam), prefix = '--nll-parameter '),
                    win = clamp_with_quote(string = ";".join(nllwindow), prefix = '--nll-interval='),
                    pnt = clamp_with_quote(string = ",".join([str(npnt) for npnt in args.nllnpnt]), prefix = '--nll-npoint '),
                    uco = clamp_with_quote(string = ",".join(args.nllunconstrained), prefix = '--nll-unconstrained '),
                )

                submit_job(jname, jarg, args.jobtime, 1, args.memory,
                           "." if dorundc else pstr + args.tag, scriptdir + "/twin_point_ahtt.py",
                           True, args.runlocal, args.writelog)

        elif runprepost or runpsfromws:
            jname = job_name + "_" + args.prepostfit
            if runpsfromws:
                jname += "_" + '-'.join(args.prepostmerge)

            logs = glob.glob(pstr + args.tag + "/" + jname + ".o*")
            if not (args.runlocal and args.forcelocal):
                if len(logs) > 0:
                    continue

            jarg = job_arg
            jarg += " {ppf} {ppm} {ppr}".format(
                ppf = clamp_with_quote(string = args.prepostfit, prefix = '--prepost-fit '),
                ppm = clamp_with_quote(
                    string = ','.join(args.prepostmerge),
                    prefix = "--prepost-merge "
                ) if runpsfromws else "",
                ppr = clamp_with_quote(
                    string = args.prepostres,
                    prefix = "--prepost-result "
                ) if runpsfromws else "",
            )

            submit_job(jname, jarg, args.jobtime, 1, args.memory,
                       "." if dorundc else pstr + args.tag, scriptdir + "/twin_point_ahtt.py",
                       True, args.runlocal, args.writelog)
        else:
            logs = glob.glob(pstr + args.tag + "/" + job_name + ".o*")

            if not (args.runlocal and args.forcelocal):
                if len(logs) > 0:
                    continue


            if len([mm for mm in mode.replace(" ", "").split(",") if "clean" not in mm and mm != ""]) > 0:
                submit_job(job_name, job_arg, args.jobtime, 1, args.memory,
                           "." if dorundc else pstr + args.tag, scriptdir + "/twin_point_ahtt.py",
                           True, runcompile or args.runlocal, args.writelog)
    flush_jobs()
