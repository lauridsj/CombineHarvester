#!/usr/bin/env python
# submit the two-point jobs and compiles its results

from argparse import ArgumentParser
import os
import sys
import glob
import numpy as np
import copy

from random import shuffle
from collections import OrderedDict
import json
import math
from datetime import datetime

from utilspy import syscall, tuplize, g_in_filename, recursive_glob, index_list, make_timestamp_dir, directory_to_delete, max_nfile_per_dir, floattopm
from utilslab import input_base, input_bkg, input_sig, remove_mjf
from utilscombine import problematic_datacard_log, min_g, max_g
from utilshtc import submit_job, aggregate_submit, flush_jobs

from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes
from argumentative import common_common, common_fit_pure, common_fit_forwarded, make_datacard_pure, make_datacard_forwarded, common_2D
from argumentative import common_submit, parse_args
from hilfemir import combine_help_messages, submit_help_messages

from twin_point_ahtt import expected_scenario

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

            if gmode == "add" or gmode == "brim":
                for gv in contour["g-grid"].keys():
                    if contour["g-grid"][gv] is not None:
                        ntoy = (contour["g-grid"][gv]["total"],) if gmode == "brim" else (0,)
                        gt = tuplize(gv)
                        if not any([gt == (ggt[0], ggt[1]) for ggt in g_grid]):
                            g_grid.append(gt + ntoy)

            if gmode == "refine":
                mintoy = sys.maxsize
                for gv in contour["g-grid"].keys():
                    mintoy = min(mintoy, contour["g-grid"][gv]["total"] if contour["g-grid"][gv] is not None else sys.maxsize)

                cuts = [mintoy > (9. / alpha) for alpha in generate_g_grid.alphas]
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
                                    if not any([half == (ggt[0], ggt[1]) for ggt in g_grid]):
                                        g_grid.append(half + (0,))
        return g_grid

    # default LO case
    gvls = [list(np.linspace(min_g, max_g, num = ndivision)), list(np.linspace(min_g, max_g, num = ndivision))]
    for ig1 in gvls[0]:
        for ig2 in gvls[1]:
            g_grid.append( (ig1, ig2, 0) )

    return g_grid

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
    print "submit_twin :: called with the following arguments"
    print sys.argv[1:]
    print "\n"
    print " ".join(sys.argv)
    print "\n"
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
    parser.add_argument("--fc-initial-distance", help = submit_help_messages["--fc-initial-distance"], default = 0.5, dest = "fcinit", required = False,
                        type = lambda s: float(remove_spaces_quotes(s)))

    parser.add_argument("--proper-sigma", help = submit_help_messages["--proper-sigma"], dest = "propersig", action = "store_true", required = False)

    args = parse_args(parser)

    remove_mjf()
    scriptdir = os.path.dirname(os.path.abspath(__file__))

    pairs = args.point

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
    rungen = "generate" in args.mode
    rungof = "gof" in args.mode
    runfc = "fc-scan" in args.mode or "contour" in args.mode
    runhadd = "hadd" in args.mode or "merge" in args.mode
    runcompile = "compile" in args.mode
    runprepost = "prepost" in args.mode or "corrmat" in args.mode
    runclean = "clean" in args.mode
    runnll = "nll" in args.mode or "likelihood" in args.mode

    if runcompile and (rundc or runfc or runhadd):
        raise RuntimeError("compile mode must be ran on its own!")

    if int(rungof) + int(runfc) + int(runnll) > 1:
        raise RuntimeError("these are the worst modes in this suite, and you wanna run them together? lmao. edit this raise out by hand, then.")

    if (runfc or runcompile) and not args.asimov and "obs" not in args.fcexp:
        args.fcexp.append("obs")

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
                if 'validate' in ll and problematic_datacard_log(ll):
                    print("WARNING :: datacard of point {pstr} is tagged as problematic by problematic_datacard_log!!!\n\n\n".format(pstr = pstr + args.tag))
                syscall("mv {lll} {ddd}".format(lll = ll, ddd = pstr + args.tag))

        if rundc and os.path.isdir(pstr + args.tag):
            args.mode = args.mode.replace("datacard,", "").replace("datacard", "").replace("workspace,", "").replace("workspace", "")

            if args.mode != "":
                rundc = False
            else:
                continue

        valid_g = any(float(gg) >= 0. for gg in args.gvalues)

        job_name = "twin_point_" + pstr + args.otag + "_" + "_".join(tokenize_to_list( remove_spaces_quotes(args.mode) ))
        job_arg = ("--point {pnt} --mode {mmm} {sus} {inj} {tag} {drp} {kee} {sig} {bkg} {cha} {yyy} {thr} {lns} {shp} {mcs} {rpr} {msk} {prj} "
                   "{cho} {rep} {fst} {hes} {kbf} {dws} {fr0} {frp} {asm} {rsd} {com} {dbg} {rmr} {igp} {gvl} {fix} {ext} {otg} {exp} {bsd}").format(
            pnt = pair,
            mmm = args.mode if not "clean" in args.mode else ','.join([mm for mm in args.mode.replace(" ", "").split(",") if "clean" not in mm]),
            sus = "--sushi-kfactor" if args.kfactor else "",
            inj = "--inject-signal " + args.inject if args.inject != "" else "",
            tag = "--tag " + args.tag if args.tag != "" else "",
            drp = "--drop '" + args.drop + "'" if args.drop != "" else "",
            kee = "--keep '" + args.keep + "'" if args.keep != "" else "",
            sig = "--signal " + input_sig(args.signal, pair, args.inject, args.channel, args.year) if rundc else "",
            bkg = "--background " + input_bkg(args.background, args.channel) if rundc else "",
            cha = "--channel " + args.channel,
            yyy = "--year " + args.year,
            thr = "--threshold " + args.threshold if args.threshold != "" else "",
            lns = "--lnN-under-threshold" if args.lnNsmall else "",
            shp = "--use-shape-always" if args.alwaysshape else "",
            mcs = "--no-mc-stats" if not args.mcstat else "",
            rpr = "--float-rate '" + args.rateparam + "'" if args.rateparam != "" else "",
            msk = "--mask '" + args.mask + "'" if args.mask != "" else "",
            prj = "--projection '" + args.projection + "'" if rundc and args.projection != "" else "",
            cho = "--chop-up '" + args.chop + "'" if args.chop != "" else "",
            rep = "--replace-nominal '" + args.repnom + "'" if args.repnom != "" else "",
            fst = "--fit-strategy {fst}".format(fst = args.fitstrat) if args.fitstrat > -1 else "",
            hes = "--use-hesse" if args.usehesse else "",
            kbf = "--redo-best-fit" if not args.keepbest else "",
            dws = "--default-workspace" if args.defaultwsp else "",
            fr0 = "--freeze-zero '" + args.frzzero + "'" if args.frzzero != "" else "",
            frp = "--freeze-post '" + args.frzpost + "'" if args.frzpost != "" else "",
            asm = "--unblind" if not args.asimov else "",
            rsd = "--seed " + args.seed if args.seed != "" else "",
            com = "--compress" if rundc else "",
            dbg = "--experimental" if args.experimental else "",
            rmr = "--delete-root" if args.rmroot else "",
            igp = "--ignore-previous" if args.ignoreprev else "",
            gvl = "--g-values{s}'".format(s = '=' if args.gvalues[0][0] == "-" else " ") + ','.join(args.gvalues) + "'" if valid_g and not runfc else "",
            fix = "--fix-poi" if valid_g and args.fixpoi else "",
            ext = "--extra-option{s}".format(s = '=' if args.extopt[0] == "-" else " ") + args.extopt if args.extopt != "" else "",
            otg = "--output-tag " + args.otag if args.otag != "" else "",
            exp = "--{fn}-expect{s}'".format(
                fn = "fc" if runfc or runcompile else "nll",
                s = '=' if args.fcexp[0][0] == "-" else " "
            ) + ";".join(args.fcexp) + "'" if runfc or runnll or runcompile else "",
            bsd = "" if rundc else "--base-directory " + os.path.abspath("./")
        )

        if rungen or rungof or runfc:
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

                    submit_job(agg, jname, jarg, args.jobtime, 1, "",
                               "." if rundc else pstr + args.tag, scriptdir + "/twin_point_ahtt.py", True, args.runlocal, writelog)

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
                        submit_job(agg, jname, jarg, "1200" if "--n-toy 0" in jarg and len(args.fcexp) < 10 else args.jobtime, 1, "",
                                   "." if rundc else pstr + args.tag, scriptdir + "/twin_point_ahtt.py", True, args.runlocal, writelog)

            if runfc:
                if args.fcmode != "" and ggrid == "":
                    print "checking last grids"
                    for fcexp in args.fcexp:
                        ggg = glob.glob(pstr + args.tag + "/" + pstr + args.otag + "_fc-scan_" + expected_scenario(fcexp)[0] + "_*.json")
                        ggg.sort(key = os.path.getmtime)
                        ggrid += ggg[-1] if ggrid == "" else "," + ggg[-1]
                print "using the following grids:"
                print ggrid
                print
                sys.stdout.flush()

                if args.fcsinglepnt:
                    gvalues = [tuple([float(gg) for gg in args.gvalues]) + (0,)]
                    toylocs = [""] + toy_locations(base = args.toyloc, savetoy = args.savetoy, gvalues = args.gvalues, indices = idxs)
                else:
                    gvalues = generate_g_grid(points, ggrid, args.fcmode, args.propersig,
                                              int(math.ceil((max_g - min_g) / args.fcinit)) + 1 if min_g < args.fcinit < max_g else 7)

                sumtoy = args.ntoy * len(idxs)
                resdir = make_timestamp_dir(base = pstr + args.tag, prefix = "fc-result")
                expnres = 0
                for ig1, ig2, ntotal in gvalues:
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
                        roots = recursive_glob(pstr + args.tag, pstr + args.otag + "_fc-scan_pnt" + scan_name + "_toys" + sdx + ".root")
                        fcrundat = args.fcrundat and firstjob

                        if not (args.runlocal and args.forcelocal):
                            if len(logs) > 0 or len(roots) > 0:
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
                            expnres += 2 * len(args.fcexp) if firstjob and fcrundat else 2 if writelog else 1
                            submit_job(agg, jname, jarg, "1200" if "--n-toy 0" in jarg and len(args.fcexp) < 10 else args.jobtime, 1, "",
                                       "." if rundc else pstr + args.tag, scriptdir + "/twin_point_ahtt.py", True, args.runlocal, writelog)

        elif runnll:
            minmax = [(values.split(",")[0], values.split(",")[1]) for values in args.nllwindow]
            jname = job_name + "_" + args.fcexp[0]
            jname += "_" + "_".join(["{pp}_{mi}to{ma}".format(pp = pp, mi = floattopm(mm[0]), ma = floattopm(mm[1])) for pp, mm in zip(args.nllparam[:len(minmax)], minmax)])
            if len(minmax) < len(args.nllparam):
                jname += "_" + "_".join(args.nllparam[len(minmax):])
            logs = glob.glob(pstr + args.tag + "/" + jname + ".o*")

            if not (args.runlocal and args.forcelocal):
                if len(logs) > 0:
                    continue

            jarg = job_arg
            jarg += " {par} {win} {pnt}".format(
                par = "--nll-parameter '" + ",".join(args.nllparam) + "'",
                win = "--nll-interval='" + ";".join(args.nllwindow) + "'" if args.nllwindow != [] else "",
                pnt = "--nll-npoint '" + ",".join([str(npnt) for npnt in args.nllnpnt]) + "'" if args.nllnpnt != [] else "",
            )

            submit_job(agg, jname, jarg, args.jobtime, 1, "",
                       "." if rundc else pstr + args.tag, scriptdir + "/twin_point_ahtt.py",
                       True, args.runlocal, args.writelog)
        else:
            if runclean:
                for job in ["generate", "gof", "contour_g1_*_g2_*", "fc-scan_g1_*_g2*", "merge", "hadd", "compile"]:
                    syscall("find {pnt}{tag} -type f -name 'twin_point_{pnt}{otg}_*{job}*.o*.*' | xargs rm".format(
                        pnt = pstr,
                        tag = args.tag,
                        otg = args.otag,
                        job = job), False, True)
                # FIXME buggy in current state since it indiscriminately cleans, even dirs that are to be used for future output
                #for tmps in ["fc-result", "toys"]:
                #    tmp = glob.glob(pstr + args.tag + "/" + tmps + "_*")
                #    for tm in tmp:
                #        directory_to_delete(location = tm)

            logs = glob.glob(pstr + args.tag + "/" + job_name + ".o*")

            if not (args.runlocal and args.forcelocal):
                if len(logs) > 0:
                    continue

            #job_mem = "12 GB" if runprepost and not (args.frzbb0 or args.frzbbp or args.frznui) else ""
            job_mem = ""

            if len([mm for mm in args.mode.replace(" ", "").split(",") if "clean" not in mm]) > 0:
                submit_job(agg, job_name, job_arg, args.jobtime, 1, job_mem,
                           "." if rundc else pstr + args.tag, scriptdir + "/twin_point_ahtt.py",
                           True, runcompile or args.runlocal, args.writelog)

        flush_jobs(agg)
