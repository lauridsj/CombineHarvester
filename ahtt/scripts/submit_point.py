#!/usr/bin/env python
# submits single_point_ahtt jobs
# for imode in 'datacard,validate'; do ./../scripts/submit_point.py --sushi-kfactor --lnN-under-threshold --use-pseudodata --year "${years}" --channel "${channels}" --tag "${tag}" --keep "${keeps}" --drop "${drops}" --mode "${imode}"; done
# get the shell vars from run_fc

from argparse import ArgumentParser
import os
import sys
import glob
import copy
from collections import OrderedDict

from utilities import syscall, submit_job, aggregate_submit, chunks, get_nbin, input_bkg, input_sig
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes
from argumentative import common_point, common_common, common_fit_pure, common_fit_forwarded, make_datacard_pure, make_datacard_forwarded, common_1D, common_submit
from hilfemir import combine_help_messages, submit_help_messages

if __name__ == '__main__':
    parser = ArgumentParser()
    common_point(parser)
    common_common(parser)
    common_fit_pure(parser)
    common_fit_forwarded(parser)
    make_datacard_pure(parser)
    make_datacard_forwarded(parser)
    common_1D(parser)
    common_submit(parser)

    parser.add_argument("--raster-i", help = submit_help_messages["--raster-i"], dest = "ichunk", default = "...[--raster-n value]", required = False)
    parser.add_argument("--impact-n", help = submit_help_messages["--impact-n"], dest = "nnuisance", default = 10, required = False, type = lambda s: int(remove_spaces_quotes(s)))
    parser.add_argument("--skip-expth", help = submit_help_messages["--skip-expth"], dest = "runexpth", action = "store_false", required = False)
    parser.add_argument("--run-mc-stats", help = submit_help_messages["--run-mc-stats"], dest = "runbb", action = "store_true", required = False)

    args = parser.parse_args()
    scriptdir = os.path.dirname(os.path.abspath(__file__))

    parities = ("A", "H")
    masses = tuple(["m365", "m380"] + ["m" + str(mm) for mm in range(400, 1001, 25)])
    widths = ("w0p5", "w1p0", "w1p5", "w2p0", "w2p5", "w3p0", "w4p0", "w5p0", "w8p0", "w10p0", "w13p0", "w15p0", "w18p0", "w21p0", "w25p0")
    points = []
    keep_point = args.point

    for parity in parities:
        for mass in masses:
            for width in widths:
                pnt = "_".join([parity, mass, width])

                if args.point == "" or any([kk in pnt for kk in keep_point]):
                    points.append(pnt)

    rundc = "datacard" in args.mode or "workspace" in args.mode
    runlimit = "limit" in args.mode
    runpull = "pull" in args.mode or "impact" in args.mode

    # generate an aggregate submission file name
    agg = aggregate_submit()

    for pnt in points:
        if not rundc and not os.path.isdir(pnt + args.tag) and os.path.isfile(pnt + args.tag + ".tar.gz"):
            syscall("tar xf {ttt} && rm {ttt}".format(ttt = pnt + args.tag + ".tar.gz"))

        hasworkspace = os.path.isfile(pnt + args.tag + "/workspace_g-scan.root") and os.path.isfile(pnt + args.tag + "/workspace_one-poi.root")
        if not rundc and not hasworkspace:
            syscall("rm -r {ddd}".format(ddd = pnt + args.tag))
            rundc = True
            args.mode = "datacard," + args.mode

        if os.path.isdir(pnt + args.tag):
            logs = glob.glob("single_point_" + pnt + args.tag + "_*.o*")
            for ll in logs:
                syscall("mv {lll} {ddd}".format(lll = ll, ddd = pnt + args.tag))

        if rundc and os.path.isdir(pnt + args.tag):
            args.mode = args.mode.replace("datacard,", "").replace("datacard", "").replace("workspace,", "").replace("workspace", "")

            if args.mode != "":
                rundc = False
            else:
                continue

        job_name = "single_point_" + pnt + args.tag + "_" + "_".join(args.mode.replace(" ", "").split(","))
        job_name += "{mod}{gvl}{rvl}{fix}".format(
            mod = "" if rundc else "_one-poi" if args.onepoi else "_g-scan",
            gvl = "_g_" + str(args.setg).replace('.', 'p') if args.setg >= 0. else "",
            rvl = "_r_" + str(args.setr).replace('.', 'p') if args.setr >= 0. and not args.onepoi else "",
            fix = "_fixed" if args.fixpoi and (args.setg >= 0. or args.setr >= 0.) else ""
        )

        job_arg = ('--point {pnt} --mode {mmm} {sus} {psd} {inj} {tag} {drp} {kee} {sig} {bkg} {cha} {yyy} {thr} {lns} '
                   '{shp} {mcs} {rpr} {msk} {prj} {frz} {asm} {one} {gvl} {rvl} {fix} {ext} {rsd} {com} {bsd}').format(
                       pnt = pnt,
                       mmm = args.mode,
                       sus = "--sushi-kfactor" if args.kfactor else "",
                       psd = "--use-pseudodata" if args.pseudodata else "",
                       inj = "--inject-signal " + args.inject if args.inject != "" else "",
                       tag = "--tag " + args.tag if args.tag != "" else "",
                       drp = "--drop '" + args.drop + "'" if args.drop != "" else "",
                       kee = "--keep '" + args.keep + "'" if args.keep != "" else "",
                       sig = "--signal " + input_sig(args.signal, pnt, args.inject, args.channel, args.year),
                       bkg = "--background " + input_bkg(args.background, args.channel),
                       cha = "--channel " + args.channel,
                       yyy = "--year " + args.year,
                       thr = "--threshold " + str(args.threshold),
                       lns = "--lnN-under-threshold" if args.lnNsmall else "",
                       shp = "--use-shape-always" if args.alwaysshape else "",
                       mcs = "--no-mc-stats" if not args.mcstat else "",
                       rpr = "--float-rate '" + args.rateparam + "'" if args.rateparam != "" else "",
                       msk = "--mask '" + args.mask + "'" if args.mask != "" else "",
                       prj = "--projection '" + args.projection + "'" if rundc and args.projection != "" else "",
                       frz = "--freeze-mc-stats-zero" if args.frzbb0 else "--freeze-mc-stats-post" if args.frzbbp else "--freeze-nuisance-post" if args.frznui else "",
                       asm = "--unblind" if not args.asimov else "",
                       one = "--one-poi" if args.onepoi else "",
                       gvl = "--g-value " + str(args.setg) if args.setg >= 0. else "",
                       rvl = "--r-value " + str(args.setr) if args.setr >= 0. else "",
                       fix = "--fix-poi" if args.fixpoi and (args.setg >= 0. or args.setr >= 0.) else "",
                       ext = "--extra-option '" + args.extopt + "'" if args.extopt != "" else "",
                       rsd = "--seed " + args.seed if args.seed != "" else "",
                       com = "--compress" if rundc else "",
                       bsd = "" if rundc else "--base-directory " + os.path.abspath("./")
                   )

        if runlimit and not args.onepoi:
            if args.nchunk < 0:
                args.nchunk = 6

            if args.ichunk == "...[--raster-n value]":
                args.ichunk = "..." + str(args.nchunk)

            idxs = []
            if args.nchunk > 1:
                if "," in args.ichunk and "..." in args.ichunk:
                    raise RuntimeError("it is said that mixing syntaxes is not allowed smh.")
                elif "," in args.ichunk:
                    idxs = [int(ii) for ii in args.ichunk.replace(" ", "").split(",")]
                elif "..." in args.ichunk:
                    idxs = args.ichunk.replace(" ", "").split("...")
                    idxs = range(int(idxs[0]), int(idxs[1])) if idxs[0] != "" else range(int(idxs[1]))
                else:
                    idxs = [int(args.ichunk.replace(" ", ""))]
            else:
                idxs = [0]

            for idx in idxs:
                jname = job_name.replace("g-scan", "g-scan_{nch}_{idx}".format(nch = "n" + str(args.nchunk), idx = "i" + str(idx)))
                logs = glob.glob(pnt + args.tag + "/" + jname + ".o*")

                if not (args.runlocal and args.forcelocal):
                    if len(logs) > 0:
                        continue

                jarg = job_arg
                jarg += " {nch} {idx}".format(
                    nch = "--raster-n " + str(args.nchunk),
                    idx = "--raster-i " + str(idx)
                )

                submit_job(agg, jname, jarg, args.jobtime, 1, "",
                           "." if rundc else "$(readlink -f " + pnt + args.tag + ")", scriptdir + "/single_point_ahtt.py", True, args.runlocal)
        elif runpull:
            if args.nnuisance < 0:
                args.nnuisance = 25

            nuisances = OrderedDict()
            if args.runexpth:
                syscall("python {cms}/src/HiggsAnalysis/CombinedLimit/test/systematicsAnalyzer.py --format brief --all {dcd}/ahtt_{ch}.txt | "
                        "grep -v -e 'NUISANCE (TYPE)' | grep -v -e '--------------------------------------------------' | awk {awk} "
                        "> {dcd}/{nui} && grep rateParam {dcd}/ahtt_{ch}.txt | awk {awk} | sort -u >> {dcd}/{nui}".format(
                            cms = r'${CMSSW_BASE}',
                            dcd = pnt + args.tag,
                            ch = "combined" if "," in args.channel or "," in args.year else args.channel + "_" + args.year,
                            nui = "ahtt_nuisance.txt",
                            awk = r"'{print $1}'"
                        ))

                with open(pnt + args.tag + "/ahtt_nuisance.txt") as fexp:
                    nparts = fexp.readlines()
                    nparts = [et.rstrip() for et in nparts]
                    nsplit = (len(nparts) // args.nnuisance) + 1
                    nparts = chunks(nparts, nsplit)

                    for ip, ipart in enumerate(nparts):
                        group = "expth_{ii}".format(ii = str(ip))
                        nuisances[group] = copy.deepcopy(ipart)
                syscall('rm {nui}'.format(nui = pnt + args.tag + "/ahtt_nuisance.txt"), False)

            if args.runbb:
                for cc in args.channel.replace(" ", "").split(','):
                    for yy in args.year.replace(" ", "").split(','):
                        if cc + "_" + yy in args.mask:
                            continue

                        nbin = get_nbin(pnt + args.tag + "/ahtt_input.root", cc, yy)
                        nsplit = (nbin // args.nnuisance) + 1 
                        nparts = chunks(range(nbin), nsplit)

                        for ip, ipart in enumerate(nparts):
                            group = "mcstat_{cc}_{yy}_{ii}".format(cc = cc, yy = yy, ii = str(ip))
                            mcstats = ["prop_bin" + "{cc}_{yy}_bin".format(cc = cc, yy = yy) + str(ii) for ii in ipart]
                            nuisances[group] = copy.deepcopy(mcstats)

            for group, nuisance in nuisances.items():
                jname = job_name + "_" + group
                logs = glob.glob(pnt + args.tag + "/" + jname + ".o*")

                if not (args.runlocal and args.forcelocal):
                    if len(logs) > 0:
                        continue

                jarg = job_arg
                jarg += " --impact-nuisances '{grp};{nui}'".format(grp = group, nui = ",".join(nuisance))

                submit_job(agg, jname, jarg, args.jobtime, 1, "",
                           "." if rundc else "$(readlink -f " + pnt + args.tag + ")", scriptdir + "/single_point_ahtt.py", True, args.runlocal)
        else:
            logs = glob.glob(pnt + args.tag + "/" + job_name + ".o*")

            if not (args.runlocal and args.forcelocal):
                if len(logs) > 0:
                    continue

            submit_job(agg, job_name, job_arg, args.jobtime, 1, "",
                       "." if rundc else "$(readlink -f " + pnt + args.tag + ")", scriptdir + "/single_point_ahtt.py", True, args.runlocal)

    if os.path.isfile(agg):
        syscall('condor_submit {agg}'.format(agg = agg), False)
        syscall('rm {agg}'.format(agg = agg), False)
