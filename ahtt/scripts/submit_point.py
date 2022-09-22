#!/usr/bin/env python
# submits single_point_ahtt jobs
# for imode in 'datacard,validate'; do ./../scripts/submit_point.py --sushi-kfactor --lnN-under-threshold --use-pseudodata --year '2016pre,2016post,2017,2018' --channel 'ee,em,mm,e3j,e4pj,m3j,m4pj' --tag lx --keep 'eff_b,eff_e,eff_m_id,eff_m_iso,eff_trigger,fake,JEC,JER,MET,QCDscale,hdamp,tmass,EWK,alphaS,PDF_PCA_0,L1,EWQCD,pileup,lumi,norm' --mode ${imode}; done

from argparse import ArgumentParser
import os
import sys
import glob
import subprocess
import copy
from collections import OrderedDict

from utilities import syscall, submit_job, aggregate_submit, chunks, get_nbin, input_bkg, input_sig

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "desired signal point to run, comma separated", default = "", required = False)
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
    parser.add_argument("--no-mc-stats",
                        help = "don't add nuisances due to limited mc stats (barlow-beeston lite) in datacard mode, "
                        "or don't add the bb-lite analytical minimization option in others",
                        dest = "mcstat", action = "store_false", required = False)

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
    parser.add_argument("--one-poi", help = "use physics model with only g as poi", dest = "onepoi", action = "store_true", required = False)

    parser.add_argument("--raster-n", help = "number of chunks to split the g raster limit scan into",
                        dest = "nchunk", default = 6, required = False, type = int)
    parser.add_argument("--raster-i", help = "which chunks to process, in doing the raster scan.\n"
                        "can be one or more comma separated non-negative integers, or something of the form A...B where A < B and A, B non-negative\n"
                        "where the comma separated version is plainly the list of indices to be given to --raster-i, if --raster-n > 0\n"
                        "and the A...B version builds a list of indices from [A, B). If A is omitted, it is assumed to be 0\n"
                        "mixing of both syntaxes are not allowed.",
                        dest = "ichunk", default = "...[--raster-n value]", required = False)

    parser.add_argument("--impact-n", help = "maximum number of nuisances to run in a single impact job",
                        dest = "nnuisance", default = 25, required = False, type = int)
    parser.add_argument("--run-mc-stats",
                        help = "in pull/impact mode, run also over the BB nuisances individually. "
                        "this option does not affect their treatment in any way (analytical minimization)",
                        dest = "runbb", action = "store_true", required = False)

    parser.add_argument("--freeze-mc-stats-zero",
                        help = "only in the pull/impact/prepost/corrmat/nll mode, freeze mc stats nuisances to zero",
                        dest = "frzbb0", action = "store_true", required = False)
    parser.add_argument("--freeze-mc-stats-post",
                        help = "only in the pull/impact/prepost/corrmat/nll mode, freeze mc stats nuisances to the postfit values. "
                        "--freeze-mc-stats-zero takes priority over this option",
                        dest = "frzbbp", action = "store_true", required = False)
    parser.add_argument("--freeze-nuisance-post", help = "only in the prepost/corrmat/nll mode, freeze all nuisances to the postfit values.",
                        dest = "frznui", action = "store_true", required = False)

    parser.add_argument("--g-value",
                        help = "g to use when evaluating impacts/fit diagnostics/nll. "
                        "does NOT freeze the value, unless --fix-poi is also used. "
                        "note: semantically sets value of 'r' with --one-poi, as despite the name it plays the role of g.",
                        dest = "setg", default = -1., required = False, type = float)
    parser.add_argument("--r-value",
                        help = "r to use when evaluating impacts/fit diagnostics/nll, if --one-poi is not used."
                        "does NOT freeze the value, unless --fix-poi is also used.",
                        dest = "setr", default = -1., required = False, type = float)
    parser.add_argument("--fix-poi", help = "fix pois in the fit, through --g-value and/or --r-value",
                        dest = "fixpoi", action = "store_true", required = False)

    parser.add_argument("--job-time", help = "time to assign to each job", default = "", dest = "jobtime", required = False)
    parser.add_argument("--local", help = "run jobs locally, do not submit to HTC", dest = "runlocal", action = "store_true", required = False)

    args = parser.parse_args()
    if (args.tag != "" and not args.tag.startswith("_")):
        args.tag = "_" + args.tag
    scriptdir = os.path.dirname(os.path.abspath(__file__))

    parities = ("A", "H")
    masses = tuple(["m365", "m380"] + ["m" + str(mm) for mm in range(400, 1001, 25)])
    widths = ("w0p5", "w1p0", "w1p5", "w2p0", "w2p5", "w3p0", "w4p0", "w5p0", "w8p0", "w10p0", "w13p0", "w15p0", "w18p0", "w21p0", "w25p0")
    points = []
    keep_point = args.point.replace(" ", "").split(',')

    for parity in parities:
        for mass in masses:
            for width in widths:
                pnt = "_".join([parity, mass, width])

                if args.point == "" or any([kk in pnt for kk in keep_point]):
                    points.append(pnt)

    if args.injectsignal != "":
        args.pseudodata = True

    if args.jobtime != "":
        args.jobtime = "-t " + args.jobtime

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
                   '{shp} {mcs} {prj} {frz} {asm} {one} {gvl} {rvl} {fix} {rsd} {com} {bsd}').format(
                       pnt = pnt,
                       mmm = args.mode,
                       sus = "--sushi-kfactor" if args.kfactor else "",
                       psd = "--use-pseudodata" if args.pseudodata else "",
                       inj = "--inject-signal " + args.injectsignal if args.injectsignal != "" else "",
                       tag = "--tag " + args.tag if args.tag != "" else "",
                       drp = "--drop '" + args.drop + "'" if args.drop != "" else "",
                       kee = "--keep '" + args.keep + "'" if args.keep != "" else "",
                       sig = "--signal " + input_sig(args.signal, pnt, args.injectsignal, args.channel, args.year),
                       bkg = "--background " + input_bkg(args.background, args.channel),
                       cha = "--channel " + args.channel,
                       yyy = "--year " + args.year,
                       thr = "--threshold " + str(args.threshold),
                       lns = "--lnN-under-threshold" if args.lnNsmall else "",
                       shp = "--use-shape-always" if args.alwaysshape else "",
                       mcs = "--no-mc-stats" if not args.mcstat else "",
                       prj = "--projection '" + args.projection + "'" if rundc and args.projection != "" else "",
                       frz = "--freeze-mc-stats-zero" if args.frzbb0 else "--freeze-mc-stats-post" if args.frzbbp else "--freeze-nuisance-post" if args.frznui else "",
                       asm = "--unblind" if not args.asimov else "",
                       one = "--one-poi" if args.onepoi else "",
                       gvl = "--g-value " + str(args.setg) if args.setg >= 0. else "",
                       rvl = "--r-value " + str(args.setr) if args.setr >= 0. else "",
                       fix = "--fix-poi" if args.fixpoi and (args.setg >= 0. or args.setr >= 0.) else "",
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
            syscall("python {cms}/src/HiggsAnalysis/CombinedLimit/test/systematicsAnalyzer.py --format brief --all {dcd}/ahtt_{ch}.txt | "
                    "grep -v -e 'NUISANCE (TYPE)' | grep -v -e '--------------------------------------------------' | awk {awk} "
                    "> {dcd}/{nui}".format(
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

                if len(logs) > 0:
                    continue

                jarg = job_arg
                jarg += " --impact-nuisances '{grp};{nui}'".format(grp = group, nui = ",".join(nuisance))

                submit_job(agg, jname, jarg, args.jobtime, 1, "",
                           "." if rundc else "$(readlink -f " + pnt + args.tag + ")", scriptdir + "/single_point_ahtt.py", True, args.runlocal)
        else:
            logs = glob.glob(pnt + args.tag + "/" + job_name + ".o*")

            if len(logs) > 0:
                continue

            submit_job(agg, job_name, job_arg, args.jobtime, 1, "",
                       "." if rundc else "$(readlink -f " + pnt + args.tag + ")", scriptdir + "/single_point_ahtt.py", True, args.runlocal)

    if os.path.isfile(agg):
        syscall('condor_submit {agg}'.format(agg = agg), False)
        syscall('rm {agg}'.format(agg = agg), False)
