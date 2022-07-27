#!/usr/bin/env python
# submits single_point_ahtt jobs
# for imode in 'datacard,validate'; do ./../scripts/submit_point.py --sushi-kfactor --lnN-under-threshold --use-pseudodata --year '2016pre,2016post,2017,2018' --channel 'ee,em,mm' --tag ll_run2 --keep 'eff,fake,JEC,JER,MET,QCDscale,hdamp,tmass,EWK,alphaS,PDF_PCA_0,L1,pileup,lumi,norm' --mode ${imode}; done

from argparse import ArgumentParser
import os
import sys
import glob
import subprocess

from utilities import syscall, submit_job, aggregate_submit

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "desired signal point to run, comma separated", default = "", required = False)
    parser.add_argument("--mode", help = "combine mode to run, comma separated", default = "datacard,validate", required = False)

    parser.add_argument("--signal", help = "signal filenames. comma separated", default = "", required = False)
    parser.add_argument("--background", help = "data/background filenames. comma separated",
                        default = "/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/bkg_ll_3D-33.root", required = False)
    parser.add_argument("--channel", help = "final state channels considered in the analysis. comma separated", default = "ll", required = False)
    parser.add_argument("--year", help = "analysis year determining the correlation model to assume. comma separated", default = "2018", required = False)
    parser.add_argument("--drop",
                        help = "comma separated list of nuisances to be dropped in datacard mode. 'XX, YY' means all sources containing XX or YY are dropped. '*' to drop all",
                        default = "", required = False)
    parser.add_argument("--keep",
                        help = "comma separated list of nuisances to be kept in datacard mode. same syntax as --drop. implies everything else is dropped",
                        default = "", required = False)
    parser.add_argument("--tag", help = "extra tag to be put on datacard names", default = "", required = False)
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
    parser.add_argument("--freeze-mc-stats-zero", help = "only in the prepost/corrmat mode, freeze mc stats nuisances to zero",
                        dest = "frzbb0", action = "store_true", required = False)

    parser.add_argument("--unblind", help = "use data when fitting", dest = "asimov", action = "store_false", required = False)
    parser.add_argument("--one-poi", help = "use physics model with only g as poi", dest = "onepoi", action = "store_true", required = False)

    parser.add_argument("--impact-sb", help = "do sb impact fit instead of b", dest = "impactsb", action = "store_true", required = False)
    parser.add_argument("--g-value", help = "g value to use when evaluating impacts/fit diagnostics, if one-poi is not used",
                        dest = "fixg", default = 1, required = False, type = float)

    parser.add_argument("--job-time", help = "time to assign to each job", default = "", dest = "jobtime", required = False)

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
    resub = "resubmit" in args.mode

    # clean up before doing anything else, in case someone aborts previous run
    if os.path.isfile(aggregate_submit):
        syscall('rm {agg}'.format(agg = aggregate_submit), False, True)

    for pnt in points:
        signals = []
        if args.signal == "":
            if "_m3" in pnt or "_m1000" in pnt or "_m3" in args.injectsignal or "_m1000" in args.injectsignal:
                if any(cc in args.channel for cc in ["ee", "em", "mm"]):
                    signals.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/sig_ll_3D-33_m3xx_and_m1000.root")
            for im in ["_m4", "_m5", "_m6", "_m7", "_m8", "_m9"]:
                if im in pnt or im in args.injectsignal:
                    if any(cc in args.channel for cc in ["ee", "em", "mm"]):
                        signals.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/sig_ll_3D-33" + im + "xx.root")
            signal = ','.join(signals)
        else:
            signal = args.signal

        if resub:
            failures = [ff.split(' ')[0] for ff in subprocess.check_output("condor_check {ddd}".format(ddd = pnt + args.tag), shell = True).split("\n")]
            for failure in failures:
                if not os.path.isfile(failure):
                    continue

                options = subprocess.check_output("grep arguments -A 1 {fff} | tail -1".format(fff = failure), shell = True)
                options = options.replace('[', '').replace(']', '').replace("',", "'").replace("'", "").split(' ')

                job_name = "single_point_" + pnt + args.tag + "_" + "_".join(options[options.index("--mode") + 1].split(","))
                # only combine-specific settings are needed; this wont tag make_datacard failures as that's the first point of entry
                job_arg = '--point {pnt} {mmm} {tag} {mcs} {asm} {one} {ims} {gvl} {frz} {com} {bsd}'.format(
                    pnt = pnt,
                    mmm = "--mode " + options[options.index("--mode") + 1] if "--mode" in options else "",
                    tag = "--tag " + args.tag if args.tag != "" else "",
                    mcs = "--no-mc-stats" if "--no-mc-stats" in options else "",
                    asm = "--unblind" if "--unblind" in options else "",
                    one = "--one-poi" if "--one-poi" in options else "",
                    ims = "--impact-sb" if "--impact-sb" in options else "",
                    gvl = "--g-value " + options[options.index("--g-value") + 1] if "--g-value" in options else "",
                    frz = "--freeze-mc-stats-zero" if "--freeze-mc-stats-zero" in options else "",
                    com = "--compress" if "--compress" in options else "",
                    bsd = "--base-directory " + options[options.index("--base-directory") + 1] if "--base-directory" in options else ""
                )

                syscall("rm {fff}".format(fff = failure))
                submit_job(job_name, job_arg, args.jobtime, "-l $(readlink -f " + pnt + args.tag + ")", scriptdir + "/single_point_ahtt.py")
            continue

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
        job_name += "{mod}".format(mod = "" if rundc else "_one-poi" if args.onepoi else "_g-scan")
        logs = glob.glob(pnt + args.tag + "/" + job_name + ".o*")

        if len(logs) > 0:
            continue

        job_arg = ('--point {pnt} --mode {mmm} {sus} {psd} {inj} {tag} {drp} {kee} {sig} {bkg} {cha} {yyy} {thr} {lns} '
                   '{shp} {mcs} {prj} {frz} {asm} {one} {ims} {gvl} {com} {bsd}').format(
            pnt = pnt,
            mmm = args.mode,
            sus = "--sushi-kfactor" if args.kfactor else "",
            psd = "--use-pseudodata" if args.pseudodata else "",
            inj = "--inject-signal " + args.injectsignal if args.injectsignal != "" else "",
            tag = "--tag " + args.tag if args.tag != "" else "",
            drp = "--drop '" + args.drop + "'" if args.drop != "" else "",
            kee = "--keep '" + args.keep + "'" if args.keep != "" else "",
            sig = "--signal " + signal,
            bkg = "--background " + args.background,
            cha = "--channel " + args.channel,
            yyy = "--year " + args.year,
            thr = "--threshold " + str(args.threshold),
            lns = "--lnN-under-threshold" if args.lnNsmall else "",
            shp = "--use-shape-always" if args.alwaysshape else "",
            mcs = "--no-mc-stats" if not args.mcstat else "",
            prj = "--projection '" + args.projection + "'" if rundc and args.projection != "" else "",
            frz = "--freeze-mc-stats-zero" if args.frzbb0 else "",
            asm = "--unblind" if not args.asimov else "",
            one = "--one-poi" if args.onepoi else "",
            ims = "--impact-sb" if args.impactsb else "",
            gvl = "--g-value " + str(args.fixg),
            com = "--compress" if rundc else "",
            bsd = "" if rundc else "--base-directory " + os.path.abspath("./")
        )

        submit_job(job_name, job_arg, args.jobtime, "" if rundc else "-l $(readlink -f " + pnt + args.tag + ")", scriptdir + "/single_point_ahtt.py", True)

    if os.path.isfile(aggregate_submit):
        syscall('condor_submit {agg}'.format(agg = aggregate_submit), False)
        syscall('rm {agg}'.format(agg = aggregate_submit), False)
