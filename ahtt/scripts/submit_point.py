#!/usr/bin/env python
# submits single_point_ahtt jobs
# itype=source; idir=/nfs/dust/cms/user/afiqaize/cms/bpark_nano_200218/cmssw_1103_analysis/src/fwk/smooth/template_ULFR2/; for tag in 2D 3D-33; do ./../scripts/submit_point.py --signal "${idir}/sig_${itype}_${tag}.root" --background "${idir}/bkg_${itype}_${tag}.root" --sushi-kfactor --lnN-under-threshold --use-pseudodata --year '2016pre,2016post,2017,2018' --channel 'll' --tag ${tag}_UL_ll --drop ColorRec,UEtune; done

from argparse import ArgumentParser
import os
import sys
import glob

from make_datacard import syscall

condordir = '/nfs/dust/cms/user/afiqaize/cms/sft/condor/'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "desired signal point to run, comma separated", default = "", required = False)
    parser.add_argument("--mode", help = "combine mode to run, comma separated", default = "datacard,validate", required = False)

    parser.add_argument("--signal", help = "signal filename. comma separated", default = "", required = False)
    parser.add_argument("--background", help = "data/background. comma separated", default = "", required = False)
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

    parities = ["A", "H"]
    masses = ("m365", "m400", "m500", "m600", "m800", "m1000")
    widths = ("w2p5", "w5p0", "w10p0", "w25p0")
    points = []

    if args.point == "":
        for parity in parities:
            for mass in masses:
                for width in widths:
                    if width == "w5p0" and mass != "m400":
                        continue

                    points.append(parity + "_" + mass + "_" + width)
    elif args.point.startswith('m'):
        for parity in parities:
            for width in widths:
                if width == "w5p0" and args.point != "m400":
                    continue

                points.append(parity + "_" + args.point + "_" + width)
    elif args.point.startswith('w'):
        for parity in parities:
            for mass in masses:
                if args.point == "w5p0" and mass != "m400":
                    continue

                points.append(parity + "_" + mass + "_" + args.point)
    else:
        points = args.point.strip().split(',')

    if args.injectsignal != "":
        args.pseudodata = True

    if args.jobtime != "":
        args.jobtime = "-t " + args.jobtime

    rundc = "datacard" in args.mode or "workspace" in args.mode

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

        job_name = "single_point_" + pnt + args.tag + "_" + "_".join(args.mode.strip().split(","))
        logs = glob.glob(pnt + args.tag + "/" + job_name + ".o*")
        if len(logs) > 0:
            continue

        job_arg = '"--point {pnt} --mode {mmm} {sus} {psd} {inj} {tag} {drp} {kee} {sig} {bkg} {cha} {yyy} {thr} {lns} {shp} {mcs} {asm} {one} {ims} {gvl} {com}"'.format(
            pnt = pnt,
            mmm = args.mode,
            sus = "--sushi-kfactor" if args.kfactor else "",
            psd = "--use-pseudodata" if args.pseudodata else "",
            inj = "--inject-signal " + args.injectsignal if args.injectsignal != "" else "",
            tag = "--tag " + args.tag if args.tag != "" else "",
            drp = "--drop '" + args.drop + "'" if args.drop != "" else "",
            kee = "--keep '" + args.keep + "'" if args.keep != "" else "",
            sig = "--signal " + args.signal,
            bkg = "--background " + args.background,
            cha = "--channel " + args.channel,
            yyy = "--year " + args.year,
            thr = "--threshold " + str(args.threshold),
            lns = "--lnN-under-threshold" if args.lnNsmall else "",
            shp = "--use-shape-always" if args.alwaysshape else "",
            mcs = "--no-mc-stats" if not args.mcstat else "",
            asm = "--unblind" if not args.asimov else "",
            one = "--one-poi" if args.onepoi else "",
            ims = "--impact-sb" if args.impactsb else "",
            gvl = "--g-value " + str(args.fixg),
            com = "--compress" if rundc else ""
        )

        syscall("{csub} -s {cpar} -w {crun} -n {name} -e {executable} -a {job_arg} {job_time} {tmp} {job}".format(
            csub = condordir + "condorSubmit.sh",
            cpar = condordir + "condorParam.txt",
            crun = condordir + "condorRun.sh",
            name = job_name,
            executable = scriptdir + "/single_point_ahtt.py",
            job_arg = job_arg,
            job_time = args.jobtime,
            tmp = "--run-in-tmp" if rundc else "",
            job = "" if rundc else "-l $(readlink -f " + pnt + args.tag + ")"
        ))
