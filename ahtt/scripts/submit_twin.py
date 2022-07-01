#!/usr/bin/env python
# submit the two-point jobs and compiles its results


from argparse import ArgumentParser
import os
import sys
import glob
import subprocess
import numpy as np

from utilities import syscall

max_g = 3.

condordir = '/nfs/dust/cms/user/afiqaize/cms/sft/condor/'
def submit_twin_job(job_arg, job_time, job_dir, script_dir):
    syscall("{csub} -s {cpar} -w {crun} -n {name} -e {executable} -a {job_arg} {job_time} {tmp} {job}".format(
        csub = condordir + "condorSubmit.sh",
        cpar = condordir + "condorParam.txt",
        crun = condordir + "condorRun.sh",
        name = job_name,
        executable = script_dir + "/twin_point_ahtt.py",
        job_arg = job_arg,
        job_time = job_time,
        tmp = "--run-in-tmp",
        job = job_dir
    ))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "desired pairs of signal points to run on, comma (between points) and semicolon (between pairs) separated", default = "", required = True)
    parser.add_argument("--mode", help = "combine mode to run, comma separated", default = "datacard,validate", required = False)

    parser.add_argument("--signal", help = "signal filenames. comma separated", default = "", required = False)
    parser.add_argument("--background", help = "data/background filenames. comma separated",
                        default = "/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/bkg_templates_3D-33.root", required = False)
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
                        default = "0., 0.", dest = "fcgvl", required = False)
    parser.add_argument("--fc-expect", help = "expected scenarios to assume in the scan. "
                        "exp-b -> g1 = g2 = 0; exp-s -> g1 = g2 = 1; exp-01 -> g1 = 0, g2 = 1; exp-10 -> g1 = 1, g2 = 0",
                        default = "exp-b", dest = "fcexp", required = False)
    parser.add_argument("--fc-max-sigma", help = "max sigma contour that is considered important",
                        default = 2, dest = "fcsigma", required = False, type = int)
    parser.add_argument("--fc-n-toy", help = "number of toys to throw per FC grid scan",
                        default = 100, dest = "fctoy", required = False, type = int)
    parser.add_argument("--fc-save-toy", help = "save toys thrown in the FC grid scan", dest = "fcsave", action = "store_true", required = False)
    parser.add_argument("--fc-idx", help = "index to append to FC grid scan",
                        default = -1, dest = "fcidx", required = False, type = int)

    parser.add_argument("--job-time", help = "time to assign to each job", default = "", dest = "jobtime", required = False)
    args = parser.parse_args()
    if (args.tag != "" and not args.tag.startswith("_")):
        args.tag = "_" + args.tag
    scriptdir = os.path.dirname(os.path.abspath(__file__))

    # FIXME something to allow --point to be non-default
    pairs = args.point.replace(" ", "").split(';')

    if args.injectsignal != "":
        args.pseudodata = True

    if args.jobtime != "":
        args.jobtime = "-t " + args.jobtime

    rundc = "datacard" in args.mode or "workspace" in args.mode
    runfc = "fc-scan" in args.mode or "contour" in args.mode

    for pair in pairs:
        points = pair.split(',')
        pstr = '__'.join(points)

        for pnt in points:
            signals = []
            if args.signal == "":
                if "_m3" in pnt or "_m1000" in pnt or "_m3" in args.injectsignal or "_m1000" in args.injectsignal:
                    if any(cc in args.channel for cc in ["ee", "em", "mm"]):
                        signals.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/sig_templates_3D-33_m3xx_m1000.root")
                for im in ["_m4", "_m5", "_m6", "_m7", "_m8", "_m9"]:
                    if im in pnt or im in args.injectsignal:
                        if any(cc in args.channel for cc in ["ee", "em", "mm"]):
                            signals.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/sig_templates_3D-33" + im + "xx.root")
            signal = ','.join(signals)
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

        # FIXME finer-grained check for FC
        job_name = "twin_point_" + pstr + args.tag + "_" + "_".join(args.mode.replace(" ", "").split(","))
        #logs = glob.glob(pstr + args.tag + "/" + job_name + ".o*")

        #if len(logs) > 0:
        #    continue

        job_arg = ('"--point {pnt} --mode {mmm} {sus} {psd} {inj} {tag} {drp} {kee} {sig} {bkg} {cha} {yyy} {thr} {lns}'
                   '{shp} {mcs} {prj} {asm} {com} {bsd}"').format(
            pnt = pair,
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
            asm = "--unblind" if not args.asimov else "",
            com = "--compress" if rundc else "",
            bsd = "" if rundc else "--base-directory " + os.path.abspath("./")
        )

        if runfc:
            gvalues = list(np.linspace(0., max_g, num = 13))
            jfile = glob.glob(pstr + args.tag + "/" + "fc_grid_{exp}.json".format(exp = args.fcexp))

            if len(jfile) == 0:
                for ig1 in gvalues:
                    for ig2 in gvalues:
                        rfile = glob.glob(pstr + args.tag + "/" + "fc_grid_{snm}.root".format(snm = "pnt_g1_" + str(ig1) + "_g2_" + str(ig2) + "_" + args.fcexp))
                        if len(jfile) == 0:
                            jarg = job_arg
                            jarg += "{gvl} {exp} {sig} {toy} {sav}".format(
                                gvl = str(ig1) + "," + str(ig2),
                                exp = args.fcexp,
                                sig = args.fcsigma,
                                toy = args.fctoy,
                                sav = "--fc-save-toy" if args.fcsave else ""
                            )

                            submit_twin_job(jarg, args.jobtime, "" if rundc else "-l $(readlink -f " + pstr + args.tag + ")", scriptdir)

            # FIXME cumulative toys, compilation, NLO submission, ...
        else:
            submit_twin_job(job_arg, args.jobtime, "" if rundc else "-l $(readlink -f " + pstr + args.tag + ")", scriptdir)
