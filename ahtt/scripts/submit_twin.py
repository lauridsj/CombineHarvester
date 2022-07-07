#!/usr/bin/env python
# submit the two-point jobs and compiles its results

from argparse import ArgumentParser
import os
import sys
import glob
import subprocess
import numpy as np

from utilities import syscall

min_g = 0.
max_g = 3.

condordir = '/nfs/dust/cms/user/afiqaize/cms/sft/condor/'
aggregate_submit = "conSub_aggregate.txt"

def generate_g_grid(fcgvl, ggrid = []):
    if fcgvl != "-1, -1":
        fcgvl = fcgvl.replace(" ", "").split(',')
        return [[float(fcgvl[0])], [float(fcgvl[1])]]
    elif len(ggrid) == 1:
        # whatever logic to generate in between points of existing grid
        pass

    return [list(np.linspace(min_g, max_g, num = 13)), list(np.linspace(min_g, max_g, num = 13))]

def submit_twin_job(job_name, job_arg, job_time, job_dir, script_dir):
    if not hasattr(submit_twin_job, "firstprint"):
        submit_twin_job.firstprint = True

    syscall('{csub} -s {cpar} -w {crun} -n {name} -e {executable} -a "{job_arg}" {job_time} {tmp} {job_dir} --debug'.format(
        csub = condordir + "condorSubmit.sh",
        cpar = condordir + "condorParam.txt",
        crun = condordir + "condorRun.sh",
        name = job_name,
        executable = script_dir + "/twin_point_ahtt.py",
        job_arg = job_arg,
        job_time = job_time,
        tmp = "--run-in-tmp",
        job_dir = job_dir
    ), submit_twin_job.firstprint)
    submit_twin_job.firstprint = False

    if not os.path.isfile(aggregate_submit):
        syscall('cp {name} {agg} && rm {name}'.format(name = 'conSub_' + job_name + '.txt', agg = aggregate_submit), False)
    else:
        syscall("echo >> {agg} && grep -F -x -v -f {agg} {name} >> {agg} && echo 'queue' >> {agg} && rm {name}".format(
            name = 'conSub_' + job_name + '.txt',
            agg = aggregate_submit), False)

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
                        default = "-1, -1", dest = "fcgvl", required = False)
    parser.add_argument("--fc-expect", help = "expected scenarios to assume in the scan, if --unblind isn't used\n"
                        "exp-b -> g1 = g2 = 0; exp-s -> g1 = g2 = 1; exp-01 -> g1 = 0, g2 = 1; exp-10 -> g1 = 1, g2 = 0",
                        default = "exp-b", dest = "fcexp", required = False)
    parser.add_argument("--fc-n-toy", help = "number of toys to throw per FC grid scan",
                        default = 200, dest = "fctoy", required = False, type = int)
    parser.add_argument("--fc-delete-data", help = "delete data file instead of returning it as output", dest = "fckeepdat", action = "store_false", required = False)
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

    # clean up before doing anything else, in case someone aborts previous run
    if os.path.isfile(aggregate_submit):
        syscall('rm {agg}'.format(agg = aggregate_submit), False, True)

    for pair in pairs:
        points = pair.split(',')
        pstr = '__'.join(points)

        if args.signal == "":
            for pnt in points:
                signals = []
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

        print signal
        continue

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

        job_arg = ("--point {pnt} --mode {mmm} {sus} {psd} {inj} {tag} {drp} {kee} {sig} {bkg} {cha} {yyy} {thr} {lns}"
                   "{shp} {mcs} {prj} {asm} {com} {bsd}").format(
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
            ggrid = glob.glob(pstr + args.tag + "/" + "fc_scan_xxx.json")
            gvalues = generate_g_grid(args.fcgvl, ggrid)

            for ig1 in gvalues[0]:
                for ig2 in gvalues[1]:
                    scan_name = "pnt_g1_" + str(ig1) + "_g2_" + str(ig2)
                    scan_name += "_" + args.fcexp if args.asimov else "_data"
                    scan_name += "_" + str(args.fcidx) if args.fcidx > -1 else ""

                    rfile = glob.glob(pstr + args.tag + "/" + "fc_scan_{snm}.root".format(snm = scan_name))
                    if len(rfile) == 0:
                        jname = job_name + scan_name.replace("pnt", "")

                        jarg = job_arg
                        jarg += " {gvl} {exp} {toy} {dat} {idx}".format(
                            gvl = "--fc-g-values '" + str(ig1) + "," + str(ig2) + "'",
                            exp = "--fc-expect " + args.fcexp,
                            toy = "--fc-n-toy " + str(args.fctoy) if args.fctoy > 0 else "",
                            dat = "--fc-delete-data " if not args.fckeepdat else "",
                            idx = "--fc-idx " + str(args.fcidx) if args.fcidx > -1 else ""
                        )

                        submit_twin_job(jname, jarg, args.jobtime, "" if rundc else "-l $(readlink -f " + pstr + args.tag + ")", scriptdir)

            # FIXME cumulative toys, compilation, NLO submission, ...
        else:
            submit_twin_job(job_name, job_arg, args.jobtime, "" if rundc else "-l $(readlink -f " + pstr + args.tag + ")", scriptdir)

        if os.path.isfile(aggregate_submit):
            syscall('condor_submit {agg}'.format(agg = aggregate_submit), False)
            syscall('rm {agg}'.format(agg = aggregate_submit), False)
