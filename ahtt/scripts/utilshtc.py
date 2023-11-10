#!/usr/bin/env python
# utilities containing functions used throughout - python/misc file

import os
import sys

from datetime import datetime
from numpy import random as rng

from utilspy import syscall, right_now
from utilslab import cluster, condorrun, input_bkg
from desalinator import clamp_with_quote


def make_submission_script_header():
    script = "Job_Proc_ID = $(Process) + 1 \n"
    script += "executable = {script}\n".format(script = condorrun)
    script += "notification = error\n"
    script += 'requirements = (OpSysAndVer == "CentOS7")\n'

    if cluster == "naf":
        script += "universe = vanilla\n"
        script += "getenv = true\n"
        script += 'environment = "LD_LIB_PATH={ldpath} JOB_PROC_ID=$INT(Job_Proc_ID)"\n'.format(ldpath = os.getenv('LD_LIBRARY_PATH'))

    elif cluster == "lxplus":
        script += 'environment = "cmssw_base={cmssw} JOB_PROC_ID=$INT(Job_Proc_ID)"\n'.format(cmssw = os.getenv('CMSSW_BASE'))

        # Afiq's Special Treatment
        if os.getlogin() == 'afiqaize':
            grp = 'group_u_CMST3.all' if rng.binomial(1, 0.9) else 'group_u_CMS.u_zh.users'
            script += '+AccountingGroup = "{grp}"\n'.format(grp = grp)

    script += "\n"
    return script
    
def make_submission_script_single(name, directory, executable, arguments, cpus = None, runtime = None, memory = None, runtmp = False, writelog = True):
    script = '''
batch_name = {name}
arguments = "{executable} {args}"
'''

    script = script.format(name = name, directory = directory, executable = executable, args = ' '.join(arguments.split()))
    if writelog:
        script += "output = {directory}/{name}.o$(Cluster).$INT(Job_Proc_ID)\n".format(name = name, directory = directory)
        script += "error = {directory}/{name}.o$(Cluster).$INT(Job_Proc_ID)\n".format(name = name, directory = directory)
    else:
        script += "output = /dev/null\n"
        script += "error = /dev/null\n"

    if cpus is not None and cpus != "" and cpus > 1:
        script += "request_cpus = {cpus}\n".format(cpus = cpus)

    if memory is not None and memory != "":
        script += "RequestMemory = {memory}\n".format(memory = memory)

    if runtime is not None and runtime != "":
        script += "+{req}Runtime = {runtime}\n".format(req = "Max" if cluster == "lxplus" else "Request", runtime = runtime)

    if runtmp or cluster == "lxplus":
        script += "should_transfer_files = YES\n"
        script += "when_to_transfer_output = ON_EXIT_OR_EVICT\n"
    else:
        script += "initialdir = {cwd}\n".format(cwd = os.getcwd())

    if cluster == "lxplus":
        script += 'transfer_output_files = tmp/\n'

    script += "queue\n\n"
    return script

# Array to store all buffered submission scripts
current_submissions = []
max_jobs_per_submit = 4000

def aggregate_submit():
    return 'conSub_' + right_now() + '.txt'

def submit_job(job_agg, job_name, job_arg, job_time, job_cpu, job_mem, job_dir, executable, runtmp = False, runlocal = False, writelog = True):
    global current_submissions
    if not hasattr(submit_job, "firstjob"):
        submit_job.firstjob = True

    # figure out symlinks (similar to $(readlink))
    job_dir = os.path.realpath(job_dir)

    if runlocal:
        lname = ""
        if writelog:
            lname = "{log}.olocal.1".format(log = job_dir + '/' + job_name)
            syscall("touch {log}".format(log = lname), False)

        syscall('echo "Job execution starts at {atm}"{log}'.format(atm = datetime.now(), log = " |& tee -a " + lname if lname != "" else ""), False)
        syscall('{executable} {job_arg}{log}'.format(executable = executable, job_arg = job_arg, log = " |& tee -a " + lname if lname != "" else ""), True)
        syscall('echo "Job execution ends at {atm}"{log}'.format(atm = datetime.now(), log = " |& tee -a " + lname if lname != "" else ""), False)
    else:
        sub_script = make_submission_script_single(
            name = job_name,
            directory = job_dir,
            executable = executable,
            arguments = job_arg,
            cpus = job_cpu,
            runtime = job_time,
            memory = job_mem,
            runtmp = runtmp,
            writelog = writelog
        )

        if submit_job.firstjob:
            print("Submission script:")
            print(make_submission_script_header() + sub_script)
            sys.stdout.flush()
            submit_job.firstjob = False

        current_submissions.append(sub_script)

        if len(current_submissions) >= max_jobs_per_submit:
            flush_jobs(job_agg)

def flush_jobs(job_agg):
    global current_submissions
    if len(current_submissions) > 0:
        print("Submitting {njobs} jobs".format(njobs = len(current_submissions)))
        header = make_submission_script_header()
        script = header + "\n" + "\n".join(current_submissions)
        with open(job_agg, "w") as f:
            f.write(script)

        syscall("condor_submit {job_agg}".format(job_agg = job_agg), False)
        os.remove(job_agg)

        current_submissions = []
    else:
        print("Nothing to submit.")

def common_job(args):
    argstr = (" {sus} {inj} {ass} {exc} {tag} {drp} {kee} {bkg} {cha} {yyy} {thr} {lns} {shp} {mcs} {rpr} {msk} {igb} {prj} "
              "{cho} {rep} {fst} {hes} {kbf} {dws} {fr0} {frp} {rsd} {asm} {com} {dbg} {ext} {otg} {bsd}").format(
                  sus = "--sushi-kfactor" if args.kfactor else "",
                  inj = clamp_with_quote(string = args.inject, prefix = '--inject-signal '),
                  ass = clamp_with_quote(string = args.assignal, prefix = '--as-signal '),
                  exc = clamp_with_quote(string = args.excludeproc, prefix = '--exclude-process '),
                  tag = clamp_with_quote(string = args.tag, prefix = '--tag '),
                  drp = clamp_with_quote(string = args.drop, prefix = '--drop '),
                  kee = clamp_with_quote(string = args.keep, prefix = '--keep '),
                  bkg = "--background " + input_bkg(args.background, args.channel) if args.rundc else "",
                  cha = "--channel " + args.channel,
                  yyy = "--year " + args.year,
                  thr = clamp_with_quote(string = args.threshold, prefix = '--threshold '),
                  lns = "--lnN-under-threshold" if args.lnNsmall else "",
                  shp = "--use-shape-always" if args.alwaysshape else "",
                  mcs = "--no-mc-stats" if not args.mcstat else "",
                  rpr = clamp_with_quote(string = args.rateparam, prefix = '--float-rate '),
                  msk = clamp_with_quote(string = args.mask, prefix = '--mask '),
                  igb = clamp_with_quote(string = args.ignorebin, prefix = '--ignore-bin '),
                  prj = clamp_with_quote(string = args.projection, prefix = '--projection '),
                  cho = clamp_with_quote(string = args.chop, prefix = '--chop-up '),
                  rep = clamp_with_quote(string = args.repnom, prefix = '--replace-nominal '),
                  fst = "--fit-strategy {fst}".format(fst = args.fitstrat) if args.fitstrat > -1 else "",
                  hes = "--use-hesse" if args.usehesse else "",
                  kbf = "--redo-best-fit" if not args.keepbest else "",
                  dws = "--default-workspace" if args.defaultwsp else "",
                  fr0 = clamp_with_quote(string = args.frzzero, prefix = '--freeze-zero '),
                  frp = clamp_with_quote(string = args.frzpost, prefix = '--freeze-post '),
                  rsd = clamp_with_quote(string = args.seed, prefix = '--seed '),
                  asm = "--unblind" if not args.asimov else "",
                  com = "--compress" if args.rundc else "",
                  dbg = "--experimental" if args.experimental else "",
                  ext = clamp_with_quote(
                      string = args.extopt,
                      prefix = '--extra-option{s}'.format(s = '=' if args.extopt.startswith('-') else ' ')
                  ),
                  otg = clamp_with_quote(string = args.otag, prefix = '--output-tag '),
                  bsd = "" if args.rundc else "--base-directory " + os.path.abspath("./")
              )
    return argstr
