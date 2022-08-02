#!/usr/bin/env python
# utilities containing functions to be imported
# as dumping everything into make_datacard is becoming unreadable

import os
import sys
import math

import glob
from datetime import datetime
from collections import OrderedDict

from ROOT import TFile, gDirectory, TH1, TH1D
TH1.AddDirectory(False)
TH1.SetDefaultSumw2(True)

def syscall(cmd, verbose = True, nothrow = False):
    if verbose:
        print ("Executing: %s" % cmd)
        sys.stdout.flush()
    retval = os.system(cmd)
    if not nothrow and retval != 0:
        raise RuntimeError("Command failed!")

def flat_reldev_wrt_nominal(varied, nominal, offset):
    for ii in range(1, nominal.GetNbinsX() + 1):
        nn = nominal.GetBinContent(ii)
        varied.SetBinContent(ii, nn * (1. + offset))

def scale(histogram, factor):
    for ii in range(1, histogram.GetNbinsX() + 1):
        histogram.SetBinContent(ii, histogram.GetBinContent(ii) * factor)
        histogram.SetBinError(ii, histogram.GetBinError(ii) * abs(factor))

def zero_out(histogram):
    for ii in range(1, histogram.GetNbinsX() + 1):
        if histogram.GetBinContent(ii) < 0.:
            histogram.SetBinContent(ii, 0.)
            histogram.SetBinError(ii, 0.)

# translate nD bin index to unrolled 1D
def index_n1(idxn, nbins):
    idx1 = idxn[0]
    for ii in range(1, len(idxn)):
        multiplier = 1
        for jj in range(ii - 1, -1, -1):
            multiplier *= nbins[jj]

        idx1 += idxn[ii] * multiplier

    return idx1

# and the inverse operation
def index_1d_1n(idx1, dim, nbins):
    multiplier = 1
    for dd in range(dim - 1, -1, -1):
        multiplier *= nbins[dd]

    return (idx1 // multiplier) % nbins[dim]

# as above, but over all dimensions in a go
def index_1n(idx1, nbins):
    idxn = [-1] * len(nbins)
    for iv in range(len(nbins)):
        idxn[iv] = index_1d_1n(idx1, iv, nbins)

    return idxn

def project(histogram, rule):
    src_nbin1 = histogram.GetNbinsX()
    src_nbinn = [int(bb) for bb in rule[0].split(',')]

    if len(src_nbinn) <= 1:
        print "aint nuthin to project for 1D histograms innit???"
        return histogram

    if reduce(lambda a, b: a * b, src_nbinn, 1) != src_nbin1:
        print "number of bins given in rule doesnt match the histogram. skipping projection."
        return histogram

    target = sorted([int(tt) for tt in rule[1].split(',')])
    if any([tt >= len(src_nbinn) or tt < 0 for tt in target]) or len(target) >= len(src_nbinn) or len(target) <= 0:
        print "target dimension indices not compatible with assumed dimensionality. skipping projection."
        return histogram

    tgt_nbinn = [src_nbinn[tt] for tt in target]
    tgt_nbin1 = reduce(lambda a, b: a * b, tgt_nbinn, 1)

    hname = histogram.GetName()
    histogram.SetName(hname + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    hist = TH1D(hname, histogram.GetTitle(), tgt_nbin1, 0., tgt_nbin1)
    for is1 in range(src_nbin1):
        isn = index_1n(is1, src_nbinn)
        src_content = histogram.GetBinContent(is1 + 1)
        src_error = histogram.GetBinError(is1 + 1)

        itn = [isn[ii] for ii in range(len(src_nbinn)) if ii in target]
        it1 = index_n1(itn, tgt_nbinn)

        tgt_content = hist.GetBinContent(it1 + 1)
        tgt_error = hist.GetBinError(it1 + 1)

        hist.SetBinContent(it1 + 1, tgt_content + src_content)
        hist.SetBinError(it1 + 1, math.sqrt(tgt_error**2 + src_error**2))

    return hist

def chunks(lst, npart):
    if npart > math.ceil(float(len(lst)) / 2) or npart < 1:
        print 'chunks called with a invalid npart. setting it to 2.'
        npart = 2

    nf = float(len(lst)) / npart
    nc = int(math.ceil(nf))
    ni = len(lst) / npart
    ii = 0
    result = []
    if nf - ni > 0.5:
        ni, nc = nc, ni
    result.append(lst[ii:ii + nc])
    ii += nc
    for i in xrange(ii, len(lst), ni):
        result.append(lst[i:i + ni])
    return result

condordir = '/nfs/dust/cms/user/afiqaize/cms/sft/condor/'

def aggregate_submit():
    return 'conSub_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.txt'

def submit_job(job_agg, job_name, job_arg, job_time, job_cpu, job_mem, job_dir, executable, runtmp = False, runlocal = False):
    if not hasattr(submit_job, "firstprint"):
        submit_job.firstprint = True

    if runlocal:
        syscall('{executable} {job_arg}'.format(executable = executable, job_arg = job_arg), True)
    else:
        syscall('{csub} -s {cpar} -w {crun} -n {name} -e {executable} -a "{job_arg}" {job_time} {job_cpu} {tmp} {job_dir} --debug'.format(
            csub = condordir + "condorSubmit.sh",
            cpar = condordir + "condorParam.txt",
            crun = condordir + "condorRun.sh",
            name = job_name,
            executable = executable,
            job_arg = job_arg,
            job_time = job_time,
            job_cpu = "-p " + str(job_cpu) if job_cpu > 1 else "",
            job_mem = "-m " + job_mem if job_mem != "" else "",
            tmp = "--run-in-tmp" if runtmp else "",
            job_dir = job_dir
        ), submit_job.firstprint)
        submit_job.firstprint = False

        if not os.path.isfile(job_agg):
            syscall('cp {name} {agg} && rm {name}'.format(name = 'conSub_' + job_name + '.txt', agg = job_agg), False)
        else:
            syscall("echo >> {agg} && grep -F -x -v -f {agg} {name} >> {agg} && echo 'queue' >> {agg} && rm {name}".format(
                name = 'conSub_' + job_name + '.txt',
                agg = job_agg), False)
