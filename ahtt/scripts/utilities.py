#!/usr/bin/env python
# utilities containing functions to be imported
# as dumping everything into make_datacard is becoming unreadable

import os
import sys
import math
import fnmatch

from datetime import datetime
from collections import OrderedDict

from ROOT import TFile, gDirectory, TH1, TH1D
TH1.AddDirectory(False)
TH1.SetDefaultSumw2(True)

min_g = 0.
max_g = 3.
condordir = '/nfs/dust/cms/user/afiqaize/cms/sft/condor/'

def syscall(cmd, verbose = True, nothrow = False):
    if verbose:
        print ("Executing: %s" % cmd)
        sys.stdout.flush()
    retval = os.system(cmd)
    if not nothrow and retval != 0:
        raise RuntimeError("Command failed with exit code {ret}!".format(ret = retval))

def get_point(sigpnt):
    pnt = sigpnt.split('_')
    return (pnt[0][0], float(pnt[1][1:]), float(pnt[2][1:].replace('p', '.')))

def stringify(gtuple):
    return str(gtuple)[1: -1]

def tuplize(gstring):
    return tuple([float(gg) for gg in gstring.replace(" ", "").split(",")])

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

def get_nbin(fname, channel, year):
    hfile = TFile.Open(fname, "read")
    hfile.cd(channel + "_" + year)
    keys = gDirectory.GetListOfKeys()
    histogram = keys[0].ReadObj()
    nbin = histogram.GetNbinsX()
    hfile.Close()
    return nbin

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

def input_bkg(background, channels):
    # far be it for us to get in the way of those who know what they are doing
    if background != "":
        return background

    backgrounds = []
    if any(cc in channels for cc in ["ee", "em", "mm"]):
        backgrounds.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/bkg_ll_3D-33_rate_mtuX_pca.root")
    if any(cc in channels for cc in ["e3j", "e4pj", "m3j", "m4pj"]):
        backgrounds.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/templates_lj_bkg_rate_mtuX_pca.root")

    return ','.join(backgrounds)

def input_sig(signal, points, injects, channels, years):
    # far be it for us to get in the way of those who know what they are doing
    if signal != "":
        return signal

    signals = []
    if any(cc in channels for cc in ["e3j", "e4pj", "m3j", "m4pj"]):
        if "2016pre" in years:
            signals.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/templates_lj_sig_2016pre.root")
        if "2016post" in years:
            signals.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/templates_lj_sig_2016post.root")
        if "2017" in years:
            signals.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/templates_lj_sig_2017.root")
        if "2018" in years:
            signals.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/templates_lj_sig_2018.root")

    if any(cc in channels for cc in ["ee", "em", "mm"]):
        if any([im in points or im in injects for im in ["_m3", "_m1000"]]):
            signals.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/sig_ll_3D-33_m3xx_and_m1000.root")
        for im in ["_m4", "_m5", "_m6", "_m7", "_m8", "_m9"]:
            if im in points or im in injects:
                signals.append("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/templates_ULFR2/sig_ll_3D-33" + im + "xx.root")

    return ','.join(signals)

def aggregate_submit():
    return 'conSub_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.txt'

def submit_job(job_agg, job_name, job_arg, job_time, job_cpu, job_mem, job_dir, executable, runtmp = False, runlocal = False):
    if not hasattr(submit_job, "firstprint"):
        submit_job.firstprint = True

    if runlocal:
        lname = "{log}.olocal.1".format(log = job_dir + '/' + job_name)
        syscall("touch {log}".format(log = lname), False)
        syscall('echo "Job execution starts at {atm}" |& tee -a {log}'.format(atm = datetime.now(), log = lname), False)
        syscall('{executable} {job_arg} |& tee -a {log}'.format(executable = executable, job_arg = job_arg, log = lname), True)
        syscall('echo "Job execution ends at {atm}" |& tee -a {log}'.format(atm = datetime.now(), log = lname), False)
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
            job_dir = "-l " + job_dir
        ), submit_job.firstprint)
        submit_job.firstprint = False

        if not os.path.isfile(job_agg):
            syscall('cp {name} {agg} && rm {name}'.format(name = 'conSub_' + job_name + '.txt', agg = job_agg), False)
        else:
            syscall("echo >> {agg} && grep -F -x -v -f {agg} {name} >> {agg} && echo 'queue' >> {agg} && rm {name}".format(
                name = 'conSub_' + job_name + '.txt',
                agg = job_agg), False)

def make_best_fit(dcdir, card, point, asimov, mcstat, strategy, poi_range, set_freeze, extopt = "", masks = []):
    fname = point + "_best_fit_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    setpar = set_freeze[0]
    frzpar = set_freeze[1]

    syscall("combineTool.py -v -1 -M MultiDimFit -d {dcd} -n _{bff} {stp} {frz} {stg} {prg} {asm} {mcs} {wsp} {ext}".format(
        dcd = dcdir + card,
        bff = fname,
        stp = "--setParameters '" + ",".join(setpar + masks) + "'" if len(setpar + masks) > 0 else "",
        frz = "--freezeParameters '" + ",".join(frzpar) + "'" if len(frzpar) > 0 else "",
        stg = strategy,
        prg = poi_range,
        asm = "-t -1" if asimov else "",
        mcs = "--X-rtd MINIMIZER_analytic" if mcstat else "",
        wsp = "--saveSpecifiedNuis=all",
        ext = extopt
    ))
    syscall("mv higgsCombine*{bff}.MultiDimFit*.root {dcd}{bff}.root".format(dcd = dcdir, bff = fname), False)
    return "{dcd}{bff}.root".format(dcd = dcdir, bff = fname)

def read_nuisance(dname, points, qexp_eq_m1 = True):
    dfile = TFile.Open(dname)
    dtree = dfile.Get("limit")

    skip = ["r", "g", "r1", "r2", "g1", "g2", "deltaNLL", "quantileExpected",
            "limit", "limitErr", "mh", "syst",
            "iToy", "iSeed", "iChannel", "t_cpu", "t_real"]

    nuisances = [bb.GetName() for bb in dtree.GetListOfBranches()]
    hasbb = False

    setpar = []
    frzpar = []

    for i in dtree:
        if (dtree.quantileExpected != -1. and qexp_eq_m1) or (dtree.quantileExpected == -1. and not qexp_eq_m1):
            continue

        for nn in nuisances:
            if nn in skip:
                continue

            if "prop_bin" not in nn:
                frzpar.append(nn)
            else:
                hasbb = True

            vv = round(getattr(dtree, nn), 2)
            if abs(vv) > 0.01:
                setpar.append(nn + "=" + str(vv))

        if len(frzpar) > 0:
            break

    if hasbb:
        frzpar.append("rgx{prop_bin.*}")

    return [setpar, frzpar]

def starting_nuisance(point, frz_bb_zero = True, frz_bb_post = False, frz_nuisance_post = False, best_fit_file = ""):
    if frz_bb_zero:
        return [["rgx{prop_bin.*}=0"], ["rgx{prop_bin.*}"]]
    elif frz_bb_post or frz_nuisance_post:
        if best_fit_file == "":
            raise RuntimeError("postfit bb/nuisance freezing is requested, but no best fit file is provided!!!")

        setpar, frzpar = read_nuisance(best_fit_file, point, True)

        if not frz_nuisance_post:
            setpar = [nn for nn in setpar if "prop_bin" in nn]
            frzpar = [nn for nn in frzpar if "prop_bin" in nn]

        return [setpar, frzpar]

    return [[], []]

def elementwise_add(list_of_lists):
    if len(list_of_lists) < 1 or any(len(ll) < 1 or len(ll) != len(list_of_lists[0]) for ll in list_of_lists):
        raise RuntimeError("this method assumes that the argument is a list of lists of nonzero equal lengths!!!")

    result = list_of_lists[0]
    for ll in range(1, len(list_of_lists)):
        for rr in range(len(result)):
            result[rr] += list_of_lists[ll][rr]

    return result

def fit_strategy(strat):
    return "--cminPreScan --cminDefaultMinimizerAlgo Migrad --cminDefaultMinimizerStrategy {ss} --cminFallbackAlgo Minuit2,Simplex,{ss}".format(ss = strat)

def recursive_glob(base_directory, pattern):
    # https://stackoverflow.com/a/2186639
    results = []
    for base, dirs, files in os.walk(base_directory):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results

def make_datacard_with_args(scriptdir, args):
    syscall("{scr}/make_datacard.py --signal {sig} --background {bkg} --point {pnt} --channel {ch} --year {yr} "
            "{psd} {inj} {tag} {drp} {kee} {kfc} {thr} {lns} {shp} {mcs} {rpr} {prj} {rsd}".format(
                scr = scriptdir,
                pnt = ','.join(args.point),
                sig = args.signal,
                bkg = args.background,
                ch = args.channel,
                yr = args.year,
                psd = "--use-pseudodata" if args.asimov else "",
                inj = "--inject-signal " + args.inject if args.inject != "" else "",
                tag = "--tag " + args.tag if args.tag != "" else "",
                drp = "--drop '" + args.drop + "'" if args.drop != "" else "",
                kee = "--keep '" + args.keep + "'" if args.keep != "" else "",
                kfc = "--sushi-kfactor" if args.kfactor else "",
                thr = "--threshold " + args.threshold if args.threshold != "" else "",
                lns = "--lnN-under-threshold" if args.lnNsmall else "",
                shp = "--use-shape-always" if args.alwaysshape else "",
                mcs = "--no-mc-stats" if not args.mcstat else "",
                rpr = "--float-rate '" + args.rateparam + "'" if args.rateparam != "" else "",
                prj = "--projection '" + args.projection + "'" if args.projection != "" else "",
                rsd = "--seed " + args.seed if args.seed != "" else ""
            ))
