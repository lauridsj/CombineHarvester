#!/usr/bin/env python
# utilities containing functions used throughout - root file

import math

from ROOT import TFile, gDirectory, TH1, TH1D, gErrorIgnoreLevel, kBreak
TH1.AddDirectory(False)
TH1.SetDefaultSumw2(True)
gErrorIgnoreLevel = kBreak

from utilspy import right_now, index_1n, index_n1

def flat_reldev_wrt_nominal(varied, nominal, offset):
    for ii in range(1, nominal.GetNbinsX() + 1):
        nn = nominal.GetBinContent(ii)
        varied.SetBinContent(ii, nn * (1. + offset))

def scale(histogram, factor, epsilon = 1e-3):
    if abs(factor - 1.) < epsilon:
        return

    for ii in range(1, histogram.GetNbinsX() + 1):
        histogram.SetBinContent(ii, histogram.GetBinContent(ii) * factor)
        histogram.SetBinError(ii, histogram.GetBinError(ii) * abs(factor))

def zero_out(histogram, indices = None):
    for ii in range(1, histogram.GetNbinsX() + 1):
        if (indices is None and histogram.GetBinContent(ii) < 0.) or (indices is not None and ii in indices):
            histogram.SetBinContent(ii, 0.)
            histogram.SetBinError(ii, 0.)

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
    histogram.SetName(hname + "_" + right_now())

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

def add_scaled_nuisance(varied, nominal, original, factor):
    added = varied.Clone(varied.GetName() + "_" + right_now())
    added.Add(original, -1.)
    scale(added, factor)
    added.Add(nominal)
    return added

def apply_relative_nuisance(varied, nominal, target):
    applied = varied.Clone(varied.GetName() + "_" + right_now())
    applied.Add(nominal, -1.)
    applied.Divide(nominal)
    applied.Multiply(target)
    applied.Add(target)
    return applied

def chop_up(varied, nominal, indices):
    chopped = varied.Clone(varied.GetName() + "_" + right_now())
    for ii in range(1, chopped.GetNbinsX() + 1):
        content = varied.GetBinContent(ii) if ii in indices else nominal.GetBinContent(ii)
        error = varied.GetBinError(ii) if ii in indices else nominal.GetBinError(ii)
        chopped.SetBinContent(ii, content)
        chopped.SetBinError(ii, error)
    return chopped

def get_nbin(fname, channel, year):
    hfile = TFile.Open(fname, "read")
    hfile.cd(channel + "_" + year)
    keys = gDirectory.GetListOfKeys()
    histogram = keys[0].ReadObj()
    nbin = histogram.GetNbinsX()
    hfile.Close()
    return nbin

def original_nominal_impl(hist = None, directory = None, process = None):
    # for saving the original nominal templates before manipulations (but after kfactors for signal)
    # to be used in some manipulations later
    if not hasattr(original_nominal_impl, "content"):
        original_nominal_impl.content = {}

    if directory is not None and process is not None:
        if hist is not None:
            if directory not in original_nominal_impl.content:
                original_nominal_impl.content[directory] = {}
            if process not in original_nominal_impl.content[directory]:
                original_nominal_impl.content[directory][process] = hist.Clone(hist.GetName() + "_original_no_bootleg_frfr")
                original_nominal_impl.content[directory][process].SetDirectory(0)
        else:
            hname = original_nominal_impl.content[directory][process].GetName().replace("_original_no_bootleg_frfr", "_") + right_now()
            return original_nominal_impl.content[directory][process].Clone(hname)

    return None

def add_original_nominal(hist, directory, process):
    return original_nominal_impl(hist, directory, process)

def read_original_nominal(directory, process):
    return original_nominal_impl(None, directory, process)
