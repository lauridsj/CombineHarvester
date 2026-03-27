#!/usr/bin/env python3
# original script by jonas ruebenach (desy) @ https://gitlab.cern.ch/jrubenac/ahtt_scripts/-/blob/a1020072d17d6813b55fc6f0c3a382538b542f3e/plot_post_fit.py
# environment: source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_102 x86_64-centos7-gcc11-opt
# updating mpl: python3 -m pip install matplotlib --upgrade
# actually using it: export PYTHONPATH=`python3 -c 'import site; print(site.getusersitepackages())'`:$PYTHONPATH

import os
from itertools import product
import re
from argparse import ArgumentParser
import numpy as np
from functools import reduce
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa:E402
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['figure.max_open_warning'] = False
plt.rcParams["font.size"] = 22.0
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle

import uproot  # noqa:E402
import mplhep as hep  # noqa:E402
import hist  # noqa:E402
from hist import Hist  # noqa:E402
import ROOT
import math
import numba

from utilspy import tuplize
from utilsmath import index_1n, index_n1
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes
from drawings import etat_blurb, channels, years, sm_procs, proc_colors, signal_zorder, binnings, ratiolabels, lumis, hatchstyle, datastyle, genlabels
from drawings import get_poi_values

parser = ArgumentParser()
parser.add_argument("--ifile", help = "input file ie fitdiagnostic results", default = "", required = True)
parser.add_argument("--lower", choices = ["ratio", "diff"], default = "ratio", required = False)
parser.add_argument("--logx", action = "store_true", required = False)
parser.add_argument("--logy", action = "store_true", required = False)
parser.add_argument("--log", action = "store_true", required = False)
parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False)
parser.add_argument("--plot-tag", help = "extra tag to append to plot names", dest = "ptag", default = "", required = False, type = prepend_if_not_empty)
parser.add_argument("--skip-each", help = "skip plotting each channel x year combination", action = "store_false", dest = "each", required = False)
parser.add_argument("--batch", help = "psfromws output containing sums of channel x year combinations to be plotted. give any string to draw only batched prefit.",
                    default = None, dest = "batch", required = False)
parser.add_argument("--read-from-batch", help = "read all plots from --batch, not ifile. mainly for morphed model.", action = "store_true", dest = "readbatch", required = False)
parser.add_argument("--skip-postfit", help = "skip plotting postfit", action = "store_false", dest = "postfit", required = False)
parser.add_argument("--skip-prefit", help = "skip plotting prefit", action = "store_false", dest = "prefit", required = False)
parser.add_argument("--prefit-signal-from", help = "read prefit signal templates from this file instead",
                    default = "", dest = "ipf", required = False)
parser.add_argument("--best-fit-from", help = "read best fit poi from this file instead",
                    default = "", dest = "poi", required = False)
parser.add_argument("--panel", help = "plot upper, lower panel or both", choices=["both", "upper", "lower"], default="both", required = False)
parser.add_argument("--plot-formats", help = "comma-separated list of formats to save the plots in", default = [".png"], dest = "fmt", required = False,
                    type = lambda s: [prepend_if_not_empty(fmt, '.') for fmt in tokenize_to_list(remove_spaces_quotes(s))])
parser.add_argument("--signal-scale", help = "scaling to apply on A/H signal (ie not promoted ones (yet!)) in drawing", default = (1., 1.),
                    dest = "sigscale", required = False, type = lambda s: tuplize(s))
parser.add_argument("--as-signal", help = "comma-separated list of background processes to draw as signal",
                    dest = "assignal", default = "", required = False,
                    type = lambda s: [] if s == "" else sorted(tokenize_to_list(remove_spaces_quotes(s))))
parser.add_argument("--ignore", help = "comma-separated list of background processes to ignore",
                    dest = "ignore", default = "", required = False,
                    type = lambda s: [] if s == "" else sorted(tokenize_to_list(remove_spaces_quotes(s))))
parser.add_argument("--skip-ah", help = "don't draw A/H signal histograms", action = "store_false", dest = "doah", required = False)
parser.add_argument("--panel-labels", help = "put labels on each panel", action = "store_true", dest = "panellabels", required = False)
parser.add_argument("--no-xaxis", help = "put labels on each panel", action = "store_true", dest = "noxaxis", required = False)
parser.add_argument("--cmslabel", help="CMS label", type=str, default=None)
parser.add_argument("--only-res", dest="onlyres", help="Resonance-only mode", action="store_true")
parser.add_argument("--split-bins", help="Split the angle/spin bins", dest="splitbins", action="store_true")
parser.add_argument("--project-to", help = "which variables to project down to, and draw the 1D plots of. implemented only for batch plotting atm.",
                    dest = "project", choices = ["none", "mtt", "mbbll", "chel", "chan"], default = "none", required = False)
parser.add_argument("--mass-cut", help = "comma-separated minmax value, to cut on mtt. must be sorted, otherwise applies no cut. if one value is outside mass range, plots binwise.",
                    dest = "cut", default = "", required = False,
                    type = lambda s: [] if s == "" else tokenize_to_list(remove_spaces_quotes(s), astype = int))
parser.add_argument("--xsec", help = "report toponia as xsec", action = "store_true", dest = "xsec", required = False)
parser.add_argument('--no-total', help = "dont plot the total signal", action="store_false", dest="total")
parser.add_argument("--generator-label", help="label for the generator", type=str, default=None, dest="genlabel")
parser.add_argument("--normalize", help="Normalize yields", action="store_true")
args = parser.parse_args()
args.logy = args.log or args.logy
args.readbatch = args.readbatch and os.path.isfile(args.batch)
if args.readbatch:
    args.prefit = False

if args.genlabel is not None:
    bgstring = genlabels[args.genlabel] + " + BG"
    if args.genlabel == "bb4l":
        for k,v in sm_procs.items():
            if k == "TW" or v == r"$\mathrm{t}\bar{\mathrm{t}}$":
                sm_procs[k] = r"$\mathrm{t}\bar{\mathrm{t}} + \mathrm{tW}$"
            if v == "tX":
                 sm_procs[k] = "Other"
        proc_colors[r"$\mathrm{t}\bar{\mathrm{t}} + \mathrm{tW}$"] = proc_colors[r"$\mathrm{t}\bar{\mathrm{t}}$"]
else:
    bgstring = "FO pQCD + BG"
    #bgstring = "BG"

fits = []
if args.postfit:
    fits += ["s", "b"]
if args.prefit:
    fits += ["p"]

def full_extent(ax, pad = 0.0):
    """
    get the full extent of an axes, including axes labels, tick labels, and titles.
    credits: https://stackoverflow.com/a/14720600
    """
    # for text objects, we need to draw the figure first, otherwise the extents are undefined
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)

def plot_eventperbin(ax, bins, centers, smhists, total, signals, data, log, fit, channel):
    single_slice = args.splitbins or args.project != "none"
    angular = args.project in ["chel", "chan"]
    #factor = 1000. if angular else 1.
    factor = 1
    if fit == "p":
        fstage = "Pre"
        ftype = " "
    else:
        fstage = "Post"
        ftype = " (s + b) " if fit == "s" else " (b) "

    width = np.ones(len(bins)-1) if angular else np.diff(bins)
    colors = [proc_colors[k] for k in smhists.keys()]
    unclabel = "Unc." if single_slice else f"{fstage}fit uncertainty"
    for ibin in range(len(bins) - 1):
        vhi = (total.values()[ibin] + total.variances()[ibin] ** .5) / width[ibin] / factor
        vlo = (total.values()[ibin] - total.variances()[ibin] ** .5) / width[ibin] / factor
        ax.fill_between(
            bins[ibin : ibin + 2],
            np.array(vhi, vhi),
            np.array(vlo, vlo),
            step = "mid",
            label = unclabel if ibin == 0 else None,
            **hatchstyle)
    ax.errorbar(
        centers,
        data[0] / width / factor,
        yerr = data[1] / width / factor,
        label = "Data",
        **datastyle,
        zorder=9999
    )
    hep.histplot(
        [hist.values() / width / factor for hist in smhists.values()],
        bins = bins,
        ax = ax,
        stack = True,
        histtype = "fill",
        label = smhists.keys(),
        color = colors,
        zorder = -90
    )
    #if args.splitbins or args.project != "none":
    #    for key, signal in signals.items():
    #        symbol, mass, decaywidth = key
    #        hep.histplot(
    #            (total.values() + signal.values()) / width / factor,
    #            bins = bins,
    #            yerr = False,
    #            ax = ax,
    #            histtype = "step",
    #            color = proc_colors[symbol],
    #            linewidth = 1.5,
    #            zorder = signal_zorder[symbol] + 100,
    #            edges=False
    #        )
    axl = "Events  " if angular else "Events / GeV"
    #if args.normalize:
    #    axl += " (norm.)"
    ax.set_ylabel(axl, fontsize=26, loc="top")
    if log[0]:
        ax.set_xscale("log")
    if log[1]:
        ax.set_yscale("log")
        mperbin = 1e-7 if args.normalize else 1.
        dperbin = np.maximum(data[0], mperbin) / width / factor
        ymin = 0.5 * np.amin(dperbin)
        if args.splitbins or single_slice:
            ymax = 1.25
        else:
            ymax = 1.08 if "j" in channel else 1.3
        ymax = np.power(10, ymax * np.log10(np.amax(dperbin) / np.amin(dperbin)) \
                + np.log10(np.amin(dperbin)))
        #ax.set_ylim(ymin, ax.transData.inverted().transform(ax.transAxes.transform([0, ymax]))[1])
        ax.set_ylim(ymin, ymax)
    else:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        if ax.get_ylim()[1] > 1:
            ax.yaxis.get_offset_text().set_x(-0.17)
        else:
            ax.yaxis.get_offset_text().set_x(-0.20)



def plot_ratio(ax, bins, centers, data, total, signals, gvalues, sigscale, fit, log):
    single_slice = args.splitbins or args.project != "none"
    ax.errorbar(
        centers,
        data[0] / total.values(),
        abs(data[1] / total.values()),
        **datastyle
    )
    if fit == "p":
        fstage = "Pre"
    else:
        fstage = "Post"
    err_up = 1 + (total.variances() ** .5) / total.values()
    err_down = 1 - (total.variances() ** .5) / total.values()
    handle_unc = ax.fill_between(
        bins,
        np.r_[err_up[0], err_up],
        np.r_[err_down[0], err_down],
        step = "pre",
        label = f"{fstage}-fit uncertainty" if args.panellabels else None,
        zorder = -99,
        **hatchstyle
    )
    handles = []
    labels = []
    for key, signal in signals.items():
        symbol, mass, decaywidth = key
        idx = 1 if symbol == 'H' else 0
        ss = int(sigscale[idx]) if abs(sigscale[idx] - int(sigscale[idx])) < 2.**-11 else sigscale[idx]
        signal_label = "" if abs(ss - 1.) < 2.**-11 else f"{ss} $\\times$ "
        if symbol == "Total":
            signal_label += "Total"
        elif symbol == "A" or symbol == "H":
            signal_label = f"{symbol}({mass}, {decaywidth}%)"
        elif symbol in [r"$\eta_{\mathrm{t}}$", r"$\chi_{\mathrm{t}}$", r"$\psi_{\mathrm{t}}$"]:
            signal_label = symbol
        else:
            signal_label = "Signal"

        poiname = "r" if args.onlyres else "g"
        if fit == "p":
            if symbol == "A" or symbol == "H":
                signal_label += f", $\\mathrm{{{poiname}}}_{{\\mathrm{{{symbol}}}}} = 1$"
            elif symbol in [r"$\eta_{\mathrm{t}}$", r"$\chi_{\mathrm{t}}$", r"$\psi_{\mathrm{t}}$"]:
                if args.xsec:
                    signal_label = f"{symbol}, $\\sigma({symbol[1:-1]}) = 6.4$ pb"
                else:
                    signal_label = f"{symbol}, $\\mu({symbol[1:-1]}) = 1$"
        elif key in gvalues and gvalues[key] is not None:
            poi = ("sigma" if args.xsec else "mu", " \,\\mathrm{pb}" if args.xsec else "")
            if symbol == "A" or symbol == "H":
                if fit == "s":
                    if len(gvalues[key]) == 2:
                        signal_label += f", $\\mathrm{{{poiname}}}_{{\\mathrm{{{symbol}}}}} = {gvalues[key][0]:.2f} \\pm {gvalues[key][1]:.2f}$"
                    elif len(gvalues[key]) == 1:
                        signal_label += f", $\\mathrm{{{poiname}}}_{{\\mathrm{{{symbol}}}}} = {gvalues[key][0]:.2f}$ (fix.)"
                    else:
                        signal_label += f", $\\mathrm{{g}}_{{\\mathrm{{{symbol}}}}} = {gvalues[key][0]}_{{-{gvalues[key][2]}}}^{{+{gvalues[key][1]}}}$"
                elif fit == "b":
                    signal_label += f", $\\mathrm{{{poiname}}}_{{\\mathrm{{{symbol}}}}} = 0$"
            elif symbol == r"$\eta_{\mathrm{t}}$":
                if fit == "s":
                    if len(gvalues[key]) == 2:
                        signal_label = f"$\\eta_{{\\mathrm{{t}}}}$, $\\{poi[0]}(\\eta_{{\\mathrm{{t}}}}) = {gvalues[key][0]:.1f} \\pm {gvalues[key][1]:.1f}{poi[1]}$"
                    else:
                        signal_label = f"$\\eta_{{\\mathrm{{t}}}}$, $\\{poi[0]}(\\eta_{{\\mathrm{{t}}}}) = {gvalues[key][0]:.1f}_{{-{gvalues[key][2]:.1f}}}^{{+{gvalues[key][1]:.1f}}}{poi[1]}$"
                elif fit == "b":
                    signal_label = f"$\\eta_{{\\mathrm{{t}}}}$, $\\{poi[0]}(\\eta_{{\\mathrm{{t}}}}) = 0{poi[1]}$"
            elif symbol == r"$\chi_{\mathrm{t}}$":
                if fit == "s":
                    if len(gvalues[key]) == 2:
                        signal_label = f"$\\chi_{{\\mathrm{{t}}}}$, $\\{poi[0]}(\\chi_{{\\mathrm{{t}}}}) = {gvalues[key][0]:.1f} \\pm {gvalues[key][1]:.1f}{poi[1]}$"
                    else:
                        signal_label = f"$\\chi_{{\\mathrm{{t}}}}$, $\\{poi[0]}(\\chi_{{\\mathrm{{t}}}}) = {gvalues[key][0]:.1f}_{{-{gvalues[key][2]:.1f}}}^{{+{gvalues[key][1]:.1f}}}{poi[1]}$"
                elif fit == "b":
                    signal_label = f"$\\chi_{{\\mathrm{{t}}}}$, $\\{poi[0]}(\\chi_{{\\mathrm{{t}}}}) = 0{poi[1]}$"
            elif symbol == r"$\psi_{\mathrm{t}}$":
                if fit == "s":
                    if len(gvalues[key]) == 2:
                        signal_label = f"$\\psi_{{\\mathrm{{t}}}}$, $\\{poi[0]}(\\psi_{{\\mathrm{{t}}}}) = {gvalues[key][0]:.1f} \\pm {gvalues[key][1]:.1f}{poi[1]}$"
                    else:
                        signal_label = f"$\\psi_{{\\mathrm{{t}}}}$, $\\{poi[0]}(\\psi_{{\\mathrm{{t}}}}) = {gvalues[key][0]:.1f}_{{-{gvalues[key][2]:.1f}}}^{{+{gvalues[key][1]:.1f}}}{poi[1]}$"
                elif fit == "b":
                    signal_label = f"$\\psi_{{\\mathrm{{t}}}}$, $\\{poi[0]}(\\psi_{{\\mathrm{{t}}}}) = 0{poi[1]}$"

        handle_signal = hep.histplot(
            (total.values() + signal.values()) / total.values(),
            bins = bins,
            yerr = False,
            ax = ax,
            histtype = "step",
            color = proc_colors[symbol],
            linewidth = 1.5,
            label = signal_label,
            zorder = signal_zorder[symbol],
            edges = (not single_slice)
        )
        handles.append(handle_signal[0][0])
        labels.append(signal_label)
    #for pos in [0.8, 0.9, 1.1, 1.2]:
    #    ax.axhline(y = pos, linestyle = ":", linewidth = 0.5, color = "black")
    ax.axhline(y = 1, linestyle = "--", linewidth = 0.35, color = "black")
    if single_slice:
        ax.set_ylim(0.887, 1.113)
        ax.set_yticks([0.9, 1.0, 1.1])
    #elif fit == "p" and not args.normalize:
    #    ax.set_ylim(0.79, 1.21)
    #    ax.set_yticks([0.8, 1.0, 1.2])
    else:
        ax.set_ylim(0.895, 1.105)
        ax.set_yticks([0.9, 1.0, 1.1])
    ax.set_ylabel("Ratio to " + bgstring, fontsize=26)
    if fit == "p":
        fittype = "Prefit"
        fittypelen = len(fittype)
    elif fit == "b":
        fittype = f"Postfit ({bgstring})"
        fittypelen = len(fittype)
    elif fit == "s" and len(args.assignal):
        fittype = f"Postfit ({bgstring} "
        fittypelen = len(fittype)
        if "EtaT" in args.assignal:
            fittype += "+ $\mathbf{\eta_{\mathrm{t}}}$"
            fittypelen += len(" + x")
        if "ChiT" in args.assignal:
            fittype += "+ $\mathbf{\chi_{\mathrm{t}}}$"
            fittypelen += len(" + x")
        if "PsiT" in args.assignal:
            fittype += "+ $\mathbf{\psi_{\mathrm{t}}}$"
            fittypelen += len(" + x")
        fittype += ")"
    else:
        fittype = f"Postfit ({bgstring} + A/H)"
        fittypelen = len(fittype)
    if args.normalize:
        fittype += ", normalized"
    if not single_slice:
        handles.insert(0, Rectangle((0,0), 0, 0, facecolor="white", edgecolor="white", alpha=0.))
        labels.insert(0, " "*(fittypelen+10))
    if not (single_slice and args.panel == "both"):
        handles.append(handle_unc)
        labels.append("Uncertainty")
    if log[0]:
        ax.set_xscale("log")
    if not (single_slice and args.panel == "both" and len(signals) == 0):
        if single_slice and args.panel == "both":
            legend_ncol = 2 if len(signals) > 2 else 1
            if len(signals) == 3:
                # reshuffle total
                handles = [*handles[1:], Rectangle((0,0), 0, 0, facecolor="white", edgecolor="white", alpha=0.), handles[0]]
                labels = [*labels[1:], "${ }^{ }_{ }$", labels[0]]
            ax.legend(handles=handles, labels=labels, loc = "lower left",  bbox_to_anchor = (0, 0.0, 1, 0.2), borderaxespad = 0, ncol = legend_ncol, mode = "expand", frameon = False, handlelength=1.5, handletextpad=0.6, labelspacing=0.3)
        else:
            legend_ncol = 1 if single_slice else 5
            ax.legend(handles=handles, labels=labels, loc = "lower left", bbox_to_anchor = (0, 1.0, 1, 0.2), borderaxespad = 0, ncol = legend_ncol, mode = "expand", fancybox = False).get_frame().set_edgecolor("black")



def plot_diff(ax, bins, centers, data, total, signals, gvalues, sigscale, fit):
    raise NotImplementedError("cba to update plot_diff")
    width = np.diff(bins)
    ax.errorbar(
        centers,
        (data[0] - total.values()) / width,
        #((data[1] ** 2 + total.variances()) ** .5) / width,
        data[1] / width, # cant add the errors as if uncorr, the total is already from a fit to the data
        **datastyle
    )
    for ibin in range(len(bins) - 1):
        vhi = (total.variances()[ibin] ** .5) / width[ibin]
        vlo = (total.variances()[ibin] ** .5) / -width[ibin]
        ax.fill_between(
            bins[ibin : ibin + 2],
            np.array(vhi, vhi),
            np.array(vlo, vlo),
            step = "pre",
            **hatchstyle)
    for idx, (key, signal) in enumerate(signals.items()):
        symbol, mass, decaywidth = key
        idx = 1 if symbol == 'H' else 0
        ss = int(sigscale[idx]) if abs(sigscale[idx] - int(sigscale[idx])) < 2.**-11 else sigscale[idx]
        signal_label = "" if abs(ss - 1.) < 2.**-11 else f"{ss} $\\times$ "
        if symbol == "Total":
            signal_label += "Total"
        elif symbol == "A" or symbol == "H":
            signal_label = f"{symbol}({mass}, {decaywidth}%)"
        elif symbol == r"$\eta_{\mathrm{t}}$":
            signal_label = r"$\eta_{\mathrm{t}}$"
        else:
            signal_label = "Signal"

        if key in gvalues and gvalues[key] is not None:
            if symbol == "A" or symbol == "H":
                if fit == "s":
                    signal_label += f", $\\mathrm{{g}}_{{\\mathrm{{{symbol}}}}} = {gvalues[key]}$"
                elif fit == "b":
                    signal_label += f", $\\mathrm{{g}}_{{\\mathrm{{{symbol}}}}} = 0$"
            elif symbol == r"$\eta_{\mathrm{t}}$":
                if fit == "s":
                    signal_label = f"$\\eta_{{\\mathrm{{t}}}}$, $\\mu^{{\\eta_{{\\mathrm{{t}}}}}} = {gvalues[key]}$"
                elif fit == "b":
                    signal_label = f"$\\eta_{{\\mathrm{{t}}}}$, $\\mu^{{\\eta_{{\\mathrm{{t}}}}}} = 0$"

        hep.histplot(
            signal.values() / width,
            bins = bins,
            yerr = np.zeros(len(signal.axes[0])),
            ax = ax,
            histtype = "step",
            color = proc_colors[symbol],
            linewidth = 1.75,
            label = signal_label,
            zorder = signal_zorder[symbol],
            edges = False
        )
    ax.set_ylabel("Difference to\nperturbative SM")
    #ax.axhline(y = 1, linestyle = "--", linewidth = 0.35, color = "black")
    ax.legend(loc = "lower left", bbox_to_anchor = (0, 1.0, 1, 0.2), borderaxespad = 0, ncol = 5, mode = "expand", fancybox = False).get_frame().set_edgecolor("black")



def plot(channel, year, fit,
         smhists, datavalues, total, promotions, signals, gvalues, sigscale, datahist_errors,
         binning, num_extrabins, extra_axes, first_ax_binning, first_ax_width, bins, centers, log, cuts):
    if len(smhists) == 0:
        return

    single_slice = args.splitbins or args.project != "none"
    ismbbll = r'$m_{\mathrm{b}\mathrm{b}\ell\ell}$' in list(binning.keys())[0]
    allsigs = signals | promotions
    if args.panel == "both":
        fig, (ax0, ax1, ax2) = plt.subplots(
            nrows = 3,
            sharex = True,
            gridspec_kw = {"height_ratios": [0.115 if single_slice else 0.001, 1, 0.9]},
            figsize = (5.0, 6.6) if single_slice else (19.2, 6.6),
            dpi=600
        )
    else:
        fig, (ax0, ax1) = plt.subplots(
            nrows = 2,
            sharex = True,
            gridspec_kw = {"height_ratios": [0.001, 1]},
            figsize = (5.0, 5.5) if single_slice else (19.2, 3.5),
            dpi=600
        )
        ax2 = ax1
    ax0.set_axis_off()   
    if args.panel != "lower":
        plot_eventperbin(ax1, bins, centers, smhists, total, allsigs, (datavalues, datahist_errors), log, fit, channel)
    if args.panel != "upper":
        if args.lower == "ratio":
            plot_ratio(ax2, bins, centers, (datavalues, datahist_errors), total, allsigs, gvalues, sigscale, fit, log)
        elif args.lower == "diff":
            plot_diff(ax2, bins, centers, (datavalues, datahist_errors), total, allsigs, gvalues, sigscale, fit)
        else:
            raise ValueError(f"Invalid lower type: {args.lower}")
    if ismbbll or year != "Run 2":
        ax2.set_ylim(0.79, 1.21)
        ax2.set_yticks([0.8, 1.0, 1.2])
    if not single_slice:
        for pos in bins[::len(first_ax_binning) - 1][1:-1]:
            if args.panel != "lower":
                ax1.axvline(x = pos, linestyle = "--", linewidth = 0.5, color = "gray")
            if args.panel != "upper":
                ax2.axvline(x = pos, linestyle = "--", linewidth = 0.5, color = "gray")
    if args.panel != "lower":
        ax1.minorticks_on()
        ax1.tick_params(axis="both", which="both", direction="in", bottom=True, top=True, left=True, right=True)
        legend_ncol = 3 if single_slice else len(smhists) + 2
        ax1.legend(loc = "lower left", bbox_to_anchor = (0, 1, 1, 0.2),
                   borderaxespad = 0, ncol = legend_ncol, mode = "expand", edgecolor = "black", framealpha = 1, fancybox = False, reverse = True)
        ax1.set_xlabel("")

    ax2.minorticks_on()
    ax2.tick_params(axis="both", which="both", direction="in", bottom=True, top=True, left=True, right=True)
    ticks = []
    if "j" in channel:
        ticklocs = np.linspace(500, 1500, 3)
        ticklocs_minor = np.linspace(400, 1600, 13)
    elif ismbbll:
        if single_slice:
            #ticklocs = np.array([150, 200, 400, 800]) if log[0] else np.linspace(200, 800, 4)
            #ticklocs_minor = np.array([140, 200, 300, 400, 500, 600, 700, 800, 900]) if log[0] else np.arange(150, 900, 50)
            ticklocs = np.array([100, 200, 400, 800]) if log[0] else np.linspace(200, 800, 4)
            ticklocs_minor = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900]) if log[0] else np.arange(100, 910, 50)
        else:
            ticklocs = np.linspace(300, 700, 2)
            ticklocs_minor = np.arange(200, 900, 100)
    else:
        if single_slice:
            ticklocs = np.linspace(400, 1300, 4)
            ticklocs_minor = np.arange(300, 1500, 100)
        else:
            ticklocs = np.linspace(600, 1200, 2)
            ticklocs_minor = np.linspace(450, 1350, 7)  
    ticks = np.concatenate(
        [ticklocs - first_ax_binning[0] + i * first_ax_width
        for i in range(num_extrabins)])
    ticks_minor = np.concatenate(
        [ticklocs_minor - first_ax_binning[0] + i * first_ax_width
        for i in range(num_extrabins)])
    
    ax2.set_xticks(ticks_minor if first_ax_width > 0 else [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], minor=True)
    if args.noxaxis:
        ax2.set_xticks(ticks if first_ax_width > 0 else [-1, 0, 1], minor=False)
        ax2.set_xticklabels([])
        ax2.set_xlabel("")
    else:       
        tickloc_labels = [f"{t:.0f}" for i in range(num_extrabins) for t in ticklocs] if first_ax_width > 0 else ["-1", "0", "1"]
        ax2.set_xticks(ticks if first_ax_width > 0 else [-1, 0, 1], tickloc_labels, minor=False)
        ax2.set_xlabel(list(binning.keys())[0], fontsize=26, loc="right")

    title = channel.replace('m', '$\\mu$').replace('4p', '4+')
    if fit == "p":
        fittype = "Prefit"
    elif fit == "b":
        fittype = f"Postfit ({bgstring})"
    elif fit == "s" and len(args.assignal):
        fittype = f"Postfit ({bgstring} "
        if "EtaT" in args.assignal:
            fittype += "+ $\mathbf{\eta_{\mathrm{t}}}$"
        if "ChiT" in args.assignal:
            fittype += "+ $\mathbf{\chi_{\mathrm{t}}}$"
        if "PsiT" in args.assignal:
            fittype += "+ $\mathbf{\psi_{\mathrm{t}}}$"
        fittype += ")"
    else:
        fittype = f"Postfit ({bgstring} + A/H)"
    
    if args.panellabels:
        if args.panel == "upper":
            xpos = 0.97
            ypos = 1.03
            va = "bottom"
            ha = "right"
        else:
            xpos = 0.04 if single_slice else 0.01
            ypos = 0.97 if single_slice else 0.95
            va = "top"
            ha = "left"

        if single_slice and args.panel == "upper":
            annstr = title
        elif single_slice and args.panel == "lower":
            annstr = fittype# + ", " + title
        else:
            annstr = fittype
        if args.normalize:
            annstr += ", normalized"
        ax2.annotate(annstr, (xpos, ypos), xycoords="axes fraction", va=va, ha=ha, fontsize=20, fontweight="normal" if (single_slice and args.panel == "upper") else "bold", zorder=7777)
        #if args.panel == "both":
        #    ax1.annotate(title, (0.97, 1.028), xycoords="axes fraction", va="bottom", ha="right", fontsize=20, fontweight="normal", zorder=7777) # xxx
    elif args.panel != "upper":
        annstr = fittype
        if args.normalize:
            annstr += ", normalized"
        ax2.annotate(annstr, (0.007, 1.038), xycoords="axes fraction", va="bottom", ha="left", fontsize=20, fontweight="bold", zorder=7777)

    if args.panel != "upper" and not any([ss in ["EtaT", "ChiT", "PsiT"] for ss in args.assignal]) and fit != "b":
        #btxt = etat_blurb([sm_procs["EtaT"] in smhists])
        if sm_procs["EtaT"] in smhists:
            btxt = "Including $t \\bar{t}$ bound state $\\eta_t$"
        else:
            btxt = "No $\\mathrm{t \\bar{t}}$ bound states"
        xpos = 0.04 if single_slice else 0.01
        ypos = 0.86 if args.panellabels else 0.96
        ax2.annotate(btxt, (xpos, ypos), xycoords="axes fraction", va="top", ha="left", fontsize=20, zorder=7777)
    

    cmslabel = args.cmslabel
    if args.panel == "both":
        if not single_slice:# and r'\ell' not in title:
            ax0.set_title(title)
        hep.cms.label(ax = ax0, data=True, label=cmslabel, lumi = lumis[year], loc = 0, year = None if year == "Run 2" else year, fontsize = 22 if single_slice and cmslabel is not None else 26)
        fig.subplots_adjust(hspace = 0.24, left = 0.055, right = 1 - 0.003, top = 1 - 0.05)
    else:
        hep.cms.label(ax = ax0, data=True, label=cmslabel, lumi = lumis[year], loc = 0, year = None if year == "Run 2" else year, fontsize = 22 if single_slice and cmslabel is not None else 26)
        if len(allsigs) == 3:
            hspace = 1.5
        else:
            hspace = 0.3 * (len(allsigs) + 1) - 0.06 if args.panel == "lower" else 0.50
        fig.subplots_adjust(hspace = hspace, left = 0.075, right = 1 - 0.025, top = 1 - 0.075)
    bbox = ax2.get_position()
    offset = -0.01
    if single_slice:
        if args.panel == "both":
            #if len(allsigs) == 0:
            #    offset = 0.05
            #elif len(allsigs) == 1:
            #    offset = -0.012
            #elif len(allsigs) == 3:
            #    offset = -0.11
            offset = 0.043
        else:
            offset = -0.025 if args.panel == "lower" else -0.02
    ax2.set_position([bbox.x0, bbox.y0 + offset, bbox.x1 - bbox.x0, bbox.y1 - bbox.y0])
    figwidth = 6.0 if single_slice else 19.2
    if args.panel == "both":
        fig.set_size_inches(w = figwidth, h = 1.5 * fig.get_figheight())
    else:
        fig.set_size_inches(w = figwidth, h = 1.0 * fig.get_figheight())
    extent = 'tight'# if args.plotupper else full_extent(ax2).transformed(fig.dpi_scale_trans.inverted())

    sstr = [ss for ss in allsigs.keys() if ss[0] != "Total"]
    if len(allsigs) == 0:
        sstr = "bkg"
    elif any([ss in sstr[0][0] for ss in ["eta", "chi", "psi"]]):
        sstr = "__".join(args.assignal)
    else:
        sstr = [ss[0] + "_m" + str(ss[1]) + "_w" + str(float(ss[2])).replace(".", "p") for ss in sstr if (ss[0] == "A" or ss[0] == "H")]
        sstr = "__".join(sstr)
    cstr = channel.replace(r'$\ell\ell$', 'll').replace(r'$\ell$j', 'lj').replace(r'$\ell$, 3j', 'l3j').replace(r'$\ell$, $\geq$ 4j', 'l4pj')
    ystr = year.replace(" ", "").lower()
    ax1.margins(x = 0, y = 0)
    ax2.margins(x = 0, y = 0)

    bintexts = []
    for fmt in args.fmt:
        if single_slice:
            for i in range(num_extrabins):
                for txt in bintexts:
                    txt.remove()
                bintexts = []
                ypos = 0.08 if args.panel == "lower" else 0.9 if single_slice else 0.912
                #if args.splitbins:
                #    for j, (variable, edges) in enumerate(reversed(extra_axes.items())):
                #        edge_idx = np.unravel_index(i, tuple(len(b) - 1 for b in extra_axes.values()))[j]
                #        text = r"{} < {} < {}".format(edges[edge_idx], variable, edges[edge_idx + 1])
                #        bintexts.append(ax1.text(1 / len(extra_axes) * (j + 0.5), ypos, text, horizontalalignment = "center", fontsize = 22, transform = ax1.transAxes))
                #else:
                for j, cuttext in enumerate(cuts[1:]):
                    bintexts.append(ax1.text(1 / len(cuts[1:]) * (j + 0.5), ypos, cuttext, horizontalalignment = "center", fontsize = 22, transform = ax1.transAxes))
                if first_ax_width > 0:
                    ax2.set_xlim(first_ax_width*i+10 if ismbbll and log[0] else first_ax_width*i, first_ax_width*(i+1))
                else:
                    ax2.set_xlim(-1, 1)
                fig.align_ylabels()
                #if args.splitbins:
                #    fig.savefig(f"{args.odir}/{sstr}{args.ptag}_fit_{fit}_{cstr}_{ystr}_{args.panel}_bin{i+1}{fmt}", transparent = True, bbox_inches = extent)
                #else:
                fig.savefig(f"{args.odir}/{sstr}{args.ptag}_fit_{fit}_{cstr}_{ystr}_{args.panel}_{args.project if args.project != 'none' else ''}{cuts[0]}{fmt}", transparent = True, bbox_inches = extent)
        else:
            if args.panel != "lower":
                for j, (variable, edges) in enumerate(extra_axes.items()):
                    for i in range(num_extrabins):
                        edge_idx = np.unravel_index(i, tuple(len(b) - 1 for b in extra_axes.values()))[j]
                        text = r"{} < {} < {}".format(edges[edge_idx], variable, edges[edge_idx + 1])
                        if not single_slice:
                            ax1.text(1 / num_extrabins * (i + 0.5), 0.912 - j * 0.11, text, horizontalalignment = "center", fontsize = 18, transform = ax1.transAxes)
            fig.align_ylabels()
            fig.savefig(f"{args.odir}/{sstr}{args.ptag}_fit_{fit}_{cstr}_{ystr}_{args.panel}{fmt}", transparent = True, bbox_inches = extent)
    fig.clf()



def sum_kwargs(channel, year, *summands):
    ret = summands[0].copy()
    ret["channel"] = channel
    ret["year"] = year
    for summand in summands[1:]:
        ret["smhists"] = {k: ret["smhists"][k] + summand["smhists"][k] for k in ret["smhists"]}
        ret["datavalues"] = ret["datavalues"] + summand["datavalues"]
        ret["total"] = ret["total"] + summand["total"]
        ret["promotions"] = {k: ret["promotions"][k] + summand["promotions"][k] for k in ret["promotions"]}
        ret["signals"] = {k: ret["signals"][k] + summand["signals"][k] for k in ret["signals"]}
        ret["datahist_errors"] = (ret["datahist_errors"] ** 2 + summand["datahist_errors"] ** 2) ** .5
    return ret

def cut_string(variable, binedges, cut, mstr):
    if cut[0] == binedges[0] and cut[1] == binedges[-1]:
        return ["", ""]
    elif cut[0] == binedges[0]:
        return [f"_{mstr}lt{cut[1]}", f"{variable} < {cut[1]} GeV"]
    elif cut[1] == binedges[-1]:
        return [f"_{mstr}gt{cut[0]}", f"{variable} > {cut[0]} GeV"]
    else:
        return [f"_{mstr}{cut[0]}to{cut[1]}", f"{cut[0]} < {variable} < {cut[1]} GeV"]

@numba.njit(cache=True, nogil=True)
def actually_project_numba(plane, nbins, target, cut, icut, matrix, lenyears, isarray):
    values = np.zeros(nbins[target])
    variances = None if isarray else np.zeros(nbins[target])
    for ibin in range(np.prod(nbins)):
        idxs = index_1n(ibin, nbins)
        if icut[0] <= idxs[0] < icut[1]:
            values[ idxs[target] ] += plane[ibin]
            if not isarray:
                nchannel = int(len(matrix) / lenyears / np.prod(nbins))
                for iyear in range(lenyears):
                    for ichannel in range(nchannel):
                        iycb0 = index_n1(np.append(idxs, [iyear, ichannel]), np.append(nbins, [lenyears, nchannel]))
                        for iycb1 in range(len(matrix)):
                            iother = index_1n(iycb1, np.append(nbins, [lenyears, nchannel]))
                            if iother[target] != idxs[target]:
                                continue
                            if icut[0] <= iother[0] < icut[1]:
                                variances[ idxs[target] ] += matrix[iycb0, iycb1]
    return values, variances

def actually_project(plane, nbins, target, cut, icut, matrix):
    isarray = isinstance(plane, np.ndarray)
    values, variances = actually_project_numba(plane if isarray else np.array(plane.values()), 
                                               np.array(nbins), target, np.array(cut), np.array(icut), matrix,
                                               len(years), isarray)
    histogram = None
    if not isarray:
        histogram = Hist.new.Regular(nbins[target], -1 if target != 0 else cut[0], 1 if target != 0 else cut[1], name = "").Weight()
        histogram.view().value = values
        histogram.view().variance = variances
    return values if isarray else histogram

def project(planes, nbins, target, cut, icut, matrix):
    targets = {
        "chel": 1,
        "chan": 2
    }
    masses = {
        r"$m_{\mathrm{t}\bar{\mathrm{t}}}$": "mtt",
        r"$m_{\mathrm{b}\mathrm{b}\ell\ell}$": "mbbll"
    }
    target = targets.get(target, 0)
    ret = planes.copy()
    if target == 1:
        label = list(planes["binning"].keys())[2]
    elif target == 2:
        label = list(planes["binning"].keys())[1]
    else:
        label = list(planes["binning"].keys())[target]

    print("Projecting SM...")
    ret["smhists"] = {k: actually_project(ret["smhists"][k], nbins, target, cut, icut, matrix) for k in ret["smhists"]}
    print("Projecting data...")
    ret["datavalues"] = actually_project(ret["datavalues"], nbins, target, cut, icut, matrix)
    print("Projecting total...")
    ret["total"] = actually_project(ret["total"], nbins, target, cut, icut, matrix)
    print("Projecting signals...")
    ret["promotions"] = {k: actually_project(ret["promotions"][k], nbins, target, cut, icut, matrix) for k in ret["promotions"]}
    ret["signals"] = {k: actually_project(ret["signals"][k], nbins, target, cut, icut, matrix) for k in ret["signals"]}
    print("Projecting data errors...")
    dataerr_lo = (actually_project(ret["datahist_errors"][0]**2, nbins, target, cut, icut, matrix))**.5
    dataerr_hi = (actually_project(ret["datahist_errors"][1]**2, nbins, target, cut, icut, matrix))**.5
    ret["datahist_errors"] = np.array([dataerr_lo, dataerr_hi])
    ret["binning"] = {label : list(planes["binning"].values())[target] if target == 0 else [-1, -1/3, 1/3, 1]}
    ret["num_extrabins"] = int(np.prod(list(len(edges) - 1 for edges in list(ret["binning"].values())[1:])))
    ret["extra_axes"] = {'none': list(ret["binning"].values())[0] if target == 0 else [-1, -1/3, 1/3, 1]}
    ret["first_ax_binning"] = list(ret["binning"].values())[0] if target == 0 else np.array([-1, -1/3, 1/3, 1])
    ret["first_ax_width"] = ret["first_ax_binning"][-1] - ret["first_ax_binning"][0] if target == 0 else 0
    binwidths = np.diff(ret["first_ax_binning"])
    ret["bins"] = np.array(ret["first_ax_binning"]) - ret["first_ax_binning"][0] if target == 0 else ret["first_ax_binning"]
    ret["centers"] = (ret["bins"][1:] + ret["bins"][:-1]) / 2
    ret["cuts"] = cut_string(
        list(planes["binning"].keys())[0].replace(" [GeV]", ""),
        list(planes["binning"].values())[0],
        cut,
        masses[ list(planes["binning"].keys())[0].replace(" [GeV]", "") ]
    )
    print("projection done.")
    return ret

def plot_projection(sums, binedges, cut, matrix):
    icut = [binedges[0].index(cc) for cc in cut]
    nbins = [len(bb) - 1 for bb in binedges]
    sums = project(sums, nbins, args.project, cut, icut, matrix)
    plot(**sums)


def project_channel_year_unc(matrix, nbins):
    assert matrix.shape[0] % nbins == 0
    nchy = matrix.shape[0] // nbins
    out = np.zeros((nbins,nbins))
    for i in range(nchy):
        for j in range(nchy):
            submat = matrix[i*nbins:(i+1)*nbins,j*nbins:(j+1)*nbins]
            out += submat
    return out

def do_slicing(h, islice, nslice):
    isarray = isinstance(h, np.ndarray)
    arr = h if isarray else h.values()
    lenslice = len(arr) // nslice
    arr = arr[islice*lenslice:(islice+1)*lenslice]
    if isarray:
        return arr
    else:
        var = h.variances()
        var = var[islice*lenslice:(islice+1)*lenslice]
        histogram = Hist.new.Regular(lenslice, 0, lenslice, name = "").Weight()
        histogram.view().value = arr
        histogram.view().variance = var
        return histogram

def plot_slices(sums, nslice=9):
    for islice in range(nslice):
        ret = sums.copy()
        ret["smhists"] = {k: do_slicing(ret["smhists"][k], islice, nslice) for k in ret["smhists"]}
        ret["datavalues"] = do_slicing(ret["datavalues"],islice, nslice)
        ret["total"] = do_slicing(ret["total"], islice, nslice)
        ret["promotions"] = {k: do_slicing(ret["promotions"][k], islice, nslice) for k in ret["promotions"]}
        ret["signals"] = {k: do_slicing(ret["signals"][k], islice, nslice) for k in ret["signals"]}
        dataerr_lo = (do_slicing(ret["datahist_errors"][0]**2, islice, nslice))**.5
        dataerr_hi = (do_slicing(ret["datahist_errors"][1]**2, islice, nslice))**.5
        ret["datahist_errors"] = np.array([dataerr_lo, dataerr_hi])
        ret["binning"] = {l: v for l,v in ret["binning"].items() if l in [r"$m_{\mathrm{t}\bar{\mathrm{t}}}$ [GeV]", r"$m_{\mathrm{b}\mathrm{b}\ell\ell}$ [GeV]"]}
        ret["num_extrabins"] = 1
        ret["extra_axes"] = {'none': list(ret["binning"].values())[0]}
        ret["first_ax_binning"] = list(ret["binning"].values())[0]
        ret["first_ax_width"] = ret["first_ax_binning"][-1] - ret["first_ax_binning"][0]
        ret["bins"] = np.array(ret["first_ax_binning"]) - ret["first_ax_binning"][0]
        ret["centers"] = (ret["bins"][1:] + ret["bins"][:-1]) / 2
        text = []
        for j, (variable, edges) in enumerate(reversed(sums["extra_axes"].items())):
            edge_idx = np.unravel_index(islice, tuple(len(b) - 1 for b in sums["extra_axes"].values()))[j]
            text.append(r"{} < {} < {}".format(edges[edge_idx], variable, edges[edge_idx + 1]))
        ret["cuts"] = [f"bin{islice+1}", *text]
        plot(**ret)

def add_covariance(histogram, matrix):
    for ibin in range(len(histogram.values())):
        histogram[ibin] = Hist.accumulators.WeightedSum(
            value = histogram.values()[ibin],
            variance = math.sqrt(matrix.values()[ibin, ibin])
        )
    return histogram

def zero_variance(histogram):
    histogram.view().variance = 0
    return histogram

def normalize_yields(h, normto=None):
    isarray = isinstance(h, np.ndarray)
    if isarray:
        values = h
        variances = np.zeros_like(h)
    else:
        values = h.values()
        variances = h.variances()
    if normto is None:
        total_yield = np.sum(values)
    else:
        total_yield = np.sum(normto)
    values_norm = values / total_yield
    variances_norm = variances / total_yield**2
    if isarray:
        return values_norm
    else:
        histogram = Hist(*[copy.deepcopy(ax) for ax in h.axes], storage="Weight")
        histogram.view().value = values_norm
        histogram.view().variance = variances_norm
        return histogram

#def normalize_variance(yields, covmat):
#    total_yield = np.sum(yields)
#    yields_normed = yields / total_yield
#    variance_norm = np.zeros_like(yields)
#    for i in range(len(yields)):
#        variance_norm[i] = (
#            covmat[i,i] - 2 * yields_normed[i] * np.sum(covmat[i]) \
#                + yields_normed[i]**2 * np.sum(covmat) 
#        ) / total_yield**2
#
#    test = normalize_covariance(yields, covmat)
#    assert np.all(np.isclose(np.diag(test), variance_norm))
#    return variance_norm

def normalize_covariance(yields, covmat):
    total_yield = np.sum(yields)
    yields_normed = yields / total_yield
    return 1/total_yield**2 * (
        covmat
        - np.outer(yields_normed, np.sum(covmat, axis=0))
        - np.outer(np.sum(covmat, axis=1), yields_normed)
        + np.outer(yields_normed, yields_normed) * np.sum(covmat)
    )

def normalize(sums, matrix):
    ret = sums.copy()
    ret["smhists"] = {k: normalize_yields(sums["smhists"][k], normto=sums["total"].values()) for k in sums["smhists"]}
    ret["datavalues"] = normalize_yields(sums["datavalues"])
    ret["total"] = normalize_yields(sums["total"])
    ret["promotions"] = {k: normalize_yields(sums["promotions"][k], normto=sums["total"].values()) for k in sums["promotions"]}
    ret["signals"] = {k: normalize_yields(sums["signals"][k], normto=sums["total"].values()) for k in sums["signals"]}

    total_var = np.diag(normalize_covariance(sums["total"].values(), matrix))
    ret["total"].view().variance = total_var

    dataerr_lo = np.sqrt(np.diag(normalize_covariance(sums["datavalues"], np.diag(sums["datahist_errors"][0]**2))))
    dataerr_hi = np.sqrt(np.diag(normalize_covariance(sums["datavalues"], np.diag(sums["datahist_errors"][1]**2))))
    ret["datahist_errors"] = np.array([dataerr_lo, dataerr_hi])

    return ret


gvalues_p = None

signal_name_pat = re.compile(r"(A|H)_m(\d+)_w(\d+p?\d*)_")
year_summed = {}
with uproot.open(args.batch if args.readbatch else args.ifile) as f:
    for channel, year, fit in product(channels, years, fits):
        for binning_channels, binning in binnings.items():
            if channel in binning_channels:
                break
        if args.readbatch:
            if not args.batch.endswith(f"_{fit}.root"):
                continue
            dname = f"{channel}_{year}_postfit"
        else:
            fitkey = "b" if fit == "s" and args.poi == "fixed" else fit
            dname = f"shapes_fit_{fitkey}/{channel}_{year}" if fit != "p" else f"shapes_prefit/{channel}_{year}"
        if dname not in f:
            continue
        directory = f[dname]
        if channel in ("ee", "em", "mm"):
            nbins = len(directory["TT"].to_hist().values()) / (len(binning[r"$c_{\mathrm{hel}}$"]) - 1)
            nbins /= len(binning[r"$c_{\mathrm{han}}$"]) - 1
            if nbins == len(binning[r"$m_{\mathrm{t}\bar{\mathrm{t}}}$ [GeV]"]) - 1:
                binning = {k: v for k, v in binning.items() if k != r"$m_{\mathrm{b}\mathrm{b}\ell\ell}$ [GeV]"}
            else:
                binning = {k: v for k, v in binning.items() if k != r"$m_{\mathrm{t}\bar{\mathrm{t}}}$ [GeV]"}
                #if args.logx:
                #    binning[list(binning.keys())[0]][0] = 0

        num_extrabins = np.prod(list(len(edges) - 1 for edges in list(binning.values())[1:]))
        extra_axes = {k: v for i, (k, v) in enumerate(binning.items()) if i != 0}
        first_ax_binning = list(binning.values())[0]
        first_ax_width = first_ax_binning[-1] - first_ax_binning[0]
        binwidths = np.diff(first_ax_binning)
        bins = (np.cumsum(binwidths)[None] + (np.arange(num_extrabins) * first_ax_width)[:, None]).flatten()
        bins = np.r_[0, bins]
        centers = (bins[1:] + bins[:-1]) / 2
        
        smhists = {}
        signals = {}
        promotions = {}
        for proc, label in sm_procs.items():
            if proc not in directory:
                continue

            # combine hack, see args.doah block
            if fit == "p" and args.ipf != "" and proc in ["EtaT", "ChiT", "PsiT"]:
                ipf = f"{os.path.dirname(args.ifile)}/ahtt_input.root" if args.ipf == 'default' else args.ipf
                with uproot.open(f"{ipf}") as ipf:
                    hist = ipf[f"{channel}_{year}"][proc].to_hist()[:len(centers)]
            else:
                hist = directory[proc].to_hist()[:len(centers)]

            if proc in args.ignore:
                continue
            elif proc not in args.assignal:
                if label not in smhists:
                    smhists[label] = hist
                else:
                    smhists[label] += hist
            else:
                if (label, None, None) not in promotions:
                    promotions[(label, None, None)] = hist
                else:
                    promotions[(label, None, None)] += hist

        if args.doah:
            for key in directory.keys():
                if (match := signal_name_pat.match(key)) is not None:
                    parity = match.group(1)
                    mass = int(match.group(2))
                    width = float(match.group(3).replace("p", "."))
                    if width % 1 == 0:
                        width = int(width)

                    if len(args.assignal) > 0 and not parity in args.assignal:
                        continue

                    # hack to get around combine's behavior of signal POIs
                    if fit == "p" and args.ipf != "":
                        ipf = f"{os.path.dirname(args.ifile)}/ahtt_input.root" if args.ipf == 'default' else args.ipf
                        with uproot.open(f"{ipf}") as ipf:
                            hist = ipf[f"{channel}_{year}"][key].to_hist()[:len(centers)]
                            if "_neg" in key:
                                hist = -1. * hist
                    else:
                        hist = directory[key].to_hist()[:len(centers)]

                    isig = 0 if parity == 'A' else 1
                    if (parity, mass, width) in signals:
                        signals[(parity, mass, width)] += args.sigscale[isig] * hist
                    else:
                        signals[(parity, mass, width)] = args.sigscale[isig] * hist

            if args.total and len(signals) > 1 and len(promotions) == 0:
                signals[("Total", None, None)] = sum(signals.values()) if fit == "p" and args.ipf != "" else directory["total_signal"].to_hist()[:len(centers)]

        if args.total and len(signals) == 0 and len(promotions) > 1:
            signals[("Total", None, None)] = sum(promotions.values())

        if fit != "p":
            if gvalues_p is None:
                if args.poi == "fixed":
                    pstr = args.ifile.split("_fixed")[0].split("result_")[-1]
                    pstr = pstr.split("_")
                    gvalues_p = {}
                    for i in range(0, len(pstr), 2):
                        poiname = pstr[i]
                        print(poiname)
                        if poiname == "g1":
                            sigkey = [s for s in signals.keys() if s[0] == "A"][0]
                        elif poiname == "g2":
                            sigkey = [s for s in signals.keys() if s[0] == "H"][0]
                        else:
                            raise NotImplementedError()
                        gvalues_p[sigkey] = (float(pstr[i+1].replace("p", ".")),)
                else:
                    gvalues_p = get_poi_values(args.ifile, signals | promotions, args.poi,
                                            6.43 if args.xsec else 1, use_cross="cross" in args.poi)
            gvalues = gvalues_p
        else:
            gvalues = {}
        total = reduce(lambda a,b: a+b, smhists.values())
        #total = directory["total_background"].to_hist()[:len(centers)]

        if args.readbatch:
            with uproot.open(args.ifile) as ff:
                dd = ff[f"shapes_fit_{fit}/{channel}_{year}" if fit != "p" else f"shapes_prefit/{channel}_{year}"]
                datavalues = dd["data"].values()[1][:len(centers)]
                datahist_errors = np.array([dd["data"].errors("low")[1], dd["data"].errors("high")[1]])[:, :len(centers)]
        else:
            datavalues = directory["data"].values()[1][:len(centers)]
            datahist_errors = np.array([directory["data"].errors("low")[1], directory["data"].errors("high")[1]])[:, :len(centers)]

        kwargs = {
            "channel": channel,
            "year": year,
            "fit": fit,
            "smhists": smhists,
            "datavalues": datavalues,
            "total": total,
            "promotions": promotions,
            "signals": signals,
            "gvalues": gvalues,
            "sigscale": args.sigscale,
            "datahist_errors": datahist_errors,
            "binning": binning,
            "num_extrabins": num_extrabins,
            "extra_axes": extra_axes,
            "first_ax_binning": first_ax_binning,
            "first_ax_width": first_ax_width,
            "bins": bins,
            "centers": centers,
            "log": (args.logx, args.logy),
            "cuts": [""]
        }

        if args.each:
            plot(**kwargs)

        if args.batch is not None:
            if (channel, fit) in year_summed:
                this_year = year_summed[(channel, fit)]
                year_summed[(channel, fit)] = sum_kwargs(channel, "Run 2", kwargs, this_year)
            else:
                year_summed[(channel, fit)] = kwargs

batches = {
    r"$\ell\ell$": ["ee", "em", "mm"],
    #r"$\ell$j":    ["e4pj", "m4pj", "e3j", "m3j"],
    #r"ej":         ["e4pj", "e3j"],
    #r"mj":         ["m4pj", "m3j"],
    r"$\ell$, 3j":   ["e3j", "m3j"],
    r"$\ell$, $\geq$ 4j":  ["e4pj", "m4pj"],
}
if args.batch is not None:
    for cltx, cmrg in batches.items():
        for fit in fits:
            has_channel = all([(channel, fit) in year_summed for channel in cmrg])
            if not has_channel:
                continue

            sums = sum_kwargs(cltx, "Run 2", *(year_summed[(channel, fit)] for channel in cmrg))
            #if fit != 'p':
            #    if not os.path.isfile(args.batch):
            #        continue

            fitkey = "prefit" if fit == "p" else "postfit"
            if os.path.isfile(args.batch):
                with uproot.open(args.batch) as f:
                    has_psfromws = all([f"{channel}_{year}_{fitkey}" in f for channel in cmrg for year in years])
                    total = f[fitkey]["TotalBkg"].to_hist()[:len(year_summed[(cmrg[0], fit)]["datavalues"])]
                if not has_psfromws:
                    continue
    
                if fit != "p":
                    for promotion in sums["promotions"].values():
                        total.view().value -= promotion.values()
                    sums["total"] = total
                else:
                    sums["total"].view().variance = total.view().variance
            elif args.batch == "project":
                print("Getting uncertainty by projecting from fitdiagnostics")
                with uproot.open(args.ifile) as ff:
                    matrix = ff["shapes_prefit" if fit == "p" else f"shapes_fit_{fit}"]["overall_total_covar"].values()
                nbins = len(sums["total"].values())
                if nbins * len(cmrg) * len(years) != matrix.shape[0]:
                    raise NotImplementedError("Too stupid to project prefit covmat for subsets of channels")
                totalunc = np.diag(project_channel_year_unc(matrix, nbins))
                sums["total"].view().variance = totalunc
            else:
                raise ValueError("Unknown argument for --batch: " + args.batch)

            if args.normalize:
                with uproot.open(args.ifile) as ff:
                    matrix = ff["shapes_prefit" if fit == "p" else f"shapes_fit_{fit}"]["overall_total_covar"].values()
                    nbins = len(sums["total"].values())
                    matrix = project_channel_year_unc(matrix, nbins)
                sums = normalize(sums, matrix)
                args.ptag = "_norm" + args.ptag 

            if args.splitbins:
                plot_slices(sums)
                continue

            if args.project != "none":
                binedges = [bb for vv, bb in sums["binning"].items()]
                cut = args.cut if len(args.cut) == 2 else None
                if cut is None and (args.project == "mtt" or args.project == "mbbll"):
                    cut = [binedges[0][0], binedges[0][-1]]
                if cut is not None and cut[0] == -1:
                    cut[0] = binedges[0][0]
                if cut is not None and cut[1] == -1:
                    cut[1] = binedges[0][-1]
                matrix = None
                with uproot.open(args.ifile) as ff:
                    matrix = ff["shapes_prefit" if fit == "p" else f"shapes_fit_{fit}"]["overall_total_covar"].values()
                    if args.normalize:
                        overall_total = ff["shapes_prefit" if fit == "p" else f"shapes_fit_{fit}"]["total_overall"].values()
                        matrix = normalize_covariance(overall_total, matrix)
                        
                if cut is not None and all([binedges[0][0] <= cc <= binedges[0][-1] for cc in cut]):
                    plot_projection(sums, binedges, cut, matrix)
                else:
                    for imin in range(len(binedges[0]) - 1):
                        cut = [binedges[0][imin], binedges[0][imin + 1]]
                        plot_projection(sums, binedges, cut, matrix)
                continue
            plot(**sums)
