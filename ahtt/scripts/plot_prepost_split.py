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
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle

import uproot  # noqa:E402
import mplhep as hep  # noqa:E402
import hist  # noqa:E402
from hist import Hist  # noqa:E402
import ROOT
import math

from utilspy import tuplize, index_1n, index_n1
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes
from drawings import etat_blurb, channels, years, sm_procs, proc_colors, signal_zorder, binnings, ratiolabels, lumis, hatchstyle, datastyle
from drawings import get_poi_values

parser = ArgumentParser()
parser.add_argument("--ifile", help = "input file ie fitdiagnostic results", default = "", required = True)
parser.add_argument("--lower", choices = ["ratio", "diff"], default = "ratio", required = False)
parser.add_argument("--log", action = "store_true", required = False)
parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False)
parser.add_argument("--plot-tag", help = "extra tag to append to plot names", dest = "ptag", default = "", required = False, type = prepend_if_not_empty)
parser.add_argument("--skip-each", help = "skip plotting each channel x year combination", action = "store_false", dest = "each", required = False)
parser.add_argument("--batch", help = "psfromws output containing sums of channel x year combinations to be plotted. give any string to draw only batched prefit.",
                    default = None, dest = "batch", required = False)
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
parser.add_argument("--preliminary", help="Write 'Preliminary' in caption", action="store_true")
parser.add_argument("--only-res", dest="onlyres", help="Resonance-only mode", action="store_true")
parser.add_argument("--split-bins", help="Split the angle/spin bins", dest="splitbins", action="store_true")
parser.add_argument("--project-to", help = "which variables to project down to, and draw the 1D plots of. implemented only for batch plotting atm.",
                    dest = "project", choices = ["none", "mtt", "mbbll", "chel", "chan"], default = "none", required = False)
parser.add_argument("--mass-cut", help = "comma-separated minmax value, to cut on mtt. must be sorted, otherwise applies no cut. if one value is outside mass range, plots binwise.",
                    dest = "cut", default = "", required = False,
                    type = lambda s: [] if s == "" else sorted(tokenize_to_list(remove_spaces_quotes(s), astype = int)))

args = parser.parse_args()

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

def plot_eventperbin(ax, bins, centers, smhists, total, data, log, fit, channel):
    single_slice = args.splitbins or args.project != "none"
    angular = args.project in ["chel", "chan"]
    factor = 1000. if angular else 1.
    if fit == "p":
        fstage = "Pre"
        ftype = " "
    else:
        fstage = "Post"
        ftype = " (s + b) " if fit == "s" else " (b) "

    width = np.diff(bins)
    colors = [proc_colors[k] for k in smhists.keys()]
    unclabel = "Unc." if single_slice else f"{fstage}fit{ftype}uncertainty"
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
        **datastyle
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
    ax.set_ylabel("<Events> / $10^{3}$" if angular else "<Events / GeV>", fontsize=24)
    if log:
        ax.set_yscale("log")
        ymin = 0.5 * np.amin(data[0] / width / factor)
        if args.splitbins:
            ymax = 1.05
        else:
            ymax = 1.08 if "j" in channel else 1.12
        ax.set_ylim(ymin, ax.transData.inverted().transform(ax.transAxes.transform([0, ymax]))[1])
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)



def plot_ratio(ax, bins, centers, data, total, signals, gvalues, sigscale, fit):
    single_slice = args.splitbins or args.project != "none"
    ax.errorbar(
        centers,
        data[0] / total.values(),
        data[1] / total.values(),
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
            elif symbol == r"$\eta_{\mathrm{t}}$":
                signal_label = f"$\\eta_{{\\mathrm{{t}}}}$, $\\mu(\\eta_{{\\mathrm{{t}}}}) = 1$"
            elif symbol == r"$\chi_{\mathrm{t}}$":
                signal_label = f"$\\chi_{{\\mathrm{{t}}}}$, $\\mu(\\chi_{{\\mathrm{{t}}}}) = 1$"
            elif symbol == r"$\psi_{\mathrm{t}}$":
                signal_label = f"$\\psi_{{\\mathrm{{t}}}}$, $\\mu(\\psi_{{\\mathrm{{t}}}}) = 1$"
        elif key in gvalues and gvalues[key] is not None:
            if symbol == "A" or symbol == "H":
                if fit == "s":
                    if len(gvalues[key]) == 2:
                        signal_label += f", $\\mathrm{{{poiname}}}_{{\\mathrm{{{symbol}}}}} = {gvalues[key][0]:.2f} \\pm {gvalues[key][1]:.2f}$"
                    else:
                        signal_label += f", $\\mathrm{{g}}_{{\\mathrm{{{symbol}}}}} = {gvalues[key][0]}_{{-{gvalues[key][2]}}}^{{+{gvalues[key][1]}}}$"
                elif fit == "b":
                    signal_label += f", $\\mathrm{{{poiname}}}_{{\\mathrm{{{symbol}}}}} = 0$"
            elif symbol == r"$\eta_{\mathrm{t}}$":
                if fit == "s":
                    if len(gvalues[key]) == 2:
                        signal_label = f"$\\eta_{{\\mathrm{{t}}}}$, $\\mu(\\eta_{{\\mathrm{{t}}}}) = {gvalues[key][0]:.2f} \\pm {gvalues[key][1]:.2f}$"
                    else:
                        signal_label = f"$\\eta_{{\\mathrm{{t}}}}$, $\\mu(\\eta_{{\\mathrm{{t}}}}) = {gvalues[key][0]:.2f}_{{-{gvalues[key][2]:.2f}}}^{{+{gvalues[key][1]:.2f}}}$"
                elif fit == "b":
                    signal_label = f"$\\eta_{{\\mathrm{{t}}}}$, $\\\mu(\\eta_{{\\mathrm{{t}}}}) = 0$"
            elif symbol == r"$\chi_{\mathrm{t}}$":
                if fit == "s":
                    if len(gvalues[key]) == 2:
                        signal_label = f"$\\chi_{{\\mathrm{{t}}}}$, $\\mu(\\chi_{{\\mathrm{{t}}}}) = {gvalues[key][0]:.2f} \\pm {gvalues[key][1]:.2f}$"
                    else:
                        signal_label = f"$\\chi_{{\\mathrm{{t}}}}$, $\\mu(\\chi_{{\\mathrm{{t}}}}) = {gvalues[key][0]:.2f}_{{-{gvalues[key][2]:.2f}}}^{{+{gvalues[key][1]:.2f}}}$"
                elif fit == "b":
                    signal_label = f"$\\chi_{{\\mathrm{{t}}}}$, $\\\mu(\\chi_{{\\mathrm{{t}}}}) = 0$"
            elif symbol == r"$\psi_{\mathrm{t}}$":
                if fit == "s":
                    if len(gvalues[key]) == 2:
                        signal_label = f"$\\psi_{{\\mathrm{{t}}}}$, $\\mu(\\psi_{{\\mathrm{{t}}}}) = {gvalues[key][0]:.2f} \\pm {gvalues[key][1]:.2f}$"
                    else:
                        signal_label = f"$\\psi_{{\\mathrm{{t}}}}$, $\\mu(\\psi_{{\\mathrm{{t}}}}) = {gvalues[key][0]:.2f}_{{-{gvalues[key][2]:.2f}}}^{{+{gvalues[key][1]:.2f}}}$"
                elif fit == "b":
                    signal_label = f"$\\psi_{{\\mathrm{{t}}}}$, $\\\mu(\\psi_{{\\mathrm{{t}}}}) = 0$"

        handle_signal = hep.histplot(
            (total.values() + signal.values()) / total.values(),
            bins = bins,
            yerr = np.zeros(len(signal.axes[0])),
            ax = ax,
            histtype = "step",
            color = proc_colors[symbol],
            linewidth = 1.75,
            label = signal_label,
            zorder = signal_zorder[symbol]
        )
        handles.append(handle_signal[0])
        labels.append(signal_label)
    #for pos in [0.8, 0.9, 1.1, 1.2]:
    #    ax.axhline(y = pos, linestyle = ":", linewidth = 0.5, color = "black")
    ax.axhline(y = 1, linestyle = "--", linewidth = 0.35, color = "black")
    if fit == "p":
        ax.set_ylim(0.79, 1.21)
        ax.set_yticks([0.8, 1.0, 1.2])
    else:
        ax.set_ylim(0.895, 1.105)
        ax.set_yticks([0.9, 1.0, 1.1])
    ax.set_ylabel(ratiolabels[fit], fontsize=24)
    if fit == "p":
        fittype = "Prefit"
    elif fit == "b":
        fittype = "Postfit (BG only)"
    elif fit == "s" and len(args.assignal):
        fittype = "Postfit (BG "
        if "EtaT" in args.assignal:
            fittype += "+ $\mathbf{\eta_{\mathrm{t}}}$"
        if "ChiT" in args.assignal:
            fittype += "+ $\mathbf{\chi_{\mathrm{t}}}$"
        if "PsiT" in args.assignal:
            fittype += "+ $\mathbf{\psi_{\mathrm{t}}}$"
        fittype += ")"
    else:
        fittype = "Postfit (BG + A/H)"
    if not single_slice:
        handles.insert(0, Rectangle((0,0), 0, 0, facecolor="white", edgecolor="white", alpha=0.))
        labels.insert(0, " "*len(fittype))
    if not single_slice or len(handles) < 2:
        handles.append(handle_unc)
        labels.append("Uncertainty")
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
    allsigs = signals | promotions
    if args.panel == "both":
        fig, (ax0, ax1, ax2) = plt.subplots(
            nrows = 3,
            sharex = True,
            gridspec_kw = {"height_ratios": [0.105 if single_slice else 0.001, 1, 0.9]},
            figsize = (5.0, 7.0) if single_slice else (19.2, 6.6),
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
        plot_eventperbin(ax1, bins, centers, smhists, total, (datavalues, datahist_errors), log, fit, channel)
    if args.panel != "upper":
        if args.lower == "ratio":
            plot_ratio(ax2, bins, centers, (datavalues, datahist_errors), total, allsigs, gvalues, sigscale, fit)
        elif args.lower == "diff":
            plot_diff(ax2, bins, centers, (datavalues, datahist_errors), total, allsigs, gvalues, sigscale, fit)
        else:
            raise ValueError(f"Invalid lower type: {args.lower}")
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
    ax2.set_xticks(ticks if first_ax_width > 0 else [-1, 0, 1], minor=False)
    ax2.set_xticks(ticks_minor if first_ax_width > 0 else [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], minor=True)
    if args.noxaxis:
        ax2.set_xticklabels([])
        ax2.set_xlabel("")
    else:       
        tickloc_labels = [f"{t:.0f}" for i in range(num_extrabins) for t in ticklocs] if first_ax_width > 0 else ["-1", "0", "1"]
        ax2.set_xticklabels(tickloc_labels)
        ax2.set_xlabel(list(binning.keys())[0], fontsize=24)

    title = channel.replace('m', '$\\mu$').replace('4p', '4+')
    if fit == "p":
        fittype = "Prefit"
    elif fit == "b":
        fittype = "Postfit (BG only)"
    elif fit == "s" and len(args.assignal):
        fittype = "Postfit (BG "
        if "EtaT" in args.assignal:
            fittype += "+ $\mathbf{\eta_{\mathrm{t}}}$"
        if "ChiT" in args.assignal:
            fittype += "+ $\mathbf{\chi_{\mathrm{t}}}$"
        if "PsiT" in args.assignal:
            fittype += "+ $\mathbf{\psi_{\mathrm{t}}}$"
        fittype += ")"
    else:
        fittype = "Postfit (BG + A/H)"
    
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
            annstr = fittype + ", " + title
        else:
            annstr = fittype
        ax2.annotate(annstr, (xpos, ypos), xycoords="axes fraction", va=va, ha=ha, fontsize=20, fontweight="normal" if (single_slice and args.panel == "upper") else "bold", zorder=7777)
        if args.panel == "both":
            ax1.annotate(title, (0.97, 1.028), xycoords="axes fraction", va="bottom", ha="right", fontsize=20, fontweight="normal", zorder=7777) # xxx
    elif args.panel != "upper":
        ax2.annotate(fittype, (0.007, 1.038), xycoords="axes fraction", va="bottom", ha="left", fontsize=20, fontweight="bold", zorder=7777)

    if args.panel != "upper" and not any([ss in ["EtaT", "ChiT", "PsiT"] for ss in args.assignal]):
        #btxt = etat_blurb([sm_procs["EtaT"] in smhists])
        btxt = "No $\\mathrm{t \\bar{t}}$ bound states"
        xpos = 0.04 if single_slice else 0.01
        ypos = 0.86 if args.panellabels else 0.96
        ax2.annotate(btxt, (xpos, ypos), xycoords="axes fraction", va="top", ha="left", fontsize=20, zorder=7777)

    cmslabel = "Preliminary" if args.preliminary else None
    if args.panel == "both":
        if not single_slice:
            ax0.set_title(title)
        hep.cms.label(ax = ax0, data=True, label=cmslabel, lumi = lumis[year], loc = 0, year = year, fontsize = 19 if args.preliminary else 24)
        fig.subplots_adjust(hspace = 0.24, left = 0.055, right = 1 - 0.003, top = 1 - 0.05)
    else:
        hep.cms.label(ax = ax0, data=True, label=cmslabel, lumi = lumis[year], loc = 0, year = year, fontsize = 19 if args.preliminary else 24)
        hspace = 0.81 if args.panel == "lower" and len(args.assignal) >= 2 else 0.50
        fig.subplots_adjust(hspace = hspace, left = 0.075, right = 1 - 0.025, top = 1 - 0.075)

    bbox = ax2.get_position()
    offset = -0.025 if single_slice and args.panel == "lower" else -0.02 if single_slice and args.panel == "upper" else -0.037
    ax2.set_position([bbox.x0, bbox.y0 + offset, bbox.x1 - bbox.x0, bbox.y1 - bbox.y0])
    figwidth = 6.0 if single_slice else 19.2
    if args.panel == "both":
        fig.set_size_inches(w = figwidth, h = 1.5 * fig.get_figheight())
    else:
        fig.set_size_inches(w = figwidth, h = 1.0 * fig.get_figheight())
    extent = 'tight'# if args.plotupper else full_extent(ax2).transformed(fig.dpi_scale_trans.inverted())

    sstr = [ss for ss in allsigs.keys() if ss[0] != "Total"]
    # FIXME brittle! assumes exclusive toponia vs A/H
    if any([ss in sstr[0][0] for ss in ["eta", "chi", "psi"]]):
        sstr = "__".join(args.assignal)
    else:
        sstr = [ss[0] + "_m" + str(ss[1]) + "_w" + str(float(ss[2])).replace(".", "p") for ss in sstr if ss[0] != r"$\eta_{\mathrm{t}}$"]
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
                if args.splitbins:
                    for j, (variable, edges) in enumerate(extra_axes.items()):
                        edge_idx = np.unravel_index(i, tuple(len(b) - 1 for b in extra_axes.values()))[j]
                        text = r"{} < {} < {}".format(edges[edge_idx], variable, edges[edge_idx + 1])
                        bintexts.append(ax1.text(1 / len(extra_axes) * (j + 0.5), ypos, text, horizontalalignment = "center", fontsize = 19, transform = ax1.transAxes))
                else:
                    bintexts.append(ax1.text(1 / len(extra_axes) * 0.5, ypos, cuts[1], horizontalalignment = "center", fontsize = 19, transform = ax1.transAxes))
                if first_ax_width > 0:
                    ax2.set_xlim(first_ax_width*i, first_ax_width*(i+1))
                else:
                    ax2.set_xlim(-1, 1)
                if args.splitbins:
                    fig.savefig(f"{args.odir}/{sstr}{args.ptag}_fit_{fit}_{cstr}_{ystr}_{args.panel}_bin{i+1}{fmt}", transparent = True, bbox_inches = extent)
                else:
                    fig.savefig(f"{args.odir}/{sstr}{args.ptag}_fit_{fit}_{cstr}_{ystr}_{args.panel}_{args.project}{cuts[0]}{fmt}", transparent = True, bbox_inches = extent)
        else:
            if args.panel != "lower":
                for j, (variable, edges) in enumerate(extra_axes.items()):
                    for i in range(num_extrabins):
                        edge_idx = np.unravel_index(i, tuple(len(b) - 1 for b in extra_axes.values()))[j]
                        text = r"{} < {} < {}".format(edges[edge_idx], variable, edges[edge_idx + 1])
                        if not single_slice:
                            ax1.text(1 / num_extrabins * (i + 0.5), 0.912 - j * 0.11, text, horizontalalignment = "center", fontsize = 19, transform = ax1.transAxes)
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

def actually_project(plane, nbins, target, cut, icut, matrix):
    isarray = isinstance(plane, np.ndarray)
    values = np.zeros(nbins[target])
    variances = None if isarray else np.zeros(nbins[target])
    for ibin in range(math.prod(nbins)):
        idxs = index_1n(ibin, nbins)
        if icut[0] <= idxs[0] < icut[1]:
            values[ idxs[target] ] += plane[ibin] if isarray else plane.values()[ibin]
            if not isarray:
                nchannel = int(len(matrix) / len(years) / math.prod(nbins))
                for iyear in range(len(years)):
                    for ichannel in range(nchannel):
                        iycb0 = index_n1(idxs + [iyear, ichannel], nbins + [len(years), nchannel])
                        for iycb1 in range(len(matrix)):
                            iother = index_1n(iycb1, nbins + [len(years), nchannel])
                            if iother[target] != idxs[target]:
                                continue
                            if icut[0] <= iother[0] < icut[1]:
                                variances[ idxs[target] ] += matrix[iycb0, iycb1]
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
        r"$m_{\mathrm{b}\bar{\mathrm{b}}\ell\bar{\ell}}$": "mbbll"
    }
    target = targets.get(target, 0)
    ret = planes.copy()

    ret["smhists"] = {k: actually_project(ret["smhists"][k], nbins, target, cut, icut, matrix) for k in ret["smhists"]}
    ret["datavalues"] = actually_project(ret["datavalues"], nbins, target, cut, icut, matrix)
    ret["total"] = actually_project(ret["total"], nbins, target, cut, icut, matrix)
    ret["promotions"] = {k: actually_project(ret["promotions"][k], nbins, target, cut, icut, matrix) for k in ret["promotions"]}
    ret["signals"] = {k: actually_project(ret["signals"][k], nbins, target, cut, icut, matrix) for k in ret["signals"]}
    dataerr_lo = (actually_project(ret["datahist_errors"][0]**2, nbins, target, cut, icut, matrix))**.5
    dataerr_hi = (actually_project(ret["datahist_errors"][1]**2, nbins, target, cut, icut, matrix))**.5
    ret["datahist_errors"] = np.array([dataerr_lo, dataerr_hi])
    ret["binning"] = {list(planes["binning"].keys())[target] : list(planes["binning"].values())[target] if target == 0 else [-1, -1/3, 1/3, 1]}
    ret["num_extrabins"] = int(np.prod(list(len(edges) - 1 for edges in list(ret["binning"].values())[1:])))
    ret["extra_axes"] = {'none': list(ret["binning"].values())[0] if target == 0 else [-1, -1/3, 1/3, 1]}
    ret["first_ax_binning"] = list(ret["binning"].values())[0] if target == 0 else np.array([-1, -1/3, 1/3, 1])
    ret["first_ax_width"] = ret["first_ax_binning"][-1] - ret["first_ax_binning"][0] if target == 0 else 0
    binwidths = np.diff(ret["first_ax_binning"])
    ret["bins"] = ret["first_ax_binning"]
    ret["centers"] = (ret["bins"][1:] + ret["bins"][:-1]) / 2
    ret["cuts"] = cut_string(
        list(planes["binning"].keys())[0].replace(" (GeV)", ""),
        list(planes["binning"].values())[0],
        cut,
        masses[ list(planes["binning"].keys())[0].replace(" (GeV)", "") ]
    )
    return ret

def plot_projection(sums, binedges, cut, matrix):
    icut = [binedges[0].index(cc) for cc in cut]
    nbins = [len(bb) - 1 for bb in binedges]
    sums = project(sums, nbins, args.project, cut, icut, matrix)
    plot(**sums)



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

gvalues_p = None

signal_name_pat = re.compile(r"(A|H)_m(\d+)_w(\d+p?\d*)_")
year_summed = {}
with uproot.open(args.ifile) as f:
    for channel, year, fit in product(channels, years, fits):
        for binning_channels, binning in binnings.items():
            if channel in binning_channels:
                break
        num_extrabins = np.prod(list(len(edges) - 1 for edges in list(binning.values())[1:]))
        extra_axes = {k: v for i, (k, v) in enumerate(binning.items()) if i != 0}
        first_ax_binning = list(binning.values())[0]
        first_ax_width = first_ax_binning[-1] - first_ax_binning[0]
        binwidths = np.diff(first_ax_binning)
        bins = (np.cumsum(binwidths)[None] + (np.arange(num_extrabins) * first_ax_width)[:, None]).flatten()
        bins = np.r_[0, bins]
        centers = (bins[1:] + bins[:-1]) / 2

        dname = f"shapes_fit_{fit}/{channel}_{year}" if fit != "p" else f"shapes_prefit/{channel}_{year}"

        if dname not in f:
            continue

        directory = f[dname]
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
                    mass = int(match.group(2))
                    width = float(match.group(3).replace("p", "."))
                    if width % 1 == 0:
                        width = int(width)

                    # hack to get around combine's behavior of signal POIs
                    if fit == "p" and args.ipf != "":
                        ipf = f"{os.path.dirname(args.ifile)}/ahtt_input.root" if args.ipf == 'default' else args.ipf
                        with uproot.open(f"{ipf}") as ipf:
                            hist = ipf[f"{channel}_{year}"][key].to_hist()[:len(centers)]
                            if "_neg" in key:
                                hist = -1. * hist
                    else:
                        hist = directory[key].to_hist()[:len(centers)]

                    isig = 0 if match.group(1) == 'A' else 1
                    if (match.group(1), mass, width) in signals:
                        signals[(match.group(1), mass, width)] += args.sigscale[isig] * hist
                    else:
                        signals[(match.group(1), mass, width)] = args.sigscale[isig] * hist

            if len(signals) > 1 and len(promotions) == 0:
                signals[("Total", None, None)] = sum(signals.values()) if fit == "p" and args.ipf != "" else directory["total_signal"].to_hist()[:len(centers)]

        if len(signals) == 0 and len(promotions) > 1:
            signals[("Total", None, None)] = sum(promotions.values())

        if fit != "p":
            if gvalues_p is None:
                gvalues_p = get_poi_values(args.ifile, signals | promotions, args.poi)
            gvalues = gvalues_p
        else:
            gvalues = {}
        datavalues = directory["data"].values()[1][:len(centers)]
        #total = directory["total_background"].to_hist()[:len(centers)]
        total = reduce(lambda a,b: a+b, smhists.values())

        # FIXME actual error: axes not mergable error when reading EtaT from args.ipf (needed because muetat = 0 at prefit, so hist = 0)
        #if fit != 'p':
        #    for promotion in promotions.values():
        #        total += -1. * promotion
        #covariance = directory["total_covar"].to_hist()[:len(centers), :len(centers)]
        #total = add_covariance(total, covariance)
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
            "log": args.log,
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
            if fit != 'p':
                if not os.path.isfile(args.batch):
                    continue

                with uproot.open(args.batch) as f:
                    has_psfromws = all([f"{channel}_{year}_postfit" in f for channel in cmrg for year in years])
                    total = f["postfit"]["TotalBkg"].to_hist()[:len(year_summed[(cmrg[0], fit)]["datavalues"])]
                if not has_psfromws:
                    continue

                for promotion in sums["promotions"].values():
                    total.view().value -= promotion.values()
                sums["total"] = total

                if args.project != "none":
                    binedges = [list(bb.values()) for cc, bb in binnings.items() if cmrg[0] in cc][0]
                    cut = args.cut if len(args.cut) == 2 and sorted(args.cut) else [binedges[0][0], binedges[0][-1]]
                    matrix = None
                    with uproot.open(args.ifile) as ff:
                        matrix = ff[f"shapes_fit_{fit}"]["overall_total_covar"].values()
                    if all([binedges[0][0] <= cc <= binedges[0][1] for cc in cut]):
                        plot_projection(sums, binedges, cut, matrix)
                    else:
                        for imin in range(len(binedges[0]) - 1):
                            cut = [binedges[0][imin], binedges[0][imin + 1]]
                            plot_projection(sums, binedges, cut, matrix)
                    continue
            plot(**sums)
