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
import ROOT
import math

from utilspy import tuplize
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes
from drawings import etat_blurb

parser = ArgumentParser()
parser.add_argument("--ifile", help = "input file ie fitdiagnostic results", default = "", required = True)
parser.add_argument("--ifileah", help = "input file ie fitdiagnostic results", default = "", required = True)
parser.add_argument("--lower", choices = ["ratio", "diff"], default = "ratio", required = False)
parser.add_argument("--log", action = "store_true", required = False)
parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False)
parser.add_argument("--plot-tag", help = "extra tag to append to plot names", dest = "ptag", default = "", required = False, type = prepend_if_not_empty)
#parser.add_argument("--skip-each", help = "skip plotting each channel x year combination", action = "store_false", dest = "each", required = False)
parser.add_argument("--batch", help = "psfromws output containing sums of channel x year combinations to be plotted. give any string to draw only batched prefit.",
                    default = None, dest = "batch", required = False)
parser.add_argument("--batchah", help = "psfromws output containing sums of channel x year combinations to be plotted. give any string to draw only batched prefit.",
                    default = None, dest = "batchah", required = False)
parser.add_argument("--prefit-signal-from", help = "read prefit signal templates from this file instead",
                    default = "", dest = "ipf", required = False)
parser.add_argument("--plot-formats", help = "comma-separated list of formats to save the plots in", default = [".png"], dest = "fmt", required = False,
                    type = lambda s: [prepend_if_not_empty(fmt, '.') for fmt in tokenize_to_list(remove_spaces_quotes(s))])
parser.add_argument("--signal-scale", help = "scaling to apply on A/H signal (ie not promoted ones (yet!)) in drawing", default = (1., 1.),
                    dest = "sigscale", required = False, type = lambda s: tuplize(s))
parser.add_argument("--preliminary", help="Write 'Preliminary' in caption", action="store_true")
args = parser.parse_args()

fits = ["p", "sah", "s"]

if args.batch is not None:
    if "_ll_" in args.batch:
        channels = ["ee", "em", "mm"]
        batches = {r"$\ell\bar{\ell}$": channels}
    elif "_l4pj_" in args.batch:
        channels = ["e4pj", "m4pj"]
        batches = {r"$\ell$, $\geq$ 4j": channels}
    elif "_l3j_" in args.batch:
        channels = ["e3j", "m3j"]
        batches = {r"$\ell$, 3j": channels}
    else:
        raise ValueError(f"Unknown batch: {args.batch}")
else:
    channels = ["ee", "em", "mm", "e4pj", "m4pj", "e3j", "m3j"]
    raise NotImplementedError()
years = ["2016pre", "2016post", "2017", "2018"]
sm_procs = {
    "TTV": r"Other",
    "VV": r"Other",
    "EWQCD": r"Other",
    "TW": "tX",
    "TB": "tX",
    "TQ": "tX",
    "DY": r"Other",
    "EtaT": r"$\eta_{\mathrm{t}}$",
    "TT": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_const_pos": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_const_neg": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_lin_pos": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_lin_neg": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_quad_pos": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_quad_neg": r"$\mathrm{t}\bar{\mathrm{t}}$",
}
proc_colors = {
    "tX": "C0", #"#5790fc",
    r"Other": "C1", #"#f89c20",
    r"$\mathrm{t}\bar{\mathrm{t}}$":  "#F3E5AB", #"#e42536",
    #r"Other": "#58A279", #"#964a8b",
    r"$\eta_{\mathrm{t}}$": "#009000",

    "A": "#cc0033",
    "H": "#0033cc",
    "Total": "#3B444B"
}
#if not args.doah:
#    proc_colors[r"$\eta_{\mathrm{t}}$"] = proc_colors['A']

signal_zorder = {
    r"$\eta_{\mathrm{t}}$": 0,
    "Total": 1,
    "A": 2,
    "H": 3
}
binnings = {
    ("ee", "em", "mm"): {
        r"$m_{\mathrm{t}\bar{\mathrm{t}}}$ (GeV)":
            [320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 845, 890, 935, 985, 1050, 1140, 1300, 1460],
        r"$c_{\mathrm{han}}$": ["-1", r"-$\frac{1}{3}$", r"$\frac{1}{3}$", "1"],
        r"$c_{\mathrm{hel}}$": ["-1", r"-$\frac{1}{3}$", r"$\frac{1}{3}$", "1"],
    },
    ("e4pj", "m4pj", "e3j", "m3j"): {
        r"$m_{\mathrm{t}\bar{\mathrm{t}}}$ (GeV)":
            [320, 360, 400, 440, 480, 520, 560, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100,  1150, 1200, 1300, 1500, 1700],
        r"$\left|\cos(\theta_{\mathrm{t}_{\ell}}^{*})\right|$": [0.0, 0.4, 0.6, 0.75, 0.9, 1.0],
    }
}
ratiolabels = {
    "b": "Ratio to background",
    "s": "Ratio to background",
    "p": "Ratio to background",
}
lumis = {
    "2016pre": "19.5",
    "2016post": "16.8",
    "2017": "41.5",
    "2018": "59.9",
    "Run 2": "138",
}
hatchstyle = dict(
    color = "black",
    alpha = 0.3,
    linewidth = 0,
)
datastyle = dict(
    marker = "o",
    markersize = 3,
    elinewidth = 0.75,
    linestyle = "none",
    color = "black"
)



def get_poi_values(fname, signals):
    signals = list(signals.keys())
    signals = [sig for sig in signals if sig != ("Total", None, None)]
    twing = len(signals) >= 2 and sum([1 if sig[0] == "A" or sig[0] == "H" else 0 for sig in signals]) == 2
    onepoi = "one-poi" in fname
    etat = r"$\eta_{\mathrm{t}}$" in [sig[0] for sig in signals]

    if not twing and not onepoi and not etat:
        raise NotImplementedError()

    ffile = ROOT.TFile.Open(fname, "read")
    if "fit_s" not in ffile.GetListOfKeys():
        return {sig: 0 for sig in signals}

    fres = ffile.Get("fit_s")
    result = {}
    if onepoi:
        gg = fres.floatParsFinal().find('g')
        result = {signals[0]: (round(gg.getValV(), 3), round(gg.getError(), 3))}
        #result = {signals[0]: (round(gg.getValV(), 3), round(gg.getAsymErrorLo(), 3), round(gg.getAsymErrorHi(), 3))}
    elif twing:
        g1 = fres.floatParsFinal().find('g1')
        g2 = fres.floatParsFinal().find('g2')
        
        result = {
            signals[0]: (round(g1.getValV(), 2), round(g1.getError(), 2)),
            signals[1]: (round(g2.getValV(), 2), round(g2.getError(), 2))
        }
        #result = {
        #    signals[0]: (round(g1.getValV(), 3), round(g1.getAsymErrorLo(), 3), round(g1.getAsymErrorHi(), 3)),
        #    signals[1]: (round(g2.getValV(), 3), round(g2.getAsymErrorLo(), 3), round(g2.getAsymErrorHi(), 3))
        #}
    if etat:
        muetat = fres.floatParsFinal().find('CMS_EtaT_norm_13TeV')
        result = result | {signals[0]: (round(muetat.getValV(), 3), round(muetat.getError(), 3))}
    return result



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
    if fit == "p":
        fstage = "Pre"
        ftype = " "
    else:
        fstage = "Post"
        ftype = " (s + b) " if fit == "s" else " (b) "

    width = np.diff(bins)
    colors = [proc_colors[k] for k in smhists.keys()]
    hep.histplot(
        [hist.values() / width for hist in smhists.values()],
        bins = bins,
        ax = ax,
        stack = True,
        histtype = "fill",
        label = smhists.keys(),
        color = colors
    )
    for ibin in range(len(bins) - 1):
        vhi = (total.values()[ibin] + total.variances()[ibin] ** .5) / width[ibin]
        vlo = (total.values()[ibin] - total.variances()[ibin] ** .5) / width[ibin]
        ax.fill_between(
            bins[ibin : ibin + 2],
            np.array(vhi, vhi),
            np.array(vlo, vlo),
            step = "mid",
            label = f"{fstage}fit{ftype}uncertainty" if ibin == 0 else None,
            **hatchstyle)
    ax.errorbar(
        centers,
        data[0] / width,
        yerr = data[1] / width,
        label = "Data",
        **datastyle
    )
    ax.set_ylabel("<Events / GeV>", fontsize=24)
    if log:
        ax.set_yscale("log")
        ymin = 0.5 * np.amin(data[0] / width)
        ymax = 1.08 if "j" in channel else 1.12
        ax.set_ylim(ymin, ax.transData.inverted().transform(ax.transAxes.transform([0, ymax]))[1])
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)



def plot_ratio(ax, bins, centers, data, total, signals, gvalues, sigscale, fit, fitstep):
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
        label = f"{fstage}-fit uncertainty",
        zorder = -99,
        **hatchstyle
    )
    handles_signals = []
    labels_signals = []
    for key, signal in signals.items():
        symbol, mass, decaywidth = key
        idx = 1 if symbol == 'H' else 0
        ss = int(sigscale[idx]) if abs(sigscale[idx] - int(sigscale[idx])) < 2.**-11 else sigscale[idx]
        signal_label = "" if abs(ss - 1.) < 2.**-11 else f"{ss} $\\times$ "
        if symbol == "Total":
            signal_label += "A + H"
        elif symbol == "A" or symbol == "H":
            signal_label = f"{symbol}({mass}, {decaywidth}%)"
        elif symbol == r"$\eta_{\mathrm{t}}$":
            signal_label = r"$\eta_{\mathrm{t}}$"
        else:
            signal_label = "Signal"

        if fit == "p":
            if symbol == "A" or symbol == "H":
                signal_label += f", $\\mathrm{{g}}_{{\\mathrm{{{symbol}}}}} = 1$"
            elif symbol == r"$\eta_{\mathrm{t}}$":
                signal_label = f"$\\eta_{{\\mathrm{{t}}}}$, $\\mu(\\eta_{{\\mathrm{{t}}}}) = 1$"
        elif key in gvalues and gvalues[key] is not None:
            if symbol == "A" or symbol == "H":
                if fit == "s":
                    signal_label += f", $\\mathrm{{g}}_{{\\mathrm{{{symbol}}}}} = {gvalues[key][0]} \\pm {gvalues[key][1]}$"
                    #signal_label += f", $\\mathrm{{g}}_{{\\mathrm{{{symbol}}}}} = {gvalues[key][0]}_{{-{gvalues[key][1]}}}^{{+{gvalues[key][2]}}}$"
                elif fit == "b":
                    signal_label += f", $\\mathrm{{g}}_{{\\mathrm{{{symbol}}}}} = 0$"
            elif symbol == r"$\eta_{\mathrm{t}}$":
                if fit == "s":
                    signal_label = f"$\\eta_{{\\mathrm{{t}}}}$, $\\mu(\\eta_{{\\mathrm{{t}}}}) = {gvalues[key][0]} \\pm {gvalues[key][1]}$"
                elif fit == "b":
                    signal_label = f"$\\eta_{{\\mathrm{{t}}}}$, $\\\mu(\\eta_{{\\mathrm{{t}}}}) = 0$"

        handle_signal = hep.histplot(
            (total.values() + signal.values()) / total.values(),
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
        handles_signals.append(handle_signal[0])
        labels_signals.append(signal_label)
    #for pos in [0.8, 0.9, 1.1, 1.2]:
    #    ax.axhline(y = pos, linestyle = ":", linewidth = 0.5, color = "black")
    if fitstep == "p":
        fittype = "Prefit"
    elif fitstep == "b":
        fittype = "Postfit (background only)"
    elif fitstep == "s":
        fittype = "Postfit (background + et)"
    elif fitstep == "sah":
        fittype = "Postfit (background + A/H)"
        #btxt = etat_blurb([sm_procs["EtaT"] in smhists])
        #fittype += "\n" + "\n".join(btxt)
    ax.axhline(y = 1, linestyle = "--", linewidth = 0.35, color = "black")
    if fit == "p":
        ax.set_ylim(0.79, 1.21)
        ax.set_yticks([0.8, 1.0, 1.2])
    else:
        ax.set_ylim(0.895, 1.105)
        ax.set_yticks([0.9, 1.0, 1.1])
    ax.set_ylabel(ratiolabels[fit], fontsize=24)
    handles = [Rectangle((0,0), 0, 0, facecolor="white", edgecolor="white", alpha=0.), *handles_signals, handle_unc]
    labels = [" "*len(fittype), *labels_signals, "Uncertainty"]
    ax.legend(handles=handles, labels=labels, loc = "lower left", bbox_to_anchor = (0, 1.0, 1, 0.2), borderaxespad = 0, ncol = 5, mode = "expand", fancybox = False).get_frame().set_edgecolor("black")



def plot_diff(ax, bins, centers, data, total, signals, gvalues, sigscale, fit):
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
            signal_label += "A + H"
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
            zorder = signal_zorder[symbol]
        )
    ax.set_ylabel("Difference to\nperturbative SM")
    #ax.axhline(y = 1, linestyle = "--", linewidth = 0.35, color = "black")
    ax.legend(loc = "lower left", bbox_to_anchor = (0, 1.0, 1, 0.2), borderaxespad = 0, ncol = 5, mode = "expand", fancybox = False).get_frame().set_edgecolor("black")



def plot(channel, year, fit,
         smhists, datavalues, total, promotions, signals, gvalues, sigscale, datahist_errors,
         binning, num_extrabins, extra_axes, first_ax_binning, first_ax_width, bins, centers, log,
         fig, ax0, ax1, ax2, fitstep):
    if len(smhists) == 0:
        return

    plotupper = fitstep == "p"
    allsigs = signals | promotions
    if plotupper:
        ax0.set_axis_off()
        plot_eventperbin(ax1, bins, centers, smhists, total, (datavalues, datahist_errors), log, fit, channel)
    if args.lower == "ratio":
        plot_ratio(ax2, bins, centers, (datavalues, datahist_errors), total, allsigs, gvalues, sigscale, fit, fitstep)
    elif args.lower == "diff":
        plot_diff(ax2, bins, centers, (datavalues, datahist_errors), total, allsigs, gvalues, sigscale, fit)
    else:
        raise ValueError(f"Invalid lower type: {args.lower}")
    for pos in bins[::len(first_ax_binning) - 1][1:-1]:
        if plotupper:
            ax1.axvline(x = pos, linestyle = "--", linewidth = 0.5, color = "gray")
        ax2.axvline(x = pos, linestyle = "--", linewidth = 0.5, color = "gray")
    if plotupper:
        for j, (variable, edges) in enumerate(extra_axes.items()):
            for i in range(num_extrabins):
                edge_idx = np.unravel_index(i, tuple(len(b) - 1 for b in extra_axes.values()))[j]
                text = r"{} < {} < {}".format(edges[edge_idx], variable, edges[edge_idx + 1])
                ax1.text(1 / num_extrabins * (i + 0.5), 0.912 - j * 0.11, text, horizontalalignment = "center", fontsize = 19, transform = ax1.transAxes)
        ax1.minorticks_on()
        ax1.tick_params(axis="both", which="both", direction="in", bottom=True, top=True, left=True, right=True)
    ax2.minorticks_on()
    ax2.tick_params(axis="both", which="both", direction="in", bottom=True, top=True, left=True, right=True)
    if "j" in channel:
        ticklocs = np.linspace(500, 1500, 3)
        ticklocs_minor = np.linspace(400, 1600, 13)
    else:
        ticklocs = np.linspace(600, 1200, 2)
        ticklocs_minor = np.linspace(450, 1350, 7)
    ticks = np.concatenate(
        [ticklocs - first_ax_binning[0] + i * first_ax_width
         for i in range(num_extrabins)])
    ticks_minor = np.concatenate(
        [ticklocs_minor - first_ax_binning[0] + i * first_ax_width
         for i in range(num_extrabins)])
    ax2.set_xticks(ticks, minor=False)
    ax2.set_xticks(ticks_minor, minor=True)
    noxaxis = fitstep != "s"
    if noxaxis:
        ax2.set_xticklabels([])
    else:       
        tickloc_labels = [f"{t:.0f}" for i in range(num_extrabins) for t in ticklocs]
        ax2.set_xticklabels(tickloc_labels)
    if plotupper:
        ax1.legend(loc = "lower left", bbox_to_anchor = (0, 1.0, 1, 0.2), borderaxespad = 0, ncol = len(smhists) + 2, mode = "expand", edgecolor = "black", framealpha = 1, fancybox = False)
        ax1.set_xlabel("")
    if noxaxis:
        ax2.set_xlabel("")
    else:
        ax2.set_xlabel(list(binning.keys())[0], fontsize=24)
    title = channel.replace('m', '$\\mu$').replace('4p', '4+')
    if fitstep == "p":
        fittype = "Prefit"
    elif fitstep == "b":
        fittype = "Postfit (BG only)"
    elif fitstep == "s":
        fittype = r"Postfit (BG + $\mathbf{\eta_{\mathrm{t}}}$)"
    elif fitstep == "sah":
        fittype = "Postfit (BG + A/H)"
        #btxt = etat_blurb([sm_procs["EtaT"] in smhists])
        #fittype += "\n" + "\n".join(btxt)
    ax2.annotate(fittype, (0.007, 1.038), xycoords="axes fraction", va="bottom", ha="left", fontsize=20, fontweight="bold", zorder=7777)
   
    if fitstep == "sah":
        btxt = "No $\\mathrm{t \\bar{t}}$ bound states"
        ax2.annotate(btxt, (0.007, 0.96), xycoords="axes fraction", va="top", ha="left", fontsize=20, zorder=7777)
        
    if plotupper:
        ax0.set_title(title)
    #if fitstep == "sah":
    #    btxt = etat_blurb([sm_procs["EtaT"] in smhists])
    #    ax2.annotate("\n".join(btxt), (0.01, 0.05), xycoords="axes fraction", va="bottom", ha="left", fontsize=15)

    if fitstep == "s":

        cmslabel = "Preliminary" if args.preliminary else None
        hep.cms.label(ax = ax0, data=True, label=cmslabel, lumi = lumis[year], loc = 0, year = year, fontsize = 24)
        fig.align_ylabels()
        fig.subplots_adjust(hspace = 0.24, left = 0.055, right = 1 - 0.003, top = 1 - 0.025, bottom = 0.045)
        

        bbox = ax0.get_position()
        offset = -0.005
        ax0.set_position([bbox.x0, bbox.y0 + offset, bbox.x1 - bbox.x0, bbox.y1 - bbox.y0])
        #fig.set_size_inches(w = 19.2, h = 1.5 * fig.get_figheight())

        #fig.set_dpi(450)
        #extent = 'tight'

        sstr = "Combined"
        cstr = channel.replace(r'$\ell\bar{\ell}$', 'll').replace(r'$\ell$j', 'lj').replace(r'$\ell$, 3j', 'l3j').replace(r'$\ell$, $\geq$ 4j', 'l4pj')
        ystr = year.replace(" ", "").lower()
        for fmt in args.fmt:
            fig.savefig(f"{args.odir}/{sstr}{args.ptag}_fit_{fit}_{cstr}_{ystr}{fmt}", transparent = True)#, bbox_inches = extent)
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



gvalues_ah = None
gvalues_etat = None

signal_name_pat = re.compile(r"(A|H)_m(\d+)_w(\d+p?\d*)_")
year_summed = {}
feta = uproot.open(args.ifile)
fah = uproot.open(args.ifileah)
print("--- Collecting data ---")
for channel, year in product(channels, years):
    for fitstep in fits:
        print(f"Channel {channel}, Year {year}, Fit step {fitstep}")
        fit = "s" if fitstep == "sah" else fitstep
        f = fah if fitstep == "sah" else feta
        ifile = args.ifileah if fitstep == "sah" else args.ifile
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

        assignal = ["EtaT"] if fitstep != "sah" else []
        directory = f[dname]
        smhists = {}
        signals = {}
        promotions = {}
        for proc, label in sm_procs.items():
            if proc not in directory:
                continue

            # combine hack, see args.doah block
            if fit == "p" and args.ipf != "" and proc == "EtaT":
                ipf = f"{os.path.dirname(ifile)}/ahtt_input.root" if args.ipf == 'default' else args.ipf
                with uproot.open(f"{ipf}") as ipf:
                    hist = ipf[f"{channel}_{year}"][proc].to_hist()[:len(centers)]
            else:
                hist = directory[proc].to_hist()[:len(centers)]

            if proc not in assignal:
                if label not in smhists:
                    smhists[label] = hist
                else:
                    smhists[label] += hist
            else:
                if (label, None, None) not in promotions:
                    promotions[(label, None, None)] = hist
                else:
                    promotions[(label, None, None)] += hist

        if fitstep == "sah" or fitstep == "p":
            for key in directory.keys():
                if (match := signal_name_pat.match(key)) is not None:
                    mass = int(match.group(2))
                    width = float(match.group(3).replace("p", "."))
                    if width % 1 == 0:
                        width = int(width)

                    # hack to get around combine's behavior of signal POIs
                    if fit == "p" and args.ipf != "":
                        ipf = f"{os.path.dirname(ifile)}/ahtt_input.root" if args.ipf == 'default' else args.ipf
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

            #if len(signals) > 1 and len(promotions) == 0:
            #    signals[("Total", None, None)] = sum(signals.values()) if fit == "p" and args.ipf != "" else directory["total_signal"].to_hist()[:len(centers)]

        if fit != "p":
            if fitstep == "sah":
                if gvalues_ah is None:
                    gvalues_ah = get_poi_values(ifile, signals | promotions)
                gvalues = gvalues_ah
            elif fitstep == "s":
                if gvalues_etat is None:
                    gvalues_etat = get_poi_values(ifile, signals | promotions)
                gvalues = gvalues_etat
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
            "fitstep": fitstep
        }

        #if args.each:
        #    plot(**kwargs)

        if args.batch is not None:
            if (channel, fitstep) in year_summed:
                this_year = year_summed[(channel, fitstep)]
                year_summed[(channel, fitstep)] = sum_kwargs(channel, "Run 2", kwargs, this_year)
            else:
                year_summed[(channel, fitstep)] = kwargs

feta.close()
fah.close()

print("--- Plotting ---")
if args.batch is not None:
    for cltx, channels in batches.items():
        fig, axes = plt.subplots(
            nrows = 5,
            sharex = True,
            gridspec_kw = {"height_ratios": [0.001, 1, 1, 1, 1]},
            figsize = (19.2, 17.75),
            dpi=600
        )
        for fitstep in fits:
            print(f"Channel {cltx}, fitstep {fitstep}")
            has_channel = all([(channel, fitstep) in year_summed for channel in channels])
            if not has_channel:
                continue

            sums = sum_kwargs(cltx, "Run 2", *(year_summed[(channel, fitstep)] for channel in channels))
            if fitstep != 'p':
                batchpath = args.batchah if fitstep == "sah" else args.batch
                if not os.path.isfile(batchpath):
                    continue

                with uproot.open(batchpath) as f:
                    has_psfromws = all([f"{channel}_{year}_postfit" in f for channel in channels for year in years])
                    total = f["postfit"]["TotalBkg"].to_hist()[:len(year_summed[(channels[0], fit)]["datavalues"])]

                if not has_psfromws:
                    continue

                for promotion in sums["promotions"].values():
                    total.view().value -= promotion.values()
                sums["total"] = total

            #print(cltx, channels, fit)
            #print(sums["datavalues"])
            #print("\n", flush = True)
            if fitstep == "p":
                lower_ax = axes[2]
            elif fitstep == "sah":
                lower_ax = axes[3]
            elif fitstep == "s":
                lower_ax = axes[4]
            sums.update({
                "fig": fig,
                "ax0": axes[0],
                "ax1": axes[1],
                "ax2": lower_ax,
            })
            plot(**sums)
