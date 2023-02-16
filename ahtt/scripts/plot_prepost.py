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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa:E402
plt.rcParams['axes.xmargin'] = 0

import uproot  # noqa:E402
import mplhep as hep  # noqa:E402

import ROOT
from ROOT import TFile

from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes

parser = ArgumentParser()
parser.add_argument("--ifile", help = "input file", default = "", required = True)
parser.add_argument("--lower", choices = ["ratio", "diff"], default = "diff", required = False)
parser.add_argument("--log", action = "store_true", required = False)
parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False)
parser.add_argument("--plot-tag", help = "extra tag to append to plot names", dest = "ptag",
                    default = "", required = False, type = prepend_if_not_empty)
parser.add_argument("--each", help = "plot also each channel x year combination", action = "store_true", required = False)
parser.add_argument("--only-lower", help = "dont plot the top panel", dest = "plotupper", action = "store_false", required = False)
parser.add_argument("--plot-format", help = "format to save the plots in", default = ".png", dest = "fmt", required = False, type = lambda s: prepend_if_not_empty(s, '.'))
args = parser.parse_args()


channels = ["ee", "em", "mm", "e4pj", "m4pj", "e3j", "m3j"]
years = ["2016pre", "2016post", "2017", "2018"]
fits = ["s", "b", "p"]
sm_procs = {
    "TTV": r"$\mathrm{VX}$, $\mathrm{t}\bar{\mathrm{t}}\mathrm{V}$",
    "VV": r"$\mathrm{VX}$, $\mathrm{t}\bar{\mathrm{t}}\mathrm{V}$",
    "EWQCD": r"$\mathrm{VX}$, $\mathrm{t}\bar{\mathrm{t}}\mathrm{V}$, QCD",
    "TW": "tX",
    "TB": "tX",
    "TQ": "tX",
    "DY": r"$\mathrm{VX}$, $\mathrm{t}\bar{\mathrm{t}}\mathrm{V}$",
    "TT": r"$\mathrm{t}\bar{\mathrm{t}}$"
}
sm_colors = {
    "tX": "C0",
    r"$\mathrm{VX}$, $\mathrm{t}\bar{\mathrm{t}}\mathrm{V}$": "C1",
    r"$\mathrm{t}\bar{\mathrm{t}}$": "#F3E5AB",
    r"$\mathrm{VX}$, $\mathrm{t}\bar{\mathrm{t}}\mathrm{V}$, QCD": "C1",
}
signal_colors = {
    "A": "#cc0033",
    "H": "#0063ab",
    "Total": "#3B444B"
}
signal_zorder = {
    "Total": 0,
    "A": 1,
    "H": 2,
}
binnings = {
    ("ee", "em", "mm"): {
        r"$m_{\mathrm{t}\bar{\mathrm{t}}}$ (GeV)":
            [320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 845, 890, 935, 985, 1050, 1140, 1300, 1700],
        r"$c_{\mathrm{han}}$": ["-1", r"-1/3", r"1/3", "1"],
        r"$c_{\mathrm{hel}}$": ["-1", r"-1/3", r"1/3", "1"],
    },
    ("e4pj", "m4pj", "e3j", "m3j"): {
        r"$m_{\mathrm{t}\bar{\mathrm{t}}}$ (GeV)":
            [320, 360, 400, 440, 480, 520, 560, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100,  1150, 1200, 1300, 1500, 2000],
        r"$\left|\cos(\theta_{\mathrm{t}_{\ell}}^{*})\right|$": [0.0, 0.4, 0.6, 0.75, 0.9, 1.0],
    }
}
ratiolabels = {
    "b": "Data / SM",
    "s": "Data / SM + {signal}",
}
lumis = {
    "2016pre": "19.5",
    "2016post": "16.8",
    "2017": "41.5",
    "2018": "59.9",
    "Run 2": "138",
}

hatchstyle = dict(
    hatch = "///",
    facecolor = "none",
    edgecolor = "black",
    linewidth = 0,
)
hatchstyle = dict(
    color = "black",
    alpha = 0.3,
    linewidth = 0,
)
datastyle = dict(
    marker = "o",
    markersize = 2,
    elinewidth = 0.6,
    linestyle = "none",
    color = "black"
)


def get_g_values(fname, signals):
    twing = len(signals) == 2
    onepoi = "one-poi" in fname

    if not twing and not onepoi:
        raise NotImplementedError()

    ffile = TFile.Open(fname, "read")
    fres = ffile.Get("fit_s")
    signals = list(signals.keys())

    if onepoi:
        return {signals[0]: round(fres.floatParsFinal().getRealValue('r'), 2)}
    else:
        return {
            signals[0]: round(fres.floatParsFinal().getRealValue('g1'), 2),
            signals[1]: round(fres.floatParsFinal().getRealValue('g2'), 2)
        }

def plot_eventperbin(ax, bins, centers, smhists, data, log):
    total = None
    for hist in smhists.values():
        if total is None:
            total = hist.copy()
        else:
            total += hist
    width = np.diff(bins)
    colors = [sm_colors[k] for k in smhists.keys()]
    hep.histplot(
        [hist.values() / width for hist in smhists.values()],
        bins = bins,
        ax = ax,
        stack = True,
        histtype = "fill",
        label = smhists.keys(),
        color = colors
    )
    ax.fill_between(
        total.axes[0].centers,
        (total.values() + total.variances() ** .5) / width,
        (total.values() - total.variances() ** .5) / width,
        step = "mid",
        label = "Post-fit uncertainty",
        **hatchstyle)
    ax.errorbar(
        centers,
        data[0] / width,
        yerr = data[1] / width,
        label = "Data",
        **datastyle
    )
    ax.set_ylabel("<Events / GeV>")
    if log:
        ax.set_yscale("log")


def plot_ratio(ax, bins, centers, data, total, signals, fit):
    ax.errorbar(
        centers,
        data[0] / total.values(),
        data[1] / total.values(),
        **datastyle
    )
    err_up = 1 + data[1][1] / data[0]
    err_down = 1 - data[1][0] / data[0]
    ax.fill_between(
        bins,
        np.r_[err_up[0], err_up],
        np.r_[err_down[0], err_down],
        step = "pre",
        **hatchstyle
    )
    for pos in [0.8, 0.9, 1.1, 1.2]:
        ax.axhline(y = pos, linestyle = ":", linewidth = 0.5, color = "black")
    ax.axhline(y = 1, linestyle = "--", linewidth = 0.5, color = "black")
    ax.set_ylim(0.75, 1.25)
    signals = list(signals.keys())
    ax.set_ylabel(ratiolabels[fit].format(
        signal = "A + H" if len(signals) == 2 and sorted([signals[0][0], signals[1][0]]) == ["A", "H"] else f"{signals[0][0]}({signals[0][1]}, {signals[0][2]}%)"
    ))

def plot_diff(ax, bins, centers, data, smhists, signals, gvalues, fit):
    total = None
    for hist in smhists.values():
        if total is None:
            total = hist.copy()
        else:
            total += hist
    width = np.diff(bins)
    ax.errorbar(
        centers,
        (data[0] - total.values()) / width,
        ((data[1] ** 2 + total.variances()) ** .5) / width,
        **datastyle
    )
    err = total.variances() ** .5
    ax.fill_between(
        bins,
        np.r_[err[0], err] / np.r_[width[0], width],
        - np.r_[err[0], err] / np.r_[width[0], width],
        step = "pre",
        **hatchstyle
    )
    for key, signal in signals.items():
        symbol, mass, decaywidth = key
        if symbol == "Total":
            signal_label = "A + H"
        else:
            signal_label = f"{symbol}({mass}, {decaywidth}%)"
        if key in gvalues and gvalues[key] is not None:
            if fit == "s":
                signal_label += f", $g_{{\\mathrm{{{symbol}}}}} = {gvalues[key]}$"
            elif fit == "b":
                signal_label += f", $g_{{\\mathrm{{{symbol}}}}} = 0$"
        hep.histplot(
            signal.values() / width,
            bins = bins,
            yerr = np.zeros(len(signal.axes[0])),
            ax = ax,
            histtype = "step",
            color = signal_colors[symbol],
            linewidth = 1.25,
            label = signal_label,
            zorder = signal_zorder[symbol]
        )
    ax.set_ylabel("<(Data - SM) / GeV>")
    ax.legend(loc = "lower left", bbox_to_anchor = (0, 1.03, 1, 0.2), borderaxespad = 0, ncol = 5, mode = "expand").get_frame().set_edgecolor("black")


def plot(
        channel,
        year,
        fit,
        smhists,
        datavalues,
        total,
        signals,
        gvalues,
        datahist_errors,
        binning,
        num_extrabins,
        extra_axes,
        first_ax_binning,
        first_ax_width,
        bins,
        centers,
        log):

    if len(smhists) == 0:
        return

    fig, (ax0, ax1, ax2) = plt.subplots(
        nrows = 3,
        sharex = True,
        gridspec_kw = {"height_ratios": [0.001, 2.5, 1]},
        figsize = (10.5, 5.5)
    )
    ax0.set_axis_off()
    plot_eventperbin(ax1, bins, centers, smhists, (datavalues, datahist_errors), log)
    if "s" in fit and args.lower == "ratio":
        raise NotImplementedError()
    if args.lower == "ratio":
        plot_ratio(ax2, bins, centers, (datavalues, datahist_errors), total, signals, fit)
    elif args.lower == "diff":
        plot_diff(ax2, bins, centers, (datavalues, datahist_errors), smhists, signals, gvalues, fit)
    else:
        raise ValueError(f"Invalid lower type: {args.lower}")
    for pos in bins[::len(first_ax_binning) - 1][1:-1]:
        ax1.axvline(x = pos, linestyle = "--", linewidth = 0.5, color = "gray")
        ax2.axvline(x = pos, linestyle = "--", linewidth = 0.5, color = "gray")
    if log:
        ax1.set_ylim(ax1.get_ylim()[0], ax1.transData.inverted().transform(ax1.transAxes.transform([0, 1.1]))[1])
    else:
        ax1.set_ylim(0, ax1.get_ylim()[1] * 1.1)
    for j, (variable, edges) in enumerate(extra_axes.items()):
        for i in range(num_extrabins):
            edge_idx = np.unravel_index(i, tuple(len(b) - 1 for b in extra_axes.values()))[j]
            text = r"{} $\leq$ {} < {}".format(edges[edge_idx], variable, edges[edge_idx + 1])
            ax1.text(1 / num_extrabins * (i + 0.5), 0.955 - j * 0.06, text, horizontalalignment = "center", fontsize = "small", transform = ax1.transAxes)
    ax1.minorticks_on()
    ax2.minorticks_on()
    ticks = []
    for i in range(num_extrabins):
        for j in (1 / 4, 3 / 4):
            ticks.append(first_ax_width * i + first_ax_width * j)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels((ax2.get_xticks() % first_ax_width + first_ax_binning[0]).astype(int))
    ax1.legend(loc = "lower left", bbox_to_anchor = (0, 1.02, 1, 0.2), borderaxespad = 0, ncol = 5, mode = "expand", edgecolor = "black", framealpha = 1)
    ax1.set_xlabel("")
    ax2.set_xlabel(list(binning.keys())[0])
    ax0.set_title(channel.replace('m', '$\\mu$'))
    hep.cms.label(ax = ax0, llabel = "Work in progress", lumi = lumis[year], loc = 0, year = year, fontsize = 13)
    fig.subplots_adjust(hspace = 0.27, left = 0.075, right = 1-0.025, top = 1 - 0.075)
    bbox = ax2.get_position()
    offset = -0.01
    ax2.set_position([bbox.x0, bbox.y0 + offset, bbox.x1 - bbox.x0, bbox.y1 - bbox.y0])
    fig.set_figwidth(12.8)
    fig.set_dpi(300)
    extent = None if args.plotupper else ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    sstr = [ss for ss in signals.keys() if ss[0] != "Total"]
    sstr = [ss[0] + "_m" + str(ss[1]) + "_w" + str(float(ss[2])).replace(".", "p") for ss in sstr]
    sstr = "__".join(sstr)
    cstr = channel.replace(r'$\ell\ell$', 'll').replace(r'$\ell$', 'l').replace('+', 'p')
    ystr = year.replace(" ", "").lower()
    fig.savefig(f"{args.odir}/{sstr}{args.ptag}_fit_{fit}_{cstr}_{ystr}{args.fmt}", transparent = True, bbox_inches = extent)
    fig.clf()

def sum_kwargs(channel, year, *summands):
    ret = summands[0].copy()
    ret["channel"] = channel
    ret["year"] = year
    for summand in summands[1:]:
        ret["smhists"] = {k: ret["smhists"][k] + summand["smhists"][k] for k in ret["smhists"]}
        ret["datavalues"] = ret["datavalues"] + summand["datavalues"]
        ret["total"] = ret["total"] + summand["total"]
        ret["signals"] = {k: ret["signals"][k] + summand["signals"][k] for k in ret["signals"]}
        ret["datahist_errors"] = (ret["datahist_errors"] ** 2 + summand["datahist_errors"] ** 2) ** .5
    return ret


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
        for proc, label in sm_procs.items():
            if proc not in directory:
                continue
            hist = directory[proc].to_hist()[:len(centers)]
            if label not in smhists:
                smhists[label] = hist
            else:
                smhists[label] += hist
        signals = {}
        for key in directory.keys():
            if (match := signal_name_pat.match(key)) is not None:
                mass = int(match.group(2))
                width = float(match.group(3).replace("p", "."))
                if width % 1 == 0:
                    width = int(width)
                hist = directory[key].to_hist()[:len(centers)]
                if (match.group(1), mass, width) in signals:
                    signals[(match.group(1), mass, width)] += hist
                else:
                    signals[(match.group(1), mass, width)] = hist
        gvalues = get_g_values(args.ifile, signals)
        if len(signals) > 1:
            total = None
            for hist in signals.values():
                if total is None:
                    total = hist.copy()
                else:
                    total += hist
            signals[("Total", None, None)] = total
        datavalues = directory["data"].values()[1][:len(centers)]
        total = directory["total"].to_hist()[:len(centers)]
        datahist_errors = np.array([directory["data"].errors("low")[1], directory["data"].errors("high")[1]])[:, :len(centers)]

        kwargs = {
            "channel": channel,
            "year": year,
            "fit": fit,
            "smhists": smhists,
            "datavalues": datavalues,
            "total": total,
            "signals": signals,
            "gvalues": gvalues,
            "datahist_errors": datahist_errors,
            "binning": binning,
            "num_extrabins": num_extrabins,
            "extra_axes": extra_axes,
            "first_ax_binning": first_ax_binning,
            "first_ax_width": first_ax_width,
            "bins": bins,
            "centers": centers,
            "log": args.log
        }

        if args.each:
            plot(**kwargs)

        if (channel, fit) in year_summed:
            this_year = year_summed[(channel, fit)]
            year_summed[(channel, fit)] = sum_kwargs(channel, "Run 2", kwargs, this_year)
        else:
            year_summed[(channel, fit)] = kwargs

batches = {
    r"$\ell\ell$": ["ee", "em", "mm"],
    r"$\ell$j":    ["e4pj", "m4pj", "e3j", "m3j"],
    #r"ej":         ["e4pj", "e3j"],
    #r"mj":         ["e4pj", "m3j"],
    r"$\ell$3j":   ["e3j", "m3j"],
    r"$\ell$4+j":  ["e4pj", "m4pj"],
}
for cltx, channels in batches.items():
    for fit in fits:
        has_channel = all([(channel, fit) in year_summed for channel in channels])
        if has_channel:
            plot(**sum_kwargs(cltx, "Run 2", *(year_summed[(channel, fit)] for channel in channels)))
