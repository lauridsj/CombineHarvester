#!/usr/bin/env python3
# original script by jonas ruebenach (desy) @ https://gitlab.cern.ch/jrubenac/ahtt_scripts/-/blob/a1020072d17d6813b55fc6f0c3a382538b542f3e/plot_post_fit.py
# environment: source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_102 x86_64-centos7-gcc11-opt
# updating mpl: python3 -m pip install matplotlib --upgrade
# actually using it: export PYTHONPATH=`python3 -c 'import site; print(site.getusersitepackages())'`:$PYTHONPATH
# FIXME: summed up templates across years not yet fully correct: check out https://cms-analysis.github.io/CombineHarvester/post-fit-shapes-ws.html
# more info: https://cms-analysis.github.io/CombineHarvester/post-fit-shapes-ws.html

import os
from itertools import product
import re
from argparse import ArgumentParser
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa:E402
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['figure.max_open_warning'] = False
plt.rcParams["font.size"] = 15.0
from matplotlib.transforms import Bbox

import uproot  # noqa:E402
import mplhep as hep  # noqa:E402
import ROOT
import math

from utilspy import tuplize
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes

parser = ArgumentParser()
parser.add_argument("--ifile", help = "input file ie fitdiagnostic results", default = "", required = True)
parser.add_argument("--lower", choices = ["ratio", "diff"], default = "diff", required = False)
parser.add_argument("--log", action = "store_true", required = False)
parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False)
parser.add_argument("--plot-tag", help = "extra tag to append to plot names", dest = "ptag", default = "", required = False, type = prepend_if_not_empty)
parser.add_argument("--skip-each", help = "skip plotting each channel x year combination", action = "store_false", dest = "each", required = False)
parser.add_argument("--batch", help = "plot sums of channels x year combinations", action = "store_true", dest = "batch", required = False)
parser.add_argument("--skip-prefit", help = "skip plotting prefit", action = "store_false", dest = "prefit", required = False)
parser.add_argument("--prefit-signal-from", help = "read prefit signal templates from this file instead",
                    default = "", dest = "ipf", required = False)
parser.add_argument("--only-lower", help = "dont plot the top panel. WIP, doesnt really work yet", dest = "plotupper", action = "store_false", required = False)
parser.add_argument("--plot-format", help = "format to save the plots in", default = ".png", dest = "fmt", required = False, type = lambda s: prepend_if_not_empty(s, '.'))
parser.add_argument("--signal-scale", help = "scaling to apply on signal in drawing", default = (1., 1.),
                    dest = "sigscale", required = False, type = lambda s: tuplize(s))
parser.add_argument("--as-signal", help = "comma-separated list of background processes to draw as signal",
                    dest = "assignal", default = "", required = False,
                    type = lambda s: [] if s == "" else tokenize_to_list(remove_spaces_quotes(s)))
parser.add_argument("--skip-ah", help = "don't draw A/H signal histograms", action = "store_false", dest = "doah", required = False)
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
    "TT": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_const_pos": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_const_neg": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_lin_pos": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_lin_neg": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_quad_pos": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_quad_neg": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EtaT": r"$\eta_{\mathrm{t}}$",
}
proc_colors = {
    "tX": "C0",
    r"$\mathrm{VX}$, $\mathrm{t}\bar{\mathrm{t}}\mathrm{V}$": "C1",
    r"$\mathrm{t}\bar{\mathrm{t}}$": "#F3E5AB",
    r"$\mathrm{VX}$, $\mathrm{t}\bar{\mathrm{t}}\mathrm{V}$, QCD": "C1",
    r"$\eta_{\mathrm{t}}$": "#cc0033",

    "A": "#cc0033",
    "H": "#0033cc",
    "Total": "#3B444B"
}
signal_zorder = {
    r"$\eta_{\mathrm{t}}$": 0,
    "Total": 1,
    "A": 2,
    "H": 3
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
    "s": "Data / SM",
    "p": "Data / SM",
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
    markersize = 2.5,
    elinewidth = 0.7,
    linestyle = "none",
    color = "black"
)



def get_poi_values(fname, signals):
    signals = list(signals.keys())
    signals = [signal for signal in signals if signal != ("Total", None, None)]
    twing = len(signals) == 2 and all([sig[0] == "A" or sig[0] == "H" for sig in signals])
    onepoi = "one-poi" in fname
    etat = signals[0][0] == r"$\eta_{\mathrm{t}}$"

    if not twing and not onepoi and not etat:
        raise NotImplementedError()

    ffile = ROOT.TFile.Open(fname, "read")
    if "fit_s" not in ffile.GetListOfKeys():
        return {sig: 0 for sig in signals}

    fres = ffile.Get("fit_s")

    if onepoi:
        return {signals[0]: round(fres.floatParsFinal().getRealValue('g'), 2)}
    elif twing:
        return {
            signals[0]: round(fres.floatParsFinal().getRealValue('g1'), 2),
            signals[1]: round(fres.floatParsFinal().getRealValue('g2'), 2)
        }
    elif etat:
        return {signals[0]: round(fres.floatParsFinal().getRealValue('CMS_EtaT_norm_13TeV'), 2)}



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



def plot_eventperbin(ax, bins, centers, smhists, total, data, log, fit):
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
            label = f"{fstage}-fit{ftype}uncertainty" if ibin == 0 else None,
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
    err_up = 1 + (total.variances() ** .5) / total.values()
    err_down = 1 - (total.variances() ** .5) / total.values()
    ax.fill_between(
        bins,
        np.r_[err_up[0], err_up],
        np.r_[err_down[0], err_down],
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
            (total.values() + signal.values()) / total.values(),
            bins = bins,
            yerr = np.zeros(len(signal.axes[0])),
            ax = ax,
            histtype = "step",
            color = proc_colors[symbol],
            linewidth = 1.25,
            label = signal_label,
            zorder = signal_zorder[symbol]
        )
    for pos in [0.8, 0.9, 1.1, 1.2]:
        ax.axhline(y = pos, linestyle = ":", linewidth = 0.5, color = "black")
    ax.axhline(y = 1, linestyle = "--", linewidth = 0.35, color = "black")
    ax.set_ylim(0.75, 1.25)
    ax.set_ylabel(ratiolabels[fit])
    ax.legend(loc = "lower left", bbox_to_anchor = (0, 1.05, 1, 0.2), borderaxespad = 0, ncol = 5, mode = "expand").get_frame().set_edgecolor("black")



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
        else:
            signal_label += f"{symbol}({mass}, {decaywidth}%)"

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
            linewidth = 1.25,
            label = signal_label,
            zorder = signal_zorder[symbol]
        )
    ax.set_ylabel("<(Data - SM) / GeV>")
    #ax.axhline(y = 1, linestyle = "--", linewidth = 0.35, color = "black")
    ax.legend(loc = "lower left", bbox_to_anchor = (0, 1.05, 1, 0.2), borderaxespad = 0, ncol = 5, mode = "expand").get_frame().set_edgecolor("black")



def plot(channel, year, fit,
         smhists, datavalues, total, promotions, signals, gvalues, sigscale, datahist_errors,
         binning, num_extrabins, extra_axes, first_ax_binning, first_ax_width, bins, centers, log):
    if len(smhists) == 0:
        return

    fig, (ax0, ax1, ax2) = plt.subplots(
        nrows = 3,
        sharex = True,
        gridspec_kw = {"height_ratios": [0.001, 2.5, 1]},
        figsize = (10.5, 5.5)
    )
    ax0.set_axis_off()
    plot_eventperbin(ax1, bins, centers, smhists, total, (datavalues, datahist_errors), log, fit)
    if "s" in fit and args.lower == "ratio":
        raise NotImplementedError()
    if args.lower == "ratio":
        plot_ratio(ax2, bins, centers, (datavalues, datahist_errors), total, promotions | signals, fit)
    elif args.lower == "diff":
        plot_diff(ax2, bins, centers, (datavalues, datahist_errors), total, promotions | signals, gvalues, sigscale, fit)
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
    ax1.legend(loc = "lower left", bbox_to_anchor = (0, 1.02, 1, 0.2), borderaxespad = 0, ncol = len(smhists) + 2, mode = "expand", edgecolor = "black", framealpha = 1)
    ax1.set_xlabel("")
    ax2.set_xlabel(list(binning.keys())[0])
    ax0.set_title(channel.replace('m', '$\\mu$'))
    ax0.set_title(channel.replace('4p', '4+'))
    hep.cms.label(ax = ax0, llabel = "", lumi = lumis[year], loc = 0, year = year, fontsize = 17)
    fig.subplots_adjust(hspace = 0.27, left = 0.075, right = 1 - 0.025, top = 1 - 0.075)
    bbox = ax2.get_position()
    offset = -0.01
    ax2.set_position([bbox.x0, bbox.y0 + offset, bbox.x1 - bbox.x0, bbox.y1 - bbox.y0])
    fig.set_size_inches(w = 19.2, h = 1.5 * fig.get_figheight())
    fig.set_dpi(450)
    extent = None if args.plotupper else full_extent(ax2).transformed(fig.dpi_scale_trans.inverted())

    sstr = [ss for ss in signals.keys() if ss[0] != "Total"]
    if sstr[0][0] == r"$\eta_{\mathrm{t}}$":
        sstr = "EtaT"
    else:
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
        ret["promotions"] = {k: ret["promotions"][k] + summand["promotions"][k] for k in ret["promotions"]}
        ret["signals"] = {k: ret["signals"][k] + summand["signals"][k] for k in ret["signals"]}
        ret["datahist_errors"] = (ret["datahist_errors"] ** 2 + summand["datahist_errors"] ** 2) ** .5
    return ret



def add_covariance(histogram, matrix):
    # still good ol when it comes to direct fudging
    roo = ROOT.TH1D("", "", len(histogram.values()), 0., len(histogram.values()))
    for ibin in range(len(histogram.values())):
        roo.SetBinContent(ibin + 1, histogram.values()[ibin])
        roo.SetBinError(ibin + 1, math.sqrt(matrix.values()[ibin, ibin]))
    return uproot.pyroot.from_pyroot(roo).to_hist()



def zero_variance(histogram):
    roo = uproot.pyroot.to_pyroot(histogram)
    for ibin in range(len(histogram.values())):
        roo.SetBinContent(ibin + 1, histogram.values()[ibin])
        roo.SetBinError(ibin + 1, 0.)
    return uproot.pyroot.from_pyroot(roo).to_hist()



signal_name_pat = re.compile(r"(A|H)_m(\d+)_w(\d+p?\d*)_")
year_summed = {}
with uproot.open(args.ifile) as f:
    for channel, year, fit in product(channels, years, fits):
        if fit == "p" and not args.prefit:
            continue
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
            hist = directory[proc].to_hist()[:len(centers)]
            if proc not in args.assignal:
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

            if len(signals) > 1:
                signals[("Total", None, None)] = directory["total_signal"].to_hist()[:len(centers)]

        gvalues = get_poi_values(args.ifile, promotions | signals)
        datavalues = directory["data"].values()[1][:len(centers)]
        total = directory["total_background"].to_hist()[:len(centers)]
        for promotion in promotions.values():
            total += -1. * promotion
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
            "log": args.log
        }

        if args.each:
            plot(**kwargs)

        if args.batch:
            if (channel, fit) in year_summed:
                this_year = year_summed[(channel, fit)]
                year_summed[(channel, fit)] = sum_kwargs(channel, "Run 2", kwargs, this_year)
            else:
                year_summed[(channel, fit)] = kwargs

batches = {
    r"$\ell\ell$": ["ee", "em", "mm"],
    r"$\ell$j":    ["e4pj", "m4pj", "e3j", "m3j"],
    r"ej":         ["e4pj", "e3j"],
    r"mj":         ["e4pj", "m3j"],
    r"$\ell$3j":   ["e3j", "m3j"],
    r"$\ell$4+j":  ["e4pj", "m4pj"],
}
if args.batch:
    #zeroed = year_summed.copy()
    #for channel, fit in year_summed.keys():
    #    for preds in ["smhists", "signals"]:
    #        zeroed[(channel, fit)][preds] = {k: zero_variance(zeroed[(channel, fit)][preds][k]) for k in zeroed[(channel, fit)]}

    for cltx, channels in batches.items():
        for fit in fits:
            has_channel = all([(channel, fit) in year_summed for channel in channels])
            if has_channel:
                plot(**sum_kwargs(cltx, "Run 2", *(year_summed[(channel, fit)] for channel in channels)))
