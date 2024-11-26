#!/usr/bin/env python3
# draw the cls(g) for some signal points
# requires matplotlib > 3.3 e.g. source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_100 x86_64-centos7-gcc10-opt

from argparse import ArgumentParser
import os
import sys
import numpy as np
from scipy import stats
import math

import glob
from collections import OrderedDict
import json

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpt
import matplotlib.lines as mln
import matplotlib.colors as mcl
import matplotlib.ticker as mtc

from utilscombine import get_fit
from drawings import min_g, max_g, epsilon, axes, first, second, get_point, str_point
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes

def parse_to_kv(tag):
    tag = tag.split(':')
    key = tag[0]
    values = tokenize_to_list(tag[1]) if len(tag) > 1 else [key]
    return [key, values]

def normal_2sided_pvalue(dnll):
    if dnll < 0.:
        return 1.
    dnll = abs(2. * dnll)
    return 2. * stats.norm.sf(math.sqrt(dnll)) if 0. <= dnll <= 100. else 1e-23

def read_pval0(which, var1, ahdirs, ahtag, etatag, etadirs):
    '''
    reads directories, and looks for files that might contain the pvalue0 numbers
    the fn looks first for the F-C version, and then the NLL version as a fallback
    the NLL version obtains the pvalue in the Gaussian approximation
    ask Afiq how wonderfully excellent of an idea this is, especially when which = full, if you've time for some comedy
    the return value is a dict for the pvalues for different scenarii
    '''
    pois = ["r1", "r2"] if which == "resonance" else ["g1", "g2"]
    result = {
        "A exp": [],
        "A obs": [],
        "H exp": [],
        "H obs": [],
        "etat exp": 1.,
        "etat obs": 1.
    }

    for ahd in ahdirs:
        apnt = get_point(ahd.split('__')[0])
        hpnt = get_point(ahd.split('__')[1])

        av1 = apnt[1] if var1 == "mass" else apnt[2]
        hv1 = hpnt[1] if var1 == "mass" else hpnt[2]

        # F-C json files, names to be checked
        afce = sorted(glob.glob(ahd + "/*_" + ahtag[1][0] + "_fc-scan_exp-10_*.json"), key = lambda name: int(name.split('_')[-1].split('.')[0]))
        afco = sorted(glob.glob(ahd + "/*_" + ahtag[1][0] + "_fc-scan_obs_*.json"), key = lambda name: int(name.split('_')[-1].split('.')[0]))
        hfce = sorted(glob.glob(ahd + "/*_" + ahtag[1][1] + "_fc-scan_exp-01_*.json"), key = lambda name: int(name.split('_')[-1].split('.')[0]))
        hfco = sorted(glob.glob(ahd + "/*_" + ahtag[1][1] + "_fc-scan_obs_*.json"), key = lambda name: int(name.split('_')[-1].split('.')[0]))

        # NLL files
        alle = glob.glob(ahd + "/*_" + ahtag[1][0] + f"_nll_exp-10_{pois[0]}_0to0.root")
        allo = glob.glob(ahd + "/*_" + ahtag[1][0] + f"_nll_obs_{pois[0]}_0to0.root")
        hlle = glob.glob(ahd + "/*_" + ahtag[1][1] + f"_nll_exp-01_{pois[1]}_0to0.root")
        hllo = glob.glob(ahd + "/*_" + ahtag[1][1] + f"_nll_obs_{pois[1]}_0to0.root")

        lookups = {
            "A exp": [afce, alle],
            "A obs": [afco, allo],
            "H exp": [hfce, hlle],
            "H obs": [hfco, hllo]
        }

        for scenario, fcnll in lookups.items():
            fc, nll = fcnll
            pval = None
            if len(fc) > 0:
                pass
            elif len(nll) > 0:
                pval = get_fit(nll[0], ["deltaNLL"], qexp_eq_m1 = False)
                if pval is not None:
                    pval = normal_2sided_pvalue(pval[0])

            if pval is not None:
                result[scenario].append((av1 if 'A' in scenario else hv1, pval))
        pass

    for kk in result.keys():
        if 'A' in kk or 'H' in kk:
            result[kk].sort(key = lambda tup: tup[0])

    for etd in etadirs:
        # F-C json files, names DEFINITELY have to be checked
        efce = sorted(glob.glob(etd + "/*_" + etatag[1][0] + "_fc-scan_exp-b_*.json"), key = lambda name: int(name.split('_')[-1].split('.')[0]))
        efco = sorted(glob.glob(etd + "/*_" + etatag[1][0] + "_fc-scan_obs_*.json"), key = lambda name: int(name.split('_')[-1].split('.')[0]))

        # NLL files
        elle = glob.glob(etd + "/*_" + etatag[1][0] + "_nll_exp-b_CMS_EtaT_norm_13TeV_0to0.root")
        ello = glob.glob(etd + "/*_" + etatag[1][0] + "_nll_obs_CMS_EtaT_norm_13TeV_0to0.root")

        lookups = {
            "etat exp": [efce, elle],
            "etat obs": [efco, ello],
        }

        for scenario, fcnll in lookups.items():
            fc, nll = fcnll
            pval = None
            if len(fc) > 0:
                pass
            elif len(nll) > 0:
                pval = get_fit(nll[0], ["deltaNLL"], qexp_eq_m1 = False)
                if pval is not None:
                    pval = normal_2sided_pvalue(pval[0])

            if pval is not None:
                result[scenario] = pval
        pass
    return result

def really_draw_pval0(onames, pvalues, label, xaxis, xticks, xlabels, legendtext, expected, observed, formal, cmsapp, luminosity, transparent):
    colors = {
        "A":     "#cc0033",
        "H":     "#0033cc",
        "etat":  "#009000",
        "sigma": "#3B444B"
    }
    # no this doesnt have to make sense - just whatever that needs to be tuned per channel
    specifics = {
        'lx': [(1e-21, 1000.), [1e-21, 1e-14, 1e-7, 1], [r'$10^{-21}$', r'$10^{-14}$', r'$10^{-7}$', '$1\,$'], (20, 0), (0.5, 0.25, 0.5, 0.1), (9., 10., 10.)],
        'll': [(1e-12, 50.), [1e-12, 1e-8, 1e-4, 1], [r'$10^{-12}$', r'$10^{-8}$', r'$10^{-4}$', '$1\,$'], (10, 10), (0.5, 0.2, 0.5, 0.1), (3.8, 4., 4.)],
        'lj': [(1e-2, 2.), [1e-2, 1e-1, 1], [r'$10^{-2}$', r'$10^{-1}$', '$1\,$'], (20, 0), (0.5, 0.19, 0.5, 0.1), (1.29, 1.3, 1.3)],
    }
    channel = 'lj' # hardcoded atm

    axe = np.array(first(pvalues['A exp']))
    hxe = np.array(first(pvalues['H exp']))

    aye = np.array(second(pvalues['A exp']))
    ayo = np.array(second(pvalues['A obs']))

    hye = np.array(second(pvalues['H exp']))
    hyo = np.array(second(pvalues['H obs']))

    minx, maxx = min([min(axe), min(hxe)]), max([max(axe), max(hxe)])
    miny, maxy = specifics[channel][0]
    handles = []
    fits = []

    fig, ax = plt.subplots(dpi = 600)
    for ii in [0, 1, 2, 3, 4, 5, 7, 9]:
        if miny < 1e-11 and ii in [1, 2, 4]:
            continue

        ax.plot(np.array([minx, maxx]), np.array([normal_2sided_pvalue(ii * ii / 2.), normal_2sided_pvalue(ii * ii / 2.)]), color = colors["sigma"], linestyle = 'dotted', linewidth = 2)
        plt.annotate(f"${ii}\sigma$", (0.97 * maxx, normal_2sided_pvalue(ii * ii / 2.)), textcoords = "offset points", xytext = (0, 2), ha = 'center', fontsize = 17, color = colors["sigma"])
    #ax.plot(np.array([minx, maxx]), np.array([1.35, 1.35]), color = colors["sigma"], linestyle = 'solid', linewidth = 1) # cms calibration line
    if expected:
        ax.plot(hxe, hye, color = colors["H"], linestyle = 'dashed', linewidth = 5)
        ax.plot(axe, aye, color = colors["A"], linestyle = 'dashed', linewidth = 5)
        ax.plot(np.array([minx]), np.array([pvalues['etat exp']]),
                marker = 'x', markersize = 20, color = colors["etat"])

    if observed:
        ax.plot(hxe, hyo, color = colors["H"], linestyle = 'solid', linewidth = 5)
        ax.plot(axe, ayo, color = colors["A"], linestyle = 'solid', linewidth = 5)
        ax.plot(np.array([minx]), np.array([pvalues['etat obs']]),
                marker = 9, markersize = 20, color = colors["etat"])
        plt.annotate(r"$\eta_{\mathrm{t}},\, \mathrm{m} =$343 GeV", (minx, pvalues['etat obs']), textcoords = "offset points", xytext = specifics[channel][3], ha = 'left', fontsize = 17, color = colors["etat"])

    handles.append((mln.Line2D([0], [0], color = colors["A"], linestyle = 'solid', linewidth = 5), r'$\mathrm{\mathsf{A}}$'))
    handles.append((mln.Line2D([0], [0], color = colors["H"], linestyle = 'solid', linewidth = 5), r'$\mathrm{\mathsf{H}}$'))
    handles.append((mln.Line2D([0], [0], color = colors["etat"], marker = 9, markersize = 20, linewidth = 0), r'$\eta_{\mathrm{t}}$'))

    if expected and observed:
        handles.append((mln.Line2D([0], [0], color = "0", linestyle = 'dashed', marker = 'x', markersize = 20, linewidth = 5), r"Expected"))
        handles.append((mln.Line2D([0], [0], color = "0", linestyle = 'solid', marker = 9, markersize = 20, linewidth = 5), r"Observed"))

    plt.xlim((minx, maxx))
    plt.ylim((miny, maxy))
    plt.xlabel(xaxis, fontsize = 26, loc = "right")
    plt.ylabel("Local p-value", fontsize = 26, loc = "top")
    if label != "":
        ax.set_title(label, fontsize = 23)
    ax.set_yscale('log')
    ax.margins(x = 0, y = 0)

    ax.legend(first(handles), second(handles), ncols = 3, title = legendtext, title_fontsize = 21, loc = 'best',
              bbox_to_anchor = specifics[channel][4],
              fontsize = 21, handlelength = 1.5, borderaxespad = 1., frameon = False)

    if formal:
        xwindow = maxx - minx
        ctxt = "{cms}".format(cms = r"$\textbf{CMS}$")
        ax.text(0.01 * xwindow + minx, specifics[channel][5][0], ctxt, fontsize = 31, ha = 'left', va = 'bottom', usetex = True)

        if cmsapp != "":
            atxt = "{app}".format(app = r" $\textit{" + cmsapp + r"}$")
            ax.text(0.02 * xwindow + minx, specifics[channel][5][1], atxt, fontsize = 26, ha = 'left', va = 'bottom', usetex = True)

        ltxt = "{lum}{ifb}".format(lum = luminosity, ifb = r" fb$^{\mathrm{\mathsf{-1}}}$ (13 TeV)")
        ax.text(0.99 * xwindow + minx, specifics[channel][5][2], ltxt, fontsize = 23, ha = 'right', va = 'bottom', usetex = True)

    ax.minorticks_off()
    ax.tick_params(axis = "both", which = "major", direction = "in", bottom = True, top = True, left = True, right = True)
    ax.tick_params(axis = "y", which = "major", width = 1, length = 8, labelsize = 22, pad = 3)
    ax.tick_params(axis = "x", which = "major", width = 1, length = 8, labelsize = 22, pad = 7)
    if xticks is not None and xlabels is not None:
        ax.xaxis.set_major_locator(mtc.FixedLocator(xticks))
        ax.xaxis.set_major_formatter(mtc.FixedFormatter(xlabels))
    ax.yaxis.set_major_locator(mtc.FixedLocator(specifics[channel][1]))
    ax.yaxis.set_major_formatter(mtc.FixedFormatter(specifics[channel][2]))

    fig.set_size_inches(8., 8.)
    fig.tight_layout()
    for oname in onames:
        fig.savefig(oname, transparent = transparent)
    plt.close()

def draw_pval0(var1, onames, which, ahdirs, apoints, hpoints, ahtag, etatag, etadirs, label, annotate, expected, observed, formal, cmsapp, luminosity, transparent):
    if not hasattr(draw_pval0, "settings"):
        draw_pval0.settings = OrderedDict([
            ("mass",  {"var2": "width", "iv1": 1, "iv2": 2, "label": r", $\Gamma_{\mathrm{\mathsf{A/H}}}=$ %.1f%% m$_{\mathrm{\mathsf{A/H}}}$"}),
            ("width", {"var2": "mass", "iv1": 2, "iv2": 1, "label": r", m$_{\mathrm{\mathsf{A/H}}}=$ %d GeV"})
        ])

    var2s = set([pnt[draw_pval0.settings[var1]["iv2"]] for pnt in apoints + hpoints])
    xaxis = r'm$_{\mathrm{\mathsf{A/H}}}$ [GeV]' if var1 == "mass" else r'$\Gamma_{\mathrm{\mathsf{A/H}}}$ [% m$_{\mathrm{\mathsf{A/H}}}$]'

    for vv in var2s:
        print(f"running {draw_pval0.settings[var1]['var2']} {vv}")
        av1s = list(set([pnt[draw_pval0.settings[var1]["iv1"]] for pnt in apoints if pnt[draw_pval0.settings[var1]["iv2"]] == vv]))
        hv1s = list(set([pnt[draw_pval0.settings[var1]["iv1"]] for pnt in hpoints if pnt[draw_pval0.settings[var1]["iv2"]] == vv]))
        xticks = None if var1 != "mass" else [450., 600., 750., 900.]
        xlabels = None if var1 != "mass" else ['450', '600', '750', '900']

        dirs = []
        for dd in ahdirs:
            fora = [(var1 == "mass" and f"A_m{int(v1)}_w{str(vv).replace('.', 'p')}" in dd) or (var1 == "width" and f"A_m{int(vv)}_w{str(v1).replace('.', 'p')}" in dd) for v1 in av1s]
            forh = [(var1 == "mass" and f"H_m{int(v1)}_w{str(vv).replace('.', 'p')}" in dd) or (var1 == "width" and f"H_m{int(vv)}_w{str(v1).replace('.', 'p')}" in dd) for v1 in hv1s]
            if any(fora + forh):
                dirs.append(dd)

        legendtext = r"Resonance" if which == "resonance" else r"Full"
        legendtext += draw_pval0.settings[var1]["label"] % vv
        if int(vv) == vv:
            legendtext = legendtext.replace('%.1f' % vv, str(int(vv)))

        really_draw_pval0(
            [oname.format(fff = 'w' + str(vv).replace('.', 'p') if var1 == "mass" else 'm' + str(int(vv))) for oname in onames],
            read_pval0(which, var1, dirs, ahtag, etatag, etadirs),
            axes[label] if label != "" and label in axes else label, xaxis, xticks, xlabels, legendtext, expected, observed, formal, cmsapp, luminosity, transparent
        )
    pass

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--function", help = "plot pvalues as a function of?", default = "mass",
                        choices = ["mass", "width"], required = False)

    parser.add_argument("--which", help = "which pvalues are being plotted?", default = "full",
                        choices = ["full", "resonance"], required = False)

    parser.add_argument("--ah-tag", help = "input tag:output-tag key:value tuples to search, for A/H p-values. "
                        "the tuples are semicolon separated, the k:v's colon-separated, and the values are comma-separated. "
                        "each individual k:v tuple is plotted separately. the values consist of a list, whose length must be 2. "
                        "they're interpreted as the output-tags for the A/H p-values respectively.\n"
                        "e.g. making pval plots for each channel: 'lx:att,htt; ll:att,htt; lj:att,htt...",
                        dest = "ahtag", default = "", required = True, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))

    parser.add_argument("--etat-tag", help = "input tag:output-tag key:value tuples to search, for etat p-values.\n"
                        "syntax is similar to --ah-tag, with the difference that output-tag value must be at most 1. "
                        "if omitted, tag is used as output-tag. there needs to be as many etat tuples as A/H tuples, or none at all.",
                        dest = "etatag", default = "", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))

    parser.add_argument("--drop",
                        help = "comma separated list of points to be dropped. 'XX, YY' means all points containing XX or YY are dropped.",
                        default = "", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s) ) )

    parser.add_argument("--plot-tag", help = "extra tag to append to plot names", dest = "ptag", default = "", required = False, type = prepend_if_not_empty)
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False, type = remove_spaces_quotes)
    parser.add_argument("--label", help = "labels to attach on plot for each A/H tags, semicolon separated", default = "", required = False, type = lambda s: tokenize_to_list(s, ';' ))

    parser.add_argument("--annotate", help = "text to annotate the plot, somewhere near the legend", dest = "annotate", default = "", required = False)

    parser.add_argument("--expected", help = "draw expected plots", dest = "expected", action = "store_true", required = False)
    parser.add_argument("--observed", help = "draw observed plots", dest = "observed", action = "store_true", required = False)

    parser.add_argument("--formal", help = "plot is for formal use - put the CMS text etc",
                        dest = "formal", action = "store_true", required = False)
    parser.add_argument("--cms-append", help = "text to append to the CMS text, if --formal is used", dest = "cmsapp", default = "", required = False)
    parser.add_argument("--luminosity", help = "integrated luminosity applicable for the plot, written if --formal is used", default = "138", required = False)

    parser.add_argument("--opaque-background", help = "make the background white instead of transparent",
                        dest = "transparent", action = "store_false", required = False)
    parser.add_argument("--plot-formats", help = "comma-separated list of formats to save the plots in", default = [".png"], dest = "fmt", required = False,
                        type = lambda s: [prepend_if_not_empty(fmt, '.') for fmt in tokenize_to_list(remove_spaces_quotes(s))])
    args = parser.parse_args()

    if not (args.expected or args.observed):
        raise RuntimeError("at least one of expected and observed must be true")

    if len(args.ahtag) == 0:
        raise RuntimeError("zero A/H tags. aborting")

    if len(args.etatag) != 0 and len(args.etatag) != len(args.ahtag):
        raise RuntimeError("nonzero number of etat tags, but also inequal to ah tags. aborting")

    if len(args.label) != 0 and len(args.label) != len(args.ahtag):
        raise RuntimeError("nonzero number of labels, but also inequal to ah tags. aborting")

    for ii in range(len(args.ahtag)):
        ahtag = parse_to_kv(args.ahtag[ii])
        if len(ahtag[1]) != 2:
            raise RuntimeError(f"expecting 2 output tags for A/H tag {ahtag[0]}, but found {ahtag[1]}. aborting")

        etatag = parse_to_kv(args.etatag[ii]) if len(args.etatag) != 0 else None
        if etatag is not None and len(etatag[1]) != 1:
            raise RuntimeError(f"expecting 1 output tag for etat tag {etatag[0]}, but found {etatag[1]}. aborting")

        ahdirs = sorted(glob.glob('A_m*_w*__H_m*_w*_' + ahtag[0]))
        apnt = [get_point(pnt.split('__')[0]) for pnt in ahdirs if not any([dd in pnt.split('__')[0] for dd in args.drop])]
        hpnt = [get_point(pnt.split('__')[1]) for pnt in ahdirs if not any([dd in pnt.split('__')[1] for dd in args.drop])]
        etadirs = sorted(glob.glob('A_m*_w*__H_m*_w*_' + etatag[0])) if etatag is not None else []

        if len(apnt) > 0 and len(hpnt) > 0:
            draw_pval0(args.function,
                          ["{ooo}/AH_pval0_{fff}_{t0}{t1}{f}".format(ooo = args.odir, fff = r"{fff}", t0 = ahtag[0], t1 = args.ptag, f = f) for f in args.fmt],
                          args.which, ahdirs, apnt, hpnt, ahtag, etatag, etadirs,
                          args.label[ii] if len(args.label) else "", args.annotate, args.expected, args.observed, args.formal, args.cmsapp, args.luminosity, args.transparent)
        pass
    pass
