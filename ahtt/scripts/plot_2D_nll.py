#!/usr/bin/env python3
# draw the dnll(gAH) for the 1D
# requires matplotlib > 3.3 e.g. source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_100 x86_64-centos7-gcc10-opt

from argparse import ArgumentParser
import os
import sys
import numpy as np
from scipy.interpolate import SmoothBivariateSpline as interpolator
import math
from functools import cmp_to_key

from ROOT import TFile, TTree

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

from utilspy import pmtofloat
from drawings import min_g, max_g, epsilon, axes, first, second, third, pruned, withinerror, get_point, stock_labels, valid_nll_fname
from drawings import default_etat_measurement, etat_blurb
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes, remove_quotes
from hilfemir import combine_help_messages

def read_nll(points, directories, parameters, intervals, drops, prunesmooth = False, maxsigma = 5):
    result = [[], []]
    best_fit, fits = result
    pstr = "__".join(points)

    for ii, (directory, scenario, tag) in enumerate(directories):
        drop = None
        if len(drops) > 0:
            drop = drops[ii if ii < len(drops) else -1]

        fexp = f"{directory}/{pstr}_{tag}_nll_{scenario}_{parameters[0]}_*_{parameters[1]}_*.root"
        files = [ifile for ifile in glob.glob(fexp) if valid_nll_fname(ifile, tag = tag, ninterval = 2)]
        best_fit.append(None)

        originals = []
        for ifile in files:
            dfile = TFile.Open(ifile)
            dtree = dfile.Get("limit")

            for i in dtree:
                valuednll = (round(getattr(dtree, parameters[0]), 5), round(getattr(dtree, parameters[1]), 5), round(2. * dtree.deltaNLL, 5))

                if dtree.quantileExpected >= 0.:
                    if drop is not None and any([window[0][0] < valuednll[0] < window[0][1] and window[1][0] < valuednll[1] < window[1][1] for window in drop]):
                        continue
                    originals.append(valuednll)
                elif best_fit[ii] is None and dtree.quantileExpected == -1.:
                    best_fit[ii] = valuednll
            dfile.Close()
        originals.append(best_fit[ii])
        originals = sorted(originals, key = cmp_to_key(lambda t0, t1: t0[1] < t1[1] if t0[0] == t1[0] else t0[0] < t1[0]))

        if prunesmooth:
            alphas = [2.29575, 6.18008, 11.82922, 19.33391, 28.74371]
            originals = [oo for oo in originals if oo[2] < 10 * math.ceil(alphas[maxsigma - 1 if 1 < maxsigma < 6 else -1] / 10)]
            prunes = [pruned(originals, keep = best_fit[ii]) for _ in range(1)]
            splines = [
                interpolator(np.array(first(prunes[_])), np.array(second(prunes[_])), np.array(third(prunes[_])), kx = 4, ky = 4, s = len(originals) / 100)
                for _ in range(len(prunes))
            ]

            interpolated = []
            for oo in range(len(originals)):
                x, y, z0 = originals[oo][0], originals[oo][1], originals[oo][2]
                if x != best_fit[ii][0] and y != best_fit[ii][1]:
                    zs = [ss(x, y).squeeze() for ss in splines]
                    zs = [zz for zz in zs if zz < 0 or withinerror(z0, zz)]
                    if len(zs) == 0:
                        print(x, y, z0, [ss(x, y).squeeze() for ss in splines])
                        continue
                interpolated.append((x, y, z0))
            del splines
        fits.append(interpolated if prunesmooth else originals)
    return result

def draw_nll(onames, points, directories, tlabel, parameters, plabels, intervals, drops, prunesmooth, maxsigma, bestfit, formal, cmsapp, luminosity, a343bkg, transparent):
    alphas = [2.29575, 6.18008, 11.82922, 19.33391, 28.74371]

    if not hasattr(draw_nll, "colors"):
        draw_nll.colors = OrderedDict([
            (1    , ["0"]),
            (2    , ["#696969", "0"]),
            (3    , ["#0033cc", "#cc0033", "0"]),
            (4    , ["#33cc00", "#0033cc", "#cc0033", "0"]),
        ])
        draw_nll.lines = ['solid', 'dashed', 'dashdot', 'dotted']
        #draw_nll.lines = ['solid', 'dashed', (0, (3, 1, 1, 1)), 'dotted']

    ndir = len(directories)
    if ndir > len(draw_nll.colors):
        raise RuntimeError("too many scenarios")

    fig, ax = plt.subplots()
    handles = []
    sigmas = []
    nlls = read_nll(points, directories, parameters, intervals, drops, prunesmooth, maxsigma)

    xlength = intervals[0][1] - intervals[0][0]
    ylength = intervals[1][1] - intervals[1][0]

    xmin, xmax = intervals[0]
    ymin, ymax = intervals[1]

    #if True:
    #    ax.plot(np.array([xmin, xmax]), np.array([0., 0.]), color = "#3B444B", linewidth = 0.5)
    #    ax.plot(np.array([0., 0.]), np.array([ymin, ymax]), color = "#3B444B", linewidth = 0.5)

    for ii, (best_fit, nll) in enumerate(zip(nlls[0], nlls[1])):
        colortouse = draw_nll.colors[len(nlls[1])][ii]

        if bestfit:
            ax.plot(np.array([best_fit[0]]), np.array([best_fit[1]]), marker = 'X', markersize = 10.0, color = colortouse)
            if ii == 0:
                sigmas.append((mln.Line2D([0], [0], color = "0", marker='X', markersize = 10., linewidth = 0), "Best fit"))

        for isig in range(maxsigma):
            if maxsigma > 3 and (isig + 1) % 2 == 0:
                continue
            iline = isig if maxsigma <= 3 else isig // 2

            if ii == 0 and maxsigma > 1:
                sigmas.append((mln.Line2D([0], [0], color = "0", linestyle = draw_nll.lines[iline], linewidth = 2), r"$\pm" + str(isig + 1) + r"$ SD"))

            alpha = alphas[isig]

            ax.tricontour(np.array(first(nll)), np.array(second(nll)), np.array(third(nll)),
                          levels = np.array([0., alpha]), colors = colortouse,
                          linestyles = [draw_nll.lines[iline]], linewidths = 2, alpha = 1. - (0.05 * isig))

            if len(tlabel) > 1 and isig == 0:
                handles.append((mln.Line2D([0], [0], color = colortouse, linestyle = 'solid', linewidth = 2), tlabel[ii]))

    plt.xlabel(plabels[0], fontsize = 23, loc = "right")
    plt.ylabel(plabels[1], fontsize = 23, loc = "top")
    plt.xlim(*intervals[0])
    plt.ylim(*intervals[1])
    ax.margins(x = 0, y = 0)

    # for the pas A/H365 plot, lx
    #bbox_sigmas = (0.175, 0.575, 0.225, 0.3)
    #bbox_expobs = (0.275, 0., 0.25, 0.2)
    #bbox_noeta = (0.85, 0.75, 0.15, 0.15)

    # for the paper etachi, ll
    bbox_sigmas = (0.96, 0.05, 0.04, 0.25)
    bbox_expobs = (0.91, 0.725, 0.1, 0.2)
    bbox_noeta = (0.85, 0.75, 0.15, 0.15)

    if len(handles) > 0 and len(sigmas) > 0:
        legend1 = ax.legend(first(sigmas), second(sigmas), loc = 'best', bbox_to_anchor = bbox_sigmas, fontsize = 19, handlelength = 2.08, handletextpad = 0.4, borderaxespad = 1., frameon = False)
        ax.add_artist(legend1)

        legend2 = ax.legend(first(handles), second(handles), loc = 'best', bbox_to_anchor = bbox_expobs, fontsize = 19, handlelength = 2.08, handletextpad = 0.4, borderaxespad = 1., frameon = False)
        ax.add_artist(legend2)
    elif len(handles) > 0:
        ax.legend(first(handles), second(handles), loc = 'lower right', fontsize = 21, handlelength = 2., borderaxespad = 1., frameon = False)
    elif len(sigmas) > 0:
        ax.legend(first(sigmas), second(sigmas), loc = 'lower right', fontsize = 21, handlelength = 2., borderaxespad = 1., frameon = False)

    if formal:
        ctxt = "{cms}".format(cms = r"$\textbf{CMS}$")
        ax.text((0.03 * xlength) + xmin, (0.96 * ylength) + ymin, ctxt, fontsize = 31, ha = 'left', va = 'top', usetex = True)

        if cmsapp != "":
            atxt = "{app}".format(app = r" $\textit{" + cmsapp + r"}$")
            ax.text((0.03 * xlength) + xmin, (0.90 * ylength) + ymin, atxt, fontsize = 26, ha = 'left', va = 'top', usetex = True)

        ltxt = "{lum}{ifb}".format(lum = luminosity, ifb = r" fb$^{\mathrm{\mathsf{-1}}}$ (13 TeV)")
        ax.text((0.985 * xlength) + xmin, (0.97 * ylength) + ymin, ltxt, fontsize = 26, ha = 'right', va = 'top', usetex = True)

        #btxt = etat_blurb(a343bkg)
        #bbln = [matplotlib.patches.Rectangle((0, 0), 1, 1, fc = "white", ec = "white", lw = 0, alpha = 0)] * len(btxt)
        #ax.legend(bbln, btxt, loc = 'lower right', bbox_to_anchor = bbox_noeta,
        #          fontsize = 14 if len(btxt) > 1 else 16, frameon = False,
        #          handlelength = 0, handletextpad = 0, borderaxespad = 1.)

    ax.minorticks_on()
    ax.tick_params(axis = "both", which = "both", direction = "in", bottom = True, top = True, left = True, right = True)
    ax.tick_params(axis = "both", which = "major", width = 1, length = 8, labelsize = 18, pad = 10)
    ax.tick_params(axis = "both", which = "minor", width = 1, length = 3)

    if xlength < 5 and xmin == int(xmin) and xmax == int(xmax):
        ax.xaxis.set_major_locator(mtc.MultipleLocator(base = 1))
    if ylength < 5 and ymin == int(ymin) and ymax == int(ymax):
        ax.yaxis.set_major_locator(mtc.MultipleLocator(base = 1))

    fig.set_size_inches(8., 8.)
    fig.set_dpi(600)
    fig.tight_layout()

    for oname in onames:
        fig.savefig(oname, transparent = transparent)
    fig.clf()

def drop_intervals(arg):
    result = tokenize_to_list(remove_spaces_quotes(arg), token = '--')
    result = [tokenize_to_list(rr, token = ':') for rr in result]
    result = [[tokenize_to_list(ii, token = ';') for ii in rr] for rr in result]
    result = [[[tokenize_to_list(minmax, astype = float) for minmax in ii] for ii in rr] for rr in result]
    return result

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "signal point pair", default = "", required = True, type = lambda s: sorted(tokenize_to_list( remove_spaces_quotes(s) )))

    parser.add_argument("--tag", help = "(input tag, scenario, output tag) triplet to search. the pairs are semicolon separated, and tags/scenario colon-separated, "
                        "so e.g. when there are 2 tags: 't1:s1:o1;t2:s2:o2 . output tag may be empty, in which case input tag is used for it.",
                        dest = "itag", default = "", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), token = ';' ))
    parser.add_argument("--plot-tag", help = "extra tag to append to plot names", dest = "ptag", default = "", required = False, type = prepend_if_not_empty)
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False, type = remove_spaces_quotes)
    parser.add_argument("--tag-label", help = "labels to attach on plot for each tag triplet, semicolon separated", default = "", dest = "tlabel",
                        required = False, type = lambda s: tokenize_to_list(s, token = ';' ))

    parser.add_argument("--parameters", help = "comma-separated names of parameters. must be exactly 2.",
                        dest = "params", type = lambda s: tokenize_to_list(remove_spaces_quotes(s)), default = ["g1", "g2"], required = False)
    parser.add_argument("--parameter-labels", help = "semicolon-separated labels of those parameters. must be either not given, or exactly 2.",
                        dest = "plabels", type = lambda s: tokenize_to_list(remove_spaces_quotes(s), ';'), default = [], required = False)
    parser.add_argument("--intervals", help = "semicolon-separated intervals of above 2 parameters to draw. an interval is specified as comma-separated minmax. must be either not given, or exactly 2.",
                        dest = "intervals", type = lambda s: [tokenize_to_list(minmax, astype = float) for minmax in tokenize_to_list(remove_spaces_quotes(s), token = ';')], default = [], required = False)
    parser.add_argument("--prune-smoothing",
                        help = "smooth the data points by making splines off the pruned points, and then taking their averages.",
                        dest = "prunesmooth", action = "store_true", required = False)

    parser.add_argument("--drops",
                        help = "syntax: int00:int01:...:int0N -- int10:int11:int1N -- intM0:intM1:intMN, where"
                        "0..M refers to the number of contours (can be less than len(--tag), in which case the last one is used),"
                        "0..N refers to the individual intervals, whose syntax is the same as --intervals, specifying the regions where NLL points are not drawn",
                        dest = "drops", type = drop_intervals, default = [], required = False)

    parser.add_argument("--draw-best-fit", help = "draw the best fit point.", dest = "bestfit", action = "store_true", required = False)
    parser.add_argument("--max-sigma", help = "max number of sigmas to be drawn on the contour", dest = "maxsigma", default = "5", required = False, type = lambda s: int(remove_spaces_quotes(s)))

    parser.add_argument("--formal", help = "plot is for formal use - put the CMS text etc",
                        dest = "formal", action = "store_true", required = False)
    parser.add_argument("--cms-append", help = "text to append to the CMS text, if --formal is used", dest = "cmsapp", default = "", required = False)
    parser.add_argument("--luminosity", help = "integrated luminosity applicable for the plot, written if --formal is used", default = "138", required = False)

    parser.add_argument("--A343-background",
                        help = "a comma-separated list of 4 values for etat background text, written if --formal is used"
                        "syntax: (bool, 1 or 0, whether it is included as bkg, best fit xsec, xsec uncertainty (lo/hi)",
                        dest = "a343bkg", default = default_etat_measurement(), required = False,
                        type = default_etat_measurement)

    parser.add_argument("--arbitrary-resonance-normalization", help = combine_help_messages["--arbitrary-resonance-normalization"],
                        dest = "arbnorm", default = 5, required = False,
                        type = lambda s: 5 if s.lower() in ["", "true", "default"] else int(s))

    parser.add_argument("--opaque-background", help = "make the background white instead of transparent",
                        dest = "transparent", action = "store_false", required = False)
    parser.add_argument("--plot-formats", help = "comma-separated list of formats to save the plots in", default = [".png"], dest = "fmt", required = False,
                        type = lambda s: [prepend_if_not_empty(fmt, '.') for fmt in tokenize_to_list(remove_spaces_quotes(s))])

    args = parser.parse_args()
    points = args.point
    if len(points) != 2:
        raise RuntimeError("this script is to be used with exactly two A/H points!")
    pstr = "__".join(points)

    if len(args.itag) != len(args.tlabel):
        raise RuntimeError("length of tags isnt the same as tag labels. aborting")

    if len(args.plabels) == 0:
        args.plabels = stock_labels(args.params, args.point, args.arbnorm)

    dirs = [tag.split(':') for tag in args.itag]
    dirs = [tag + tag[:1] if len(tag) == 2 else tag for tag in dirs]
    dirs = [[f"{pstr}_{tag[0]}"] + tag[1:] for tag in dirs]

    draw_nll([f"{args.odir}/{pstr}_nll_{'__'.join(args.params)}{args.ptag}{fmt}" for fmt in args.fmt],
             points, dirs, args.tlabel, args.params, args.plabels, args.intervals, args.drops, args.prunesmooth, args.maxsigma, args.bestfit,
             args.formal, args.cmsapp, args.luminosity, args.a343bkg, args.transparent)
    pass
