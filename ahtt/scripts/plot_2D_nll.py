#!/usr/bin/env python3
# draw the dnll(gAH) for the 1D
# requires matplotlib > 3.3 e.g. source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_100 x86_64-centos7-gcc10-opt

from argparse import ArgumentParser
import os
import sys
import numpy as np
from scipy.interpolate import UnivariateSpline
import math
from functools import cmp_to_key

from ROOT import TFile, TTree

import glob
from collections import OrderedDict
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpt
import matplotlib.lines as mln
import matplotlib.colors as mcl

from utilspy import pmtofloat
from drawings import min_g, max_g, epsilon, axes, first, second, get_point, stock_labels, valid_nll_fname
from drawings import default_etat_measurement, etat_blurb
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes, remove_quotes

def read_nll(points, directories, parameters, intervals):
    result = [[], []]
    best_fit, fits = result
    pstr = "__".join(points)

    for ii, (directory, scenario, tag) in enumerate(directories):
        fexp = f"{directory}/{pstr}_{tag}_nll_{scenario}_{parameters[0]}_*_{parameters[1]}_*.root"
        files = [ifile for ifile in glob.glob(fexp) if valid_nll_fname(ifile, ninterval = 2)]
        best_fit.append(None)

        originals = []
        for ifile in files:
            dfile = TFile.Open(ifile)
            dtree = dfile.Get("limit")

            for i in dtree:
                valuednll = (getattr(dtree, parameters[0]), getattr(dtree, parameters[1]), 2. * dtree.deltaNLL)

                if dtree.quantileExpected >= 0.:
                    originals.append(valuednll)
                elif best_fit[ii] is None and dtree.quantileExpected == -1.:
                    best_fit[ii] = valuednll
            dfile.Close()
        originals.append(best_fit[ii])
        originals = sorted(originals, key = cmp_to_key(lambda t0, t1: t0[1] < t1[1] if t0[0] == t1[0] else t0[0] < t1[0]))
        mm0, mm1 = intervals
        originals = [(v0, v1, dnll) for v0, v1, dnll in originals if mm0[0] <= v0 <= mm0[1] and mm1[0] <= v1 <= mm1[1]]
        fits.append(originals)
    return result

def draw_nll(oname, points, directories, parameters, labels, intervals, maxsigma, formal, cmsapp, luminosity, a343bkg, transparent):
    dnlls = [nsigma * nsigma for nsigma in range(1, 6)]

    if not hasattr(draw_nll, "colors"):
        draw_nll.colors = OrderedDict([
            (1    , ["0"]),
            (2    , ["#cc0033", "0"]),
            (3    , ["#0033cc", "#cc0033", "0"]),
            (4    , ["#33cc00", "#0033cc", "#cc0033", "0"]),
        ])
        draw_nll.lines = ['solid', 'dashed', 'dashdot', 'dashdotdotted', 'dotted']

    ndir = len(directories)
    if ndir > len(draw_nll.colors):
        raise RuntimeError("too many scenarios")

    fig, ax = plt.subplots()
    handles = []
    sigmas = []
    name, xlabel = namelabel if len(namelabel) > 1 else namelabel + namelabel
    nlls = read_nll(points, directories, parameters, intervals)

    for ii, nll in enumerate(nlls):
        colortouse = draw_contour.colors[len(nlls)][ii]
        ax.plot(np.array([nll[0][ii][0]]), np.array([nll[0][ii][1]]), marker = 'X', markersize = 10.0, color = colortouse)
        if ii == 0:
            sigmas.append((mln.Line2D([0], [0], color = "0", marker='X', markersize = 10., linewidth = 0), "Best fit"))

        for isig in range(maxsigma):
            if ii == 0 and maxsigma > 1:
                if isig > 1:
                    sigmas.append((mln.Line2D([0], [0], color = "0", linestyle = draw_nll.lines[isig], linewidth = 2), r"$\pm" + str(isig + 1) + r"\sigma$"))

            dnll = dnlls[isig]

            ax.tricontour(np.array(first(nll)), np.array(second(nll)), np.array([nn[2] for nn in nll]),
                          levels = np.array([0., alpha]), colors = colortouse,
                          linestyles = draw_nll.lines[isig], linewidths = 2, alpha = 1. - (0.05 * isig))

            if len(labels) > 1 and isig == 0:
                handles.append((mln.Line2D([0], [0], color = colortouse, linestyle = 'solid', linewidth = 2), labels[ic]))

    plt.xlabel(labels[0], fontsize = 23, loc = "right")
    plt.ylabel(labels[1], fontsize = 23, loc = "top")
    ax.margins(x = 0, y = 0)

    if formal:
        ctxt = "{cms}".format(cms = r"$\textbf{CMS}$")
        ax.text(0.03 * max_g, 0.96 * max_g, ctxt, fontsize = 31, ha = 'left', va = 'top', usetex = True)

        if cmsapp != "":
            atxt = "{app}".format(app = r" $\textit{" + cmsapp + r"}$")
            ax.text(0.03 * max_g, 0.90 * max_g, atxt, fontsize = 26, ha = 'left', va = 'top', usetex = True)

        ltxt = "{lum}{ifb}".format(lum = luminosity, ifb = r" fb$^{\mathrm{\mathsf{-1}}}$ (13 TeV)")
        ax.text(0.985 * max_g, 0.97 * max_g, ltxt, fontsize = 26, ha = 'right', va = 'top', usetex = True)

        btxt = etat_blurb(a343bkg)
        bbln = [matplotlib.patches.Rectangle((0, 0), 1, 1, fc = "white", ec = "white", lw = 0, alpha = 0)] * len(btxt)
        ax.legend(bbln, btxt, loc = 'lower right', bbox_to_anchor = (0.825, 0.55, 0.15, 0.15),
                  fontsize = 14 if len(btxt) > 1 else 15, frameon = False,
                  handlelength = 0, handletextpad = 0, borderaxespad = 1.)

    ax.minorticks_on()
    ax.tick_params(axis = "both", which = "both", direction = "in", bottom = True, top = True, left = True, right = True)
    ax.tick_params(axis = "both", which = "major", width = 1, length = 8, labelsize = 18, pad = 10)
    ax.tick_params(axis = "both", which = "minor", width = 1, length = 3)

    ### -------------------------------------------------------

    for ii, nll in enumerate(nlls[1]):
        values = np.array([nn[0] for nn in nll])
        dnlls = np.array([nn[1] for nn in nll])
        measurement = get_interval(parameter = xlabel, best_fit = nlls[0][ii], fits = nll)
        color = colors[ii]
        style = styles[ii]
        label = labels[ii] + measurement
        ax.plot(values, dnlls, color = color, linestyle = style, linewidth = 2)
        handles.append((mln.Line2D([0], [0], color = color, linestyle = style, linewidth = 2), label))

    plt.xlim(rangex)
    plt.ylim(rangey)
    #ax.plot(rangex, [rangey[1], rangey[1]], color = "black", linestyle = 'solid', linewidth = 1)
    plt.xlabel(xlabel, fontsize = 21, loc = "right")
    #plt.ylabel(axes["dnll"] % 'A/H', fontsize = 21, loc = "top")
    plt.ylabel("2dNLL", fontsize = 21, loc = "top")
    ax.margins(x = 0, y = 0)

    legend = ax.legend(first(handles), second(handles),
	               loc = legendloc, ncol = 1 if ndir <= len(draw_nll.settings) else 2, borderaxespad = 1., fontsize = 18, frameon = False,
                       title = legendtitle, title_fontsize = 21)
    ax.add_artist(legend)
    ax.minorticks_on()
    ax.tick_params(axis = "both", which = "both", direction = "in", bottom = True, top = True, left = True, right = True)
    ax.tick_params(axis = "both", which = "major", width = 1, length = 8, labelsize = 18)
    ax.tick_params(axis = "both", which = "minor", width = 1, length = 3)

    fig.set_size_inches(8., 8.)
    fig.tight_layout()
    fig.savefig(oname, transparent = transparent)
    fig.clf()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "signal point pair", default = "", required = True, type = lambda s: sorted(tokenize_to_list( remove_spaces_quotes(s) )))

    parser.add_argument("--tag", help = "(input tag, scenario, output tag) triplet to search. the pairs are semicolon separated, and tags/scenario colon-separated, "
                        "so e.g. when there are 2 tags: 't1:s1:o1;t2:s2:o2 . output tag may be empty, in which case input tag is used for it.",
                        dest = "itag", default = "", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), token = ';' ))
    parser.add_argument("--plot-tag", help = "extra tag to append to plot names", dest = "ptag", default = "", required = False, type = prepend_if_not_empty)
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False, type = remove_spaces_quotes)
    parser.add_argument("--tag-label", help = "labels to attach on plot for each tag triplet, semicolon separated", default = "", required = False,
                        type = lambda s: tokenize_to_list(s, token = ';' ))

    parser.add_argument("--parameters", help = "comma-separated names of parameters. must be exactly 2.",
                        dest = "params", type = lambda s: tokenize_to_list(remove_spaces_quotes(s)), default = ["g1", "g2"], required = False)
    parser.add_argument("--parameter-labels", help = "semicolon-separated labels of those parameters. must be either not given, or exactly 2.",
                        dest = "labels", type = lambda s: tokenize_to_list(remove_spaces_quotes(s), ';'), default = [], required = False)
    parser.add_argument("--intervals", help = "semicolon-separated intervals of above 2 parameters to draw. an interval is specified as comma-separated minmax. must be either not given, or exactly 2.",
                        dest = "intervals", type = lambda s: [tokenize_to_list(minmax) for minmax in tokenize_to_list(remove_spaces_quotes(s), token = ';')], default = [], required = False)

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

    parser.add_argument("--opaque-background", help = "make the background white instead of transparent",
                        dest = "transparent", action = "store_false", required = False)
    parser.add_argument("--plot-format", help = "format to save the plots in", default = ".png", dest = "fmt", required = False, type = lambda s: prepend_if_not_empty(s, '.'))

    args = parser.parse_args()
    points = args.point
    if len(points) != 2:
        raise RuntimeError("this script is to be used with exactly two A/H points!")
    pstr = "__".join(points)

    if len(args.itag) != len(args.label):
        raise RuntimeError("length of tags isnt the same as labels. aborting")

    if len(args.labels) == 0:
        args.labels = stock_labels(args.params, args.point)

    dirs = [tag.split(':') for tag in args.itag]
    dirs = [tag + tag[:1] if len(tag) == 2 else tag for tag in dirs]
    dirs = [[f"{pstr}_{tag[0]}"] + tag[1:] for tag in dirs]

    draw_nll(f"{args.odir}/{pstr}_nll_{'__'.join(args.parameters)}{args.ptag}{args.fmt}",
             points, dirs, args.parameters, args.labels, args.intervals, args.maxsigma, args.formal, args.cmsapp, args.luminosity, args.a343bkg, args.transparent)
    pass
