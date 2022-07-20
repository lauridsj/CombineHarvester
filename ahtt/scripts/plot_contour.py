#!/usr/bin/env python3
# draw the model-independent FC scan countour on gAH plot
# requires matplotlib > 3.3 e.g. source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_100 x86_64-centos7-gcc10-opt

from argparse import ArgumentParser
import os
import sys
import numpy as np
from scipy.interpolate import UnivariateSpline
import math

import glob
from collections import OrderedDict
import json

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
plt.rcParams["mathtext.default"] = 'regular'
plt.rc('font', **{'family' : 'sans-serif', 'sans-serif' : ['Helvetica']})

import matplotlib.patches as mpt
import matplotlib.lines as mln
import matplotlib.colors as mcl

from drawings import min_g, max_g, epsilon, axes, first, second, get_point, str_point

def read_contour(cfiles):
    contours = [OrderedDict() for cf in cfiles]

    for ii, cf in enumerate(cfiles):
        with open(cf) as ff:
            cc = json.load(ff)

        contours[ii]["best_fit"] = [cc["best_fit_g1_g2_dnll"][0], cc["best_fit_g1_g2_dnll"][1]]
        contours[ii]["g1"] = []
        contours[ii]["g2"] = []
        contours[ii]["eff"] = []
        contours[ii]["min"] = sys.maxsize

        for gv in cc["g-grid"].keys():
            contours[ii]["g1"].append( gv.replace(" ", "").split(",")[0] )
            contours[ii]["g2"].append( gv.replace(" ", "").split(",")[1] )
            contours[ii]["eff"].append( cc["g-grid"][gv]["pass"] / cc["g-grid"][gv]["total"] )
            contours[ii]["min"] = min(contours[ii]["min"], cc["g-grid"][gv]["total"])

    return contours

def draw_contour(oname, pair, cfiles, labels, maxsigma, scatter, formal, cmsapp, luminosity, transparent):
    contours = read_contour(cfiles)
    ncontour = len(contours)
    alphas = [1 - pval for pval in [0.6827, 0.9545, 0.9973, 0.999937, 0.9999997]]

    if not hasattr(draw_contour, "colors"):
        draw_contour.colors = OrderedDict([
            (1    , ["0"]),
            (2    , ["#cc0033", "#0033cc"]),
            (3    , ["0", "#cc0033", "#0033cc"]),
            (4    , ["0", "#cc0033", "#0033cc", "#33cc00"]),
        ])
        draw_contour.lines = ['solid', 'dashed', 'dashdot', 'dashdotdotted', 'dotted']


    fig, ax = plt.subplots()
    handles = []
    sigmas = []

    for ic, contour in enumerate(contours):
        for isig in range(maxsigma):
            if ic == 0 and maxsigma > 1:
                sigmas.append((mln.Line2D([0], [0], color = "0", linestyle = draw_contour.lines[isig], linewidth = 2), r"$\pm" + str(isig + 1) + r"\sigma$"))

            alpha = alphas[isig]

            if contour["min"] < (4.5 / alpha):
                print("minimum toy count of " + str(contour["min"]) + " likely insufficient to determine contour with CL " + str(alpha) + "\n")

            ax.tricontour(contour["g1"], contour["g2"], contour["eff"],
                          levels = np.array([alpha, 2.]), colors = draw_contour.colors[len(contours)][ic],
                          linestyles = draw_contour.lines[isig], linewidths = 2, alpha = 1. - (0.05 * isig))

            if len(labels) > 1 and isig == 0:
                handles.append((mln.Line2D([0], [0], color = draw_contour.colors[len(contours)][ic], linestyle = 'solid', linewidth = 2), labels[ic]))

    plt.xlabel(axes["coupling"] % str_point(pair[0]), fontsize = 23, loc = "right")
    plt.ylabel(axes["coupling"] % str_point(pair[1]), fontsize = 23, loc = "top")
    ax.margins(x = 0, y = 0)

    if formal:
        ctxt = "{cms}".format(cms = r"\textbf{CMS}")
        ax.text(0.02 * max_g, 0.98 * max_g, ctxt, fontsize = 36, ha = 'left', va = 'top')

        if cmsapp != "":
            atxt = "{app}".format(app = r" \textit{" + cmsapp + r"}")
            ax.text(0.02 * max_g, 0.91 * max_g, atxt, fontsize = 26, ha = 'left', va = 'top')

        ltxt = "{lum}{ifb}".format(lum = luminosity, ifb = r" fb$^{\mathrm{\mathsf{-1}}}$ (13 TeV)")
        ax.text(0.98 * max_g, 0.98 * max_g, ltxt, fontsize = 26, ha = 'right', va = 'top')

    if len(handles) > 0 and len(sigmas) > 0:
        pass
    elif len(handles) > 0:
        pass
    elif len(sigmas) > 0:
        pass

    ax.minorticks_on()
    ax.tick_params(axis = "both", which = "both", direction = "in", bottom = True, top = False, left = True, right = True)
    ax.tick_params(axis = "both", which = "major", width = 1, length = 8, labelsize = 18, pad = 10)
    ax.tick_params(axis = "both", which = "minor", width = 1, length = 3)

    fig.set_size_inches(8., 8.)
    fig.tight_layout()

    # matplotlib is insane, so this must happen last, don't ask why
    if scatter:
        ax.autoscale(False)
        for ic, contour in enumerate(contours):
            ax.plot(np.array(contour["g1"]), np.array(contour["g2"]),
                    marker = '.', ls = '', lw = 0., color = draw_contour.colors[len(contours)][ic], alpha = 0.5)
        ax.axis([min_g, max_g, min_g, max_g])

    fig.savefig(oname, transparent = transparent)
    fig.clf()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--contour", help = "the json files containing the contour information, semicolon separated. must be of the same signal pair.", default = "", required = True)
    parser.add_argument("--otag", help = "extra tag to append to plot names", default = "", required = False)
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False)
    parser.add_argument("--max-sigma", help = "max number of sigmas to be drawn on the contour", dest = "maxsigma", default = 2, type = int, required = False)
    parser.add_argument("--label", help = "labels to attach on plot for each json input, semicolon separated", default = "", required = False)

    parser.add_argument("--draw-scatter", help = "draw the scatter points used to build the contours",
                        dest = "scatter", action = "store_true", required = False)

    parser.add_argument("--formal", help = "plot is for formal use - put the CMS text etc",
                        dest = "formal", action = "store_true", required = False)
    parser.add_argument("--cms-append", help = "text to append to the CMS text, if --formal is used", dest = "cmsapp", default = "", required = False)
    parser.add_argument("--luminosity", help = "integrated luminosity applicable for the plot, written if --formal is used", default = "XXX", required = False)

    parser.add_argument("--transparent-background", help = "make the background transparent instead of white",
                        dest = "transparent", action = "store_true", required = False)
    parser.add_argument("--plot-format", help = "format to save the plots in", default = "pdf", dest = "fmt", required = False)

    args = parser.parse_args()
    if (args.otag != "" and not args.otag.startswith("_")):
        args.otag = "_" + args.otag

    if (args.fmt != "" and not args.fmt.startswith(".")):
        args.fmt = "." + args.fmt

    contours = args.contour.split(';')
    labels = args.label.split(';')

    if len(contours) != len(labels):
        raise RuntimeError("there aren't as many input contours as there are labels. aborting")

    pairs = [os.path.basename(os.path.dirname(cc)).split(".")[0].split("__") for cc in contours]
    pairs = [[pp[0], "_".join(pp[1].split("_")[:3])] for pp in pairs]

    if not all([pp == pairs[0] for pp in pairs]):
        raise RuntimeError("provided contours are not all of the same pair of points!!")

    draw_contour("{ooo}/{prs}_fc-contour{tag}{fmt}".format(ooo = args.odir, prs = "__".join(pairs[0]), tag = args.otag, fmt = args.fmt), pairs[0], contours, labels, args.maxsigma,
                 args.scatter, args.formal, args.cmsapp, args.luminosity, args.transparent)
