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
import matplotlib.pyplot as plt
import matplotlib.patches as mpt
import matplotlib.lines as mln
import matplotlib.colors as mcl

from drawings import min_g, max_g, epsilon, axes, first, second, get_point

def read_contour(cfiles):
    contours = [OrderedDict() for cf in cfiles]

    for ii, cf in enumerate(cfiles):
        with open(cf) as ff:
            cc = json.load(ff)

        contours[ii]["best_fit"] = [cc["best_fit_g1_g2_dnll"][0], cc["best_fit_g1_g2_dnll"][1]]
        contours[ii]["g1"] = []
        contours[ii]["g2"] = []
        contours[ii]["eff"] = []
        contours[ii]["avg"] = 0

        for gv in cc["g-grid"].keys():
            contours[ii]["g1"].append( gv.replace(" ", "").split(",")[0] )
            contours[ii]["g2"].append( gv.replace(" ", "").split(",")[1] )
            contours[ii]["eff"].append( cc["g-grid"][gv]["pass"] / cc["g-grid"][gv]["total"] )
            contours[ii]["avg"] += cc["g-grid"][gv]["total"]

        contours[ii]["avg"] /= len(contours[ii]["eff"])

    return contours

def draw_contour(oname, pair, cfiles, labels, maxsigma, formal, cmsapp, luminosity, transparent):
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
        #xv, yv = np.meshgrid(np.array(set(contour["g1"])), np.array(set(contour["g2"])))
        #zv = np.zeros_like(xv)

        #for ir, xr in enumerate(xv):
        #    for ic, xc in enumerate(xr):
        #        g1 = xv[ir][ic]
        #        g2 = yv[ir][ic]

        #        for i1, i2, ie in zip(contour["g1"], contour["g2"], contour["eff"]):
        #            if i1 == g1 and i2 == g2:
        #                zv[ir][ic] = ie
        #                break

        for isig in range(maxsigma):
            if ic == 0 and maxsigma > 1:
                sigmas.append((mln.Line2D([0], [0], color = "0", linestyle = draw_contour.lines[isig], linewidth = 2), r"$\pm" + str(isig + 1) + r"\sigma$"))

            alpha = alphas[isig]

            if contour["avg"] < (4.5 / alpha):
                print("average toy count of " + str(contour["avg"]) + " likely insufficient to determine contour with CL " + str(alpha) + "\n")

            #cf = ax.contourf(xv, yv, zv, [-1., alpha], colors = ["#ffffff"], alpha = 0.)
            #ax.contour(cf, colors = draw_contour.colors[len(contours)][ic], linestyles = draw_contour.lines[isig], linewidths = 2, alpha = 1. - (0.05 * isig))

            #cf = ax.tricontourf(contour["g1"], contour["g2"], contour["eff"],
            #                    levels = np.array([-1., alpha]), colors = ["#ffffff"], alpha = 0.)
            cf = ax.tricontour(contour["g1"], contour["g2"], contour["eff"],
                               levels = np.array([alpha, 2.]), colors = draw_contour.colors[len(contours)][ic],
                               linestyles = draw_contour.lines[isig], linewidths = 2, alpha = 1. - (0.05 * isig))
            #ax.tricontour(cf, colors = draw_contour.colors[len(contours)][ic], linestyles = draw_contour.lines[isig], linewidths = 2, alpha = 1. - (0.05 * isig))

            if len(labels) > 1 and isig == 0:
                handles.append((mln.Line2D([0], [0], color = draw_contour.colors[len(contours)][ic], linestyle = solid, linewidth = 2), labels[ic]))

    plt.xlabel(axes["coupling"] % get_point(pair[0])[0], fontsize = 21, loc = "right", labelpad = 6.0)
    plt.ylabel(axes["coupling"] % get_point(pair[1])[0], fontsize = 21, loc = "top", labelpad = 6.0)
    ax.margins(x = 0, y = 0)

    if formal:
        ctxt = r"\textbf{CMS}"
        ctxt += r" \textit{" + cmsapp + r"}" if cmsapp != "" else "" 
        plt.text(0.02 * max_g, 0.98 * max_g, ctxt, fontsize = 32, ha = 'left', va = 'top')

        ltxt = luminosity + r" fb$^{\mathrm{\mathsf{-1}}}$ (13 TeV)"
        plt.text(0.98 * max_g, 0.98 * max_g, ctxt, fontsize = 32, ha = 'right', va = 'top')

    if len(handles) > 0 and len(sigmas) > 0:
        pass
    elif len(handles) > 0:
        pass
    elif len(sigmas) > 0:
        pass

    ax.minorticks_on()
    ax.tick_params(axis = "both", which = "both", direction = "in", bottom = True, top = False, left = True, right = True)
    ax.tick_params(axis = "both", which = "major", width = 1, length = 8, labelsize = 18)
    ax.tick_params(axis = "both", which = "minor", width = 1, length = 3)

    fig.set_size_inches(8., 8.)
    fig.tight_layout()
    fig.savefig(oname, transparent = transparent)
    fig.clf()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--contour", help = "the json files containing the contour information, semicolon separated. must be of the same signal pair.", default = "", required = True)
    parser.add_argument("--otag", help = "extra tag to append to plot names", default = "", required = False)
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False)
    parser.add_argument("--max-sigma", help = "max number of sigmas to be drawn on the contour", dest = "maxsigma", default = 2, type = int, required = False)
    parser.add_argument("--label", help = "labels to attach on plot for each json input, semicolon separated", default = "", required = False)
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
                 args.formal, args.cmsapp, args.luminosity, args.transparent)
