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
            contours[ii]["g1"].append( float(gv.replace(" ", "").split(",")[0]) )
            contours[ii]["g2"].append( float(gv.replace(" ", "").split(",")[1]) )
            contours[ii]["eff"].append( cc["g-grid"][gv]["pass"] / cc["g-grid"][gv]["total"] )
            contours[ii]["min"] = min(contours[ii]["min"], cc["g-grid"][gv]["total"])

    return contours

def draw_contour(oname, pair, cfiles, labels, maxsigma, propersig, drawcontour, bestfit, scatter, formal, cmsapp, luminosity, transparent):
    contours = read_contour(cfiles)
    ncontour = len(contours)
    alphas = [0.6827, 0.9545, 0.9973, 0.999937, 0.9999997] if propersig else [0.68, 0.95, 0.9973, 0.999937, 0.9999997]
    alphas = [1. - pval for pval in alphas]

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
        if bestfit:
            ax.plot(np.array([contour["best_fit"][0]]), np.array([contour["best_fit"][1]]),
                    marker = 'X', markersize = 10.0, color = draw_contour.colors[len(contours)][ic])
            if ic == 0:
                sigmas.append((mln.Line2D([0], [0], color = "0", marker='X', markersize = 10., linewidth = 0), "Best fit"))

        if scatter:
            yv = list(set([yy for yy in contour["g2"]]))
            yv.sort()

            for yy in yv:
                ps = [(x, y) for x, y in zip(contour["g1"], contour["g2"]) if y == yy]
                xs = first(ps)
                ys = second(ps)

                xs.sort()

                ax.plot(np.array(xs), np.array(ys),
                        marker = '.', ls = '', lw = 0., color = draw_contour.colors[len(contours)][ic], alpha = 0.5)

        for isig in range(maxsigma):
            if ic == 0 and maxsigma > 1:
                if isig > 1:
                    sigmas.append((mln.Line2D([0], [0], color = "0", linestyle = draw_contour.lines[isig], linewidth = 2), r"$\pm" + str(isig + 1) + r"\sigma$"))
                elif isig == 1:
                    sigmas.append((mln.Line2D([0], [0], color = "0", linestyle = draw_contour.lines[isig], linewidth = 2), r"95% CL"))
                elif isig == 0:
                    sigmas.append((mln.Line2D([0], [0], color = "0", linestyle = draw_contour.lines[isig], linewidth = 2), r"68% CL"))

            alpha = alphas[isig]

            if contour["min"] < (4.5 / alpha):
                print("minimum toy count of " + str(contour["min"]) + " likely insufficient to determine contour with CL " + str(alpha) + "\n")

            if drawcontour:
                ax.tricontour(np.array(contour["g1"]), np.array(contour["g2"]), contour["eff"],
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

    if not scatter:
        if len(handles) > 0 and len(sigmas) > 0:
            legend1 = ax.legend(first(sigmas), second(sigmas), loc = 'upper right', fontsize = 21, handlelength = 2.4, borderaxespad = 1., frameon = False)
            ax.add_artist(legend1)

            ax.legend(first(handles), second(handles), loc = 'lower right', fontsize = 21, handlelength = 2., borderaxespad = 1., frameon = False)
        elif len(handles) > 0:
            ax.legend(first(handles), second(handles), loc = 'lower right', fontsize = 21, handlelength = 2.4, borderaxespad = 1., frameon = False)
        elif len(sigmas) > 0:
            ax.legend(first(sigmas), second(sigmas), loc = 'lower right', fontsize = 21, handlelength = 2.4, borderaxespad = 1., frameon = False)

    ax.minorticks_on()
    ax.tick_params(axis = "both", which = "both", direction = "in", bottom = True, top = True, left = True, right = True)
    ax.tick_params(axis = "both", which = "major", width = 1, length = 8, labelsize = 18, pad = 10)
    ax.tick_params(axis = "both", which = "minor", width = 1, length = 3)

    fig.set_size_inches(8., 8.)
    fig.tight_layout()

    fig.savefig(oname, transparent = transparent)
    fig.clf()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point",
                        help = "desired pairs of signal points to run on, comma (between points) and semicolon (between pairs) separated\n"
                        "another syntax is: m1,m2,...,mN;w1,w2,...,wN;m1,m2,...,mN;w1,w2,...,wN, where:\n"
                        "the first mass and width strings refer to the A grid, and the second to the H grid.\n"
                        "both mass and width strings must include their m and w prefix, and for width, their p0 suffix.\n"
                        "e.g. m400,m450;w5p0;m750;w2p5,w10p0 expands to A_m400_w5p0,H_m750_w2p5;A_m400_w5p0,H_m750_w10p0;A_m450_w5p0,H_m750_w2p5;A_m450_w5p0,H_m750_w10p0",
                        default = "", required = False)
    parser.add_argument("--contour", help = "the json files containing the contour information, semicolon separated.\n"
                        "two separate syntaxes are possible:\n"
                        "'latest': t1/s1,s2;t2/s3... expands to <pnt>_<t1>/<pnt>_fc_scan_<s1>.json;<pnt>_<t1>/<pnt>_fc_scan_<s2>.json;<pnt>_<t2>/<pnt>_fc_scan_<s3>.json, "
                        " where the code will search for the latest indices corresponding to scenario s1 and so on. used if --point is non-empty, and looped over all pairs. \n"
                        "'direct': <json 1>;<json 2>;... used only when --point is empty",
                        default = "", required = True)

    parser.add_argument("--otag", help = "extra tag to append to plot names", default = "", required = False)
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False)
    parser.add_argument("--max-sigma", help = "max number of sigmas to be drawn on the contour", dest = "maxsigma", default = 2, type = int, required = False)
    parser.add_argument("--label", help = "labels to attach on plot for each json input, semicolon separated", default = "", required = False)

    parser.add_argument("--draw-scatter", help = "draw the scatter points used to build the contours",
                        dest = "scatter", action = "store_true", required = False)
    parser.add_argument("--draw-best-fit", help = "draw the best fit point.",
                        dest = "bestfit", action = "store_true", required = False)
    parser.add_argument("--skip-contour", help = "dont draw the contour",
                        dest = "drawcontour", action = "store_false", required = False)
    parser.add_argument("--proper-sigma", help = "use proper 1 or 2 sigma CLs instead of 68% and 95% in alphas",
                        dest = "propersig", action = "store_true", required = False)

    parser.add_argument("--formal", help = "plot is for formal use - put the CMS text etc",
                        dest = "formal", action = "store_true", required = False)
    parser.add_argument("--cms-append", help = "text to append to the CMS text, if --formal is used", dest = "cmsapp", default = "", required = False)
    parser.add_argument("--luminosity", help = "integrated luminosity applicable for the plot, written if --formal is used", default = "XXX", required = False)

    parser.add_argument("--transparent-background", help = "make the background transparent instead of white",
                        dest = "transparent", action = "store_true", required = False)
    parser.add_argument("--plot-format", help = "format to save the plots in", default = "png", dest = "fmt", required = False)

    args = parser.parse_args()
    if (args.otag != "" and not args.otag.startswith("_")):
        args.otag = "_" + args.otag

    if (args.fmt != "" and not args.fmt.startswith(".")):
        args.fmt = "." + args.fmt

    labels = args.label.split(';')

    if args.point != "":
        pairs = args.point.replace(" ", "").split(';')

        # handle the case of gridding pairs
        if len(pairs) == 4:
            pairgrid = [pp.split(",") for pp in pairs]
            if all([mm.startswith("m") for mm in pairgrid[0] + pairgrid[2]]) and all([ww.startswith("w") for ww in pairgrid[1] + pairgrid[3]]):
                alla = []
                for mm in pairgrid[0]:
                    for ww in pairgrid[1]:
                        alla.append("_".join(["A", mm, ww]))

                allh = []
                for mm in pairgrid[2]:
                    for ww in pairgrid[3]:
                        allh.append("_".join(["H", mm, ww]))

                pairs = []
                for aa in alla:
                    for hh in allh:
                        pairs.append([aa, hh])
    else:
        pairs = [os.path.basename(os.path.dirname(cc)).split(".")[0].split("__") for cc in contours]
        pairs = [[pp[0], "_".join(pp[1].split("_")[:3])] for pp in pairs]

        if not all([pp == pairs[0] for pp in pairs]):
            raise RuntimeError("provided contours are not all of the same pair of points!!")

        pairs = [pairs[0]]
        contours = args.contour.split(';')

    for pair in pairs:
        pstr = "__".join(pair)

        if args.point != "":
            contour = []
            tags = args.contour.replace(" ", "").split(';')
            tags = [tt if tt.startswith("_") else "_" + tt for tt in tags]

            for tag in tags:
                fcexps = tag.split('/')[1].split(',')
                for fcexp in fcexps:
                    ggg = glob.glob(pstr + tag + "/" + pstr + "_fc_scan_" + fcexp + "_*.json")
                    ggg.sort(key = os.path.getmtime)
                    contour.append(ggg[-1])
        else:
            contour = contours

        if len(contour) != len(labels):
            raise RuntimeError("there aren't as many input contours as there are labels. aborting")

        print("drawing contours for pair: ", pair)
        print("using the following contours: ", contour)
        draw_contour("{ooo}/{prs}_fc-contour{tag}{fmt}".format(ooo = args.odir, prs = pstr, tag = args.otag, fmt = args.fmt), pair, contour, labels,
                     args.maxsigma, args.propersig, args.drawcontour, args.bestfit, args.scatter, args.formal, args.cmsapp, args.luminosity, args.transparent)
