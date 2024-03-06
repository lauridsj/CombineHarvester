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

from drawings import min_g, max_g, epsilon, axes, first, second, get_point, str_point, default_etat_blurb
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes

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

def draw_contour(oname, pair, cfiles, labels, maxsigma, propersig, drawcontour, bestfit, scatter, formal, cmsapp, luminosity, a343bkg, transparent):
    contours = read_contour(cfiles)
    ncontour = len(contours)
    alphas = [0.6827, 0.9545, 0.9973, 0.999937, 0.9999997] if propersig else [0.68, 0.95, 0.9973, 0.999937, 0.9999997]
    alphas = [1. - pval for pval in alphas]

    if not hasattr(draw_contour, "colors"):
        draw_contour.colors = OrderedDict([
            (1    , ["0"]),
            (2    , ["#cc0033", "0"]),
            (3    , ["#0033cc", "#cc0033", "0"]),
            (4    , ["#33cc00", "#0033cc", "#cc0033", "0"]),
        ])
        draw_contour.lines = ['solid', 'dashed', 'dashdot', 'dashdotdotted', 'dotted']


    fig, ax = plt.subplots()
    handles = []
    sigmas = []

    for ic, contour in enumerate(contours):
        colortouse = draw_contour.colors[len(contours)][ic]

        if bestfit:
            ax.plot(np.array([contour["best_fit"][0]]), np.array([contour["best_fit"][1]]),
                    marker = 'X', markersize = 10.0, color = colortouse)
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
                        marker = '.', ls = '', lw = 0., color = colortouse, alpha = 0.5)

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
                drawcontour = False

            if drawcontour:
                ax.tricontour(np.array(contour["g1"]), np.array(contour["g2"]), contour["eff"],
                              levels = np.array([alpha, 2.]), colors = colortouse,
                              linestyles = draw_contour.lines[isig], linewidths = 2, alpha = 1. - (0.05 * isig))

            if len(labels) > 1 and isig == 0:
                handles.append((mln.Line2D([0], [0], color = colortouse, linestyle = 'solid', linewidth = 2), labels[ic]))

    plt.xlabel(axes["coupling"] % str_point(pair[0]), fontsize = 23, loc = "right")
    plt.ylabel(axes["coupling"] % str_point(pair[1]), fontsize = 23, loc = "top")
    ax.margins(x = 0, y = 0)

    if not scatter:
        if len(handles) > 0 and len(sigmas) > 0:
            legend1 = ax.legend(first(sigmas), second(sigmas), loc = 'best', bbox_to_anchor = (0.75, 0.625, 0.225, 0.3), fontsize = 21, handlelength = 2, borderaxespad = 1., frameon = False)
            ax.add_artist(legend1)

            legend2 = ax.legend(first(handles), second(handles), loc = 'best', bbox_to_anchor = (0.75, 0., 0.25, 0.2), fontsize = 21, handlelength = 2., borderaxespad = 1., frameon = False)
            ax.add_artist(legend2)

        elif len(handles) > 0:
            ax.legend(first(handles), second(handles), loc = 'lower right', fontsize = 21, handlelength = 2., borderaxespad = 1., frameon = False)
        elif len(sigmas) > 0:
            ax.legend(first(sigmas), second(sigmas), loc = 'lower right', fontsize = 21, handlelength = 2., borderaxespad = 1., frameon = False)

    if formal:
        ctxt = "{cms}".format(cms = r"$\textbf{CMS}$")
        ax.text(0.03 * max_g, 0.96 * max_g, ctxt, fontsize = 31, ha = 'left', va = 'top', usetex = True)

        if cmsapp != "":
            atxt = "{app}".format(app = r" $\textit{" + cmsapp + r"}$")
            ax.text(0.03 * max_g, 0.90 * max_g, atxt, fontsize = 26, ha = 'left', va = 'top', usetex = True)

        ltxt = "{lum}{ifb}".format(lum = luminosity, ifb = r" fb$^{\mathrm{\mathsf{-1}}}$ (13 TeV)")
        ax.text(0.985 * max_g, 0.97 * max_g, ltxt, fontsize = 26, ha = 'right', va = 'top', usetex = True)

        if a343bkg[0]:
            btxt = [
                r"$\mathbf{Including}$ $\mathbf{\eta_{t}}$ $\mathbf{approximation}$",
                r"PRD 104, 034023 ($\mathbf{2021}$)"
            ]

            # disabled because adding the profiled number depends on signal point
            #if len(a343bkg) > 3:
            #    btxt += [r"Best fit $\sigma_{\eta_{\mathrm{t}}}$: $" + "{val}".format(val = a343bkg[1]) + r"_{-" + "{ulo}".format(ulo = a343bkg[2]) + r"}^{+" + "{uhi}".format(uhi = a343bkg[3]) + r"}$ pb ($\mathrm{g}_{\mathrm{\mathsf{A/H}}} = 0$)"]
            #else:
            #    btxt += [r"Best fit $\sigma^{\eta_{\mathrm{t}}}$: $" + "{val}".format(val = a343bkg[1]) + r" \pm " + "{unc}".format(unc = a343bkg[2]) + r"$ pb ($\mathrm{g}_{\mathrm{\mathsf{A/H}}} = 0$)"]
        else:
            btxt = [
                r"$\mathbf{Excluding}$ $\mathbf{\eta_{t}}$ $\mathbf{approximation}$",
                r"PRD 104, 034023 ($\mathbf{2021}$)"
            ]
        bbln = [matplotlib.patches.Rectangle((0, 0), 1, 1, fc = "white", ec = "white", lw = 0, alpha = 0)] * len(btxt)
        ax.legend(bbln, btxt, loc = 'best', bbox_to_anchor = (0.85, 0.5, 0.15, 0.15), fontsize = 14, frameon = False, handlelength = 0, handletextpad = 0, borderaxespad = 1.)

    ax.minorticks_on()
    ax.tick_params(axis = "both", which = "both", direction = "in", bottom = True, top = True, left = True, right = True)
    ax.tick_params(axis = "both", which = "major", width = 1, length = 8, labelsize = 18, pad = 10)
    ax.tick_params(axis = "both", which = "minor", width = 1, length = 3)

    fig.set_size_inches(8., 8.)
    fig.set_dpi(450)
    fig.tight_layout()

    fig.savefig(oname, transparent = transparent)
    fig.clf()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point",
                        help = "desired pairs of signal points to run on, comma (between points) and semicolon (between pairs) separated\n"
                        "another syntax is: m1,m2,...,mN ; w1,w2,...,wN ; m1,m2,...,mN ; w1,w2,...,wN where:\n"
                        "the first mass and width strings refer to the A grid, and the second to the H grid.\n"
                        "both mass and width strings must include their m and w prefix, and for width, their p0 suffix.\n"
                        "e.g. m400,m450 ;w5p0 ; m750 ; w2p5,w10p0 expands to \n"
                        "A_m400_w5p0,H_m750_w2p5 ; A_m400_w5p0,H_m750_w10p0 ; A_m450_w5p0,H_m750_w2p5 ; A_m450_w5p0,H_m750_w10p0",
                        default = "", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))
    parser.add_argument("--contour", help = "the json files containing the contour information, semicolon separated.\n"
                        "two separate syntaxes are possible:\n"
                        "'tag': t1/s1:o1:i1,s2:o2:i2 ; t2/s3:o3:i3 ; ... expands to (let p being the considered point, and <fc> = fc-scan):"
                        "'<p>_<t1>/<p>_<o1>_<fc>_<s1>_<i1>.json ; <p>_<t1>/<p>_<o2>_<fc>_<s2>_<i2>.json ; <p>_<t2>/<p>_<o3>_<fc>_<s3>_<i3>.json', "
                        "where the code will search for output tag o1 and index i1 corresponding to scenario s1 and so on."
                        "if :o1 etc is omitted, they default to o1 = t1, ..., oN = tN.\n"
                        "if :i1 etc is omitted, the latest index will be picked.\n"
                        "used if --point is non-empty, and looped over all pairs. \n"
                        "'direct': <json 1> ; <json 2> ; ... used only when --point is empty",
                        default = "", required = True, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))

    parser.add_argument("--plot-tag", help = "extra tag to append to plot names", default = "", dest = "ptag", required = False, type = prepend_if_not_empty)
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False, type = remove_spaces_quotes)
    parser.add_argument("--max-sigma", help = "max number of sigmas to be drawn on the contour", dest = "maxsigma", default = "2", required = False, type = lambda s: int(remove_spaces_quotes(s)))
    parser.add_argument("--label", help = "labels to attach on plot for each json input, semicolon separated", default = "", required = False, type = lambda s: tokenize_to_list(s, ';' ))

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
    parser.add_argument("--luminosity", help = "integrated luminosity applicable for the plot, written if --formal is used", default = "138", required = False)
    parser.add_argument("--A343-background",
                        help = "a comma-separated list of 4 values for etat background text, written if --formal is used"
                        "syntax: (bool, 1 or 0, whether it is included as bkg, best fit xsec, xsec uncertainty (lo/hi)",
                        dest = "a343bkg", default = default_etat_blurb(), required = False,
                        type = default_etat_blurb)

    parser.add_argument("--opaque-background", help = "make the background white instead of transparent",
                        dest = "transparent", action = "store_false", required = False)
    parser.add_argument("--plot-format", help = "format to save the plots in", default = ".png", dest = "fmt", required = False, type = lambda s: prepend_if_not_empty(s, '.'))
    args = parser.parse_args()

    if args.point != "":
        pairs = args.point

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
                        pairs.append(aa + "," + hh)
    else:
        contours = args.contour
        pairs = [os.path.basename(os.path.dirname(cc)).split(".")[0].split("__") for cc in contours]
        pairs = [[pp[0], "_".join(pp[1].split("_")[:3])] for pp in pairs]

        if not all([pp == pairs[0] for pp in pairs]):
            raise RuntimeError("provided contours are not all of the same pair of points!!")

        pairs = [','.join(pairs[0])]

    for pair in pairs:
        pair = pair.split(',')
        pstr = "__".join(pair)

        if args.point != "":
            contour = []
            tags = [prepend_if_not_empty(tt) for tt in args.contour]

            for tag in tags:
                fcexps = tag.split('/')[1].split(',')
                for fcexp in fcexps:
                    exp = fcexp
                    otg = tag.split('/')[0]
                    idx = "_*"
                    if ':' in fcexp:
                        exp = fcexp.split(':')[0]
                        otg = '_' + fcexp.split(':')[1]
                        idx = '_' + fcexp.split(':')[2] if len(fcexp.split(':')) > 2 else idx

                    ggg = glob.glob(pstr + tag.split('/')[0] + "/" + pstr + otg + "_fc-scan_" + exp + idx + ".json")
                    ggg.sort(key = lambda name: int(name.split('_')[-1].split('.')[0]))
                    contour.append(ggg[-1])
        else:
            contour = contours

        if len(contour) != len(args.label):
            raise RuntimeError("there aren't as many input contours as there are labels. aborting")

        print("drawing contours for pair: ", pair)
        print("using the following contours: ", contour)
        draw_contour("{ooo}/{prs}_fc-contour{tag}{fmt}".format(ooo = args.odir, prs = pstr, tag = args.ptag, fmt = args.fmt), pair, contour, args.label,
                     args.maxsigma, args.propersig, args.drawcontour, args.bestfit, args.scatter, args.formal, args.cmsapp, args.luminosity, args.a343bkg, args.transparent)
        print()
