#!/usr/bin/env python3
# draw the dnll(gAH) for the 1 signal point scenario
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
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes

def read_nll(directories, onepoi, max_g):
    nlls = [OrderedDict() for dd in directories]
    for ii, directory, tag in enumerate(directories):
        with open("{dcd}/{pnt}_{tag}_nll_{mod}.json".format(dcd = directory, tag = tag, pnt = '_'.join(directory.split('_')[:3]), mod = "one-poi" if onepoi else "g-scan")) as ff:
            result = json.load(ff)

        obs_r = sys.float_info.max
        obs_g = sys.float_info.max

        for sce in ["exp-bkg", "exp-sig", "obs"]:
            nll = result[sce]
            nll0 = nll["obs"]["nll0"]
            dnlls = []

            for g, dnll in nll["dnll"].items():
                if abs(float(g)) > epsilon:
                    print("WARNING: in directory " + dd + ", first nll for scenario " + sce + " is at g = " + g + " instead of 0\n")
                nll1 = dnll + nll0
                break

            if sce == "obs":
                obs_r = nll["obs"]["r"]
                obs_g = nll["obs"]["g"]

            for g, dnll in nll["dnll"].items():
                gval = float(g)
                if gval <= max_g:
                    ispline = -1
                    for ii, kk in enumerate(kinks):
                        ispline = ii if kk[0] < gval < kk[1] else ispline

                    if ispline > -1:
                        spline = UnivariateSpline(np.array(kinks[ispline]),
                                                  np.array([dd + nll0 - nll1 for gg, dd in nll["dnll"].items() if kinks[ispline][0] <= float(gg) <= kinks[ispline][1]]))
                        dnlls.append( (gval, spline(gval)) )
                    else:
                        dnlls.append( (gval, dnll + nll0 - nll1) )

            nlls[ii][sce] = dnlls

        nlls[ii]["r"] = obs_r
        nlls[ii]["g"] = obs_g

    return nlls

def draw_nll(oname, directories, labels, onepoi, smooth, kinks, max_g, max_dnll, bestfit, transparent, plotformat):
    if len(directories) > 3:
        raise RuntimeError("current plotting code is not meant for more than 3 tags. aborting")

    # FIXME bestfit

    nlls = read_nll(directories, onepoi, kinks, max_g)
    point = get_point('_'.join(directories[0].split('_')[:3]))

    if not hasattr(draw_nll, "colors"):
        draw_nll.colors = OrderedDict([
            (1, ["black"]),
            (2, ["#cc0033", "#0033cc"]),
            (3, ["#cc0033", "#0033cc", "black"])
        ])
        draw_nll.lines = ["dotted", "dashed", "solid"]

    min_dnll = sys.float_info.max
    fig, ax = plt.subplots()
    handles = []
    ymax = max_dnll

    for ii, nll in enumerate(nlls):
        if ii == 0:
            for jj, sce in enumerate([("exp-bkg", "Expected"), ("exp-sig", "Signal injected"), ("obs", "Observed")]):
                handles.append((mln.Line2D([0], [0], color = "black", linestyle = draw_nll.lines[jj], linewidth = 1.5),
                                sce[1]))

        for jj, sce in enumerate(["exp-bkg", "exp-sig", "obs"]):
            gs = [nn[0] for nn in nll[sce] if 2. * nn[1] < ymax]
            dnlls = [2. * nn[1] for nn in nll[sce] if 2. * nn[1] < ymax]

            if min(dnlls) < min_dnll:
                min_dnll = min(dnlls)

            ax.plot(np.array(gs), np.array(dnlls), color = draw_nll.colors[len(directories)][ii], linestyle = draw_nll.lines[jj], linewidth = 1.5)

            if jj == 0:
                handles.append((mln.Line2D([0], [0], color = draw_nll.colors[len(directories)][ii], linestyle = draw_nll.lines[-1], linewidth = 1.5),
                                labels[ii]))

    ymin = math.ceil(math.sqrt(abs(min_dnll))) # ceil of sqrt ~ upper nsigma
    ymin *= ymin # go back to the dnll
    ymin = -round(ymin, -2)

    plt.ylim((ymin, 2. * ymax))
    ax.plot([0., max_g], [ymax, ymax], color = "black", linestyle = 'solid', linewidth = 2)
    plt.xlabel(axes["coupling"] % point[0], fontsize = 21, loc = "right")
    plt.ylabel(axes["dnll"] % point[0], fontsize = 21, loc = "top")
    ax.margins(x = 0, y = 0)

    lheight = ((2. * ymax) - ymax) / ((2. * ymax) - ymin)
    lmargin = 0.02
    lwidth = 1. - (2. * lmargin)
    dummy = ax.legend([], [],
	              loc = "upper right", ncol = 1, bbox_to_anchor = (lmargin, 1. - lheight, lwidth, lheight - 0.025),
                      mode = "expand", borderaxespad = 0., handletextpad = 0.5, fontsize = 21, frameon = False,
                      title = "Signal point", title_fontsize = 21)
    ax.add_artist(dummy)

    tags = ax.legend(first(handles[3:]), second(handles[3:]),
	             loc = "upper right", ncol = 1, bbox_to_anchor = (lmargin, 1. - lheight, 0.5 * lwidth, lheight - 0.025),
                     mode = "expand", borderaxespad = 0., handletextpad = 0.5, fontsize = 21, frameon = False,
                     title = " ", title_fontsize = 21)
    ax.add_artist(tags)

    scenarii = ax.legend(first(handles[:3]), second(handles[:3]),
	                 loc = "upper right", ncol = 1, bbox_to_anchor = (lmargin + (0.5 * lwidth), 1. - lheight, 0.5 * lwidth, lheight - 0.025),
                         mode = "expand", borderaxespad = 0., handletextpad = 0.5, fontsize = 21, frameon = False,
                         title = " ", title_fontsize = 21)
    ax.add_artist(scenarii)

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
    parser.add_argument("--point", help = "signal point to plot the pulls of", default = "", required = True, type = remove_spaces_quotes)
    parser.add_argument("--tag", help = "input tag-output-tag pairs to search. the pairs are semicolon separated, and tags colon-separated, "
                        "so e.g. when there are 2 tags: 't1:o1;t2:o2...", dest = "itag", default = "", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))
    parser.add_argument("--plot-tag", help = "extra tag to append to plot names", dest = "ptag", default = "", required = False, type = prepend_if_not_empty)
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False, type = remove_spaces_quotes)
    parser.add_argument("--label", help = "labels to attach on plot for each input tags, semicolon separated", default = "", required = False, type = lambda s: tokenize_to_list(s, ';' ))
    parser.add_argument("--one-poi", help = "plot pulls obtained with the g-only model", dest = "onepoi", action = "store_true", required = False)
    parser.add_argument("--smooth", help = "use spline to smooth kinks up. kinks are given in --kinks", action = "store_true", required = False)
    parser.add_argument("--kinks", help = "comma separated list of g values to be used by --smooth. every 2 values are treated as min and max of kink range",
                        default = "", required = False, type = lambda s: None if s == "" else tokenize_to_list( remove_spaces_quotes(s) ) )
    parser.add_argument("--max-g", help = "max value of g to incude in plot", dest = "max_g", type = float, default = 3., required = False)
    parser.add_argument("--max-dnll", help = "max value of dnll to incude in plot", dest = "max_dnll", type = float, default = 36., required = False)
    parser.add_argument("--best-fit", help = "write the observed best fit points on the plot",
                        dest = "bestfit", action = "store_true", required = False)

    parser.add_argument("--opaque-background", help = "make the background white instead of transparent",
                        dest = "transparent", action = "store_false", required = False)
    parser.add_argument("--plot-format", help = "format to save the plots in", default = ".png", dest = "fmt", required = False, type = lambda s: prepend_if_not_empty(s, '.'))

    args = parser.parse_args()
    if len(args.itag) != len(args.label):
        raise RuntimeError("length of tags isnt the same as labels. aborting")

    if args.kinks is not None:
        if len(args.kinks) % 2 == 1:
            raise RuntimeError("kinks given don't correspond to list of minmaxes. aborting!")

        args.kinks = [[kinks[ii], kinks[ii + 1]] for ii in range(len(args.kinks) - 1)]

    dirs = [[[args.point + '_' + tag.split(':')[0]], tag.split(':')[1] if len(tag.split(':')) > 1 else tag.split(':')[0]] for tag in args.itag]
    draw_nll("{ooo}/{pnt}_nll_{mod}{tag}{fmt}".format(ooo = args.odir, pnt = args.point, mod = "one-poi" if args.onepoi else "g-scan", tag = args.ptag, fmt = args.fmt),
             dirs, args.label, args.onepoi, args.smooth, args.kinks, args.max_g, args.max_dnll, args.bestfit, args.transparent, args.fmt)

    pass
