#!/usr/bin/env python3
# draw the dnll(gAH) for the 1D
# requires matplotlib > 3.3 e.g. source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_100 x86_64-centos7-gcc10-opt

from argparse import ArgumentParser
import os
import sys
import numpy as np
from scipy.interpolate import UnivariateSpline
import math

from ROOT import TFile, TTree

import glob
from collections import OrderedDict
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpt
import matplotlib.lines as mln
import matplotlib.colors as mcl

from drawings import min_g, max_g, epsilon, axes, first, second, get_point
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes, remove_quotes

def read_nll(points, directories, name, rangex, rangey, kinks, zeropoint):
    result = []
    pstr = "__".join(points)

    if zeropoint == "minimum":
        zeros = None
    else:
        zeros = []
        zeropoint = float(zeropoint)

        for ii, (directory, scenario, tag) in enumerate(directories):
            files = glob.glob(f"{directory}/{pstr}_{tag}_nll_{scenario}_{name}*.root") # FIXME this will barf if there are more than 1 file with the same param
            zero = (sys.float_info.max, sys.float_info.max)

            dfile = TFile.Open(files[0])
            dtree = dfile.Get("limit")

            for i in dtree:
                if dtree.quantileExpected >= 0.:
                    value = getattr(dtree, name)
                    delta = abs(value - zeropoint)
                    if delta < zero[-1]:
                        zero = (dtree.nll, delta)
            zeros.append(zero)
            dfile.Close()

    for ii, (directory, scenario, tag) in enumerate(directories):
        originals = []
        files = glob.glob(f"{directory}/{pstr}_{tag}_nll_{scenario}_{name}*.root")
        dfile = TFile.Open(files[0])
        dtree = dfile.Get("limit")

        for i in dtree:
            if dtree.quantileExpected >= 0.:
                value = getattr(dtree, name)
                dnll = dtree.deltaNLL if zeros is None else dtree.deltaNLL + dtree.nll0 - zeros[ii][0]
                dnll *= 2.
                originals.append((value, dnll))
        dfile.Close()
        originals = sorted(originals, key = lambda tup: tup[0])
        intx = []
        inty = []
        dataset = []

        if kinks is not None:
            for kink in kinks:
                values = [vv for vv, dd in originals if not (kink[0] <= vv <= kink[1])]
                dnlls = [dd for vv, dd in originals if not (kink[0] <= vv <= kink[1])]

                if len(values) > 6 and len(dnlls) == len(values):
                    spline = UnivariateSpline(np.array(values), np.array(dnlls))
                    intx += [vv for vv, dd in originals if kink[0] <= vv <= kink[1]]
                    inty += [spline(vv) for vv, dd in originals if kink[0] <= vv <= kink[1]]

        for value, dnll in originals:
            if rangex[0] <= value <= rangex[1]:
                data = inty[intx.index(value)] if value in intx else dnll
                if rangey[0] <= data <= rangey[1]:
                    dataset.append((value, data))
        result.append(dataset)
    return result

def draw_nll(oname, points, directories, labels, kinks, namelabel, rangex, rangey, zeropoint, legendloc, transparent, plotformat):
    if not hasattr(draw_nll, "colors"):
        draw_nll.settings = OrderedDict([
            (1, [["black"], ["solid"]]),
            (2, [["#cc0033", "#0033cc"], ["solid", "dashed"]]),
            (3, [["black", "#cc0033", "#0033cc"], ["solid", "dashed", "dotted"]]),
            (4, [["black", "#cc0033", "#0033cc", "#33cc00"], ["solid", "dashed", "dashdot", "dotted"]]),
        ])

    ndir = len(directories)
    if ndir < 5:
        colors, styles = draw_nll.settings[ndir]
    elif ndir < 9:
        if ndir % 2 == 0:
            colors = []
            styles = []
            for ii in range(ndir / 2):
                colors.append(draw_nll.settings[ndir / 2][0][ii])
                colors.append(draw_nll.settings[ndir / 2][0][ii])
                styles.append("solid")
                styles.append("dashed")
        else:
            raise RuntimeError("cant assign by color/style for uneven counts")
    else:
        raise RuntimeError("too many scenarios")

    fig, ax = plt.subplots()
    handles = []
    name, xlabel = namelabel if len(namelabel) > 1 else namelabel + namelabel
    nlls = read_nll(points, directories, name, rangex, rangey, kinks, zeropoint)

    for ii, nll in enumerate(nlls):
        color = colors[ii]
        style = styles[ii]
        label = labels[ii]
        handles.append((mln.Line2D([0], [0], color = color, linestyle = style, linewidth = 1.5), label))
        values = np.array([nn[0] for nn in nll])
        dnlls = np.array([nn[1] for nn in nll])
        ax.plot(values, dnlls, color = color, linestyle = style, linewidth = 1.5)

    plt.ylim(rangey)
    ax.plot(rangex, [rangey[1], rangey[1]], color = "black", linestyle = 'solid', linewidth = 2)
    plt.xlabel(xlabel, fontsize = 21, loc = "right")
    #plt.ylabel(axes["dnll"] % 'A/H', fontsize = 21, loc = "top")
    plt.ylabel("2dNLL", fontsize = 21, loc = "top")
    ax.margins(x = 0, y = 0)

    legend = ax.legend(first(handles), second(handles),
	               loc = legendloc, ncol = 1 if ndir < 5 else 2, borderaxespad = 1., fontsize = 21, frameon = False)
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
    parser.add_argument("--label", help = "labels to attach on plot for each tag triplet, semicolon separated", default = "", required = False,
                        type = lambda s: tokenize_to_list(s, token = ';' ))
    parser.add_argument("--smooth", help = "use spline to smooth kinks up. kinks are given in --kinks", action = "store_true", required = False)
    parser.add_argument("--kinks", help = "comma separated list of values to be used by --smooth. every 2 values are treated as min and max of kink range",
                        default = "", required = False, type = lambda s: None if s == "" else tokenize_to_list( remove_spaces_quotes(s), astype = float ) )
    parser.add_argument("--x-name-label", help = "semicolon-separated parameter name and label on the x-axis. empty label means same as name for NPs.",
                        dest = "namelabel", type = lambda s: tokenize_to_list( remove_spaces_quotes(s), token = ';'), default = ["g1"], required = False)
    parser.add_argument("--x-range", help = "comma-separated min and max values in the x-axis", dest = "rangex",
                        type = lambda s: tokenize_to_list( remove_spaces_quotes(s), astype = float), default = [0., 2.], required = False)
    parser.add_argument("--y-range", help = "comma-separated min and max values in the y-axis", dest = "rangey",
                        type = lambda s: tokenize_to_list( remove_spaces_quotes(s), astype = float), default = [0., 36.], required = False)
    parser.add_argument("--zero-point", help = "point to mark as 0 on the 2dNLL axis. can be the 2dNLL minimum, or a given value value on the x axis.", dest = "zeropoint",
                        default = "minimum", required = False)
    parser.add_argument("--legend-position", help = "where to put the legend. passed as-is to mpl loc.", dest = "legendloc",
                        default = "upper right", required = False, type = remove_quotes)

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

    if args.smooth and args.kinks is not None:
        if len(args.kinks) % 2 == 1:
            raise RuntimeError("kinks given don't correspond to list of minmaxes. aborting!")

        args.kinks = [[args.kinks[ii], args.kinks[ii + 1]] for ii in range(0, len(args.kinks), 2)]
    else:
        args.kinks = None

    if len(args.namelabel) == 1 and args.namelabel[0] in ["g1", "g2"]:
        args.namelabel[1] = axes["coupling"] % str_point(points[0]) if args.namelabel[0] == "g1" else axes["coupling"] % str_point(points[1])

    dirs = [tag.split(':') for tag in args.itag]
    dirs = [tag + tag[:1] if len(tag) == 2 else tag for tag in dirs]
    dirs = [[f"{pstr}_{tag[0]}"] + tag[1:] for tag in dirs]

    draw_nll(f"{args.odir}/{pstr}_nll_{args.namelabel[0]}{args.ptag}{args.fmt}",
             points, dirs, args.label, args.kinks, args.namelabel, args.rangex, args.rangey, args.zeropoint, args.legendloc, args.transparent, args.fmt)
    pass
