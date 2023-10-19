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

from utilspy import pmtofloat
from drawings import min_g, max_g, epsilon, axes, first, second, get_point
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes, remove_quotes

def nice_number(value, epsilon):
    if abs(value - int(value) - epsilon) < epsilon:
        value = int(value)
    return round(value, -math.ceil(math.log10(epsilon))) + 0.

def get_interval(parameter, best_fit, fits, delta = 1., epsilon = 1.e-2):
    islatexeqn = '$' in parameter
    parameter = parameter.replace('$', '') if islatexeqn else "\mathrm{0}".format('{' + parameter + '}')
    ndelta = (math.sqrt(delta) + 2.) ** 2
    pdelta = (math.sqrt(delta) - 1.) ** 2

    comparators = [lambda v0, v1: v0 < v1, lambda v0, v1: v0 > v1]
    uncertainties = []
    for icompare, comparator in enumerate(comparators):
        side = [fit for fit in fits if fit[1] > best_fit[1] and comparator(fit[0], best_fit[0])]
        side = [ss for ss in side if pdelta < ss[1] < ndelta]
        values = [[], []]
        for ss in side:
            if len(values[0]) == 0 or comparator(ss[1], values[1][-1]):
                values[0].append(ss[0])
                values[1].append(ss[1])

        if len(values[0]) < 4:
            uncertainties.append(None)
            continue
        if icompare == 0:
            values = [list(reversed(vv)) for vv in values]
        spline = UnivariateSpline(np.array(values[1]), np.array(values[0]), k = min(3, len(side) - 1))
        estimate = float(spline(delta))
        legit = comparators[1 if icompare == 0 else 0](estimate, values[0][-1])
        uncertainties.append(abs(estimate - best_fit[0]) if legit else None)

    if None not in uncertainties and abs(uncertainties[0] - uncertainties[1]) < epsilon:
        return f", ${parameter} = {nice_number(best_fit[0], epsilon)} \pm {round(uncertainties[0], -math.ceil(math.log10(epsilon)))}$"
    else:
        result = f", ${parameter} = {nice_number(best_fit[0], epsilon)}"
        for pos, sym, unc in zip(['_', '^'], ['-', '+'], uncertainties):
            result += "{0}".format(pos + '{' + sym + str(round(unc, -math.ceil(math.log10(epsilon)))) + '}') if unc is not None else "{0}".format(pos + '{' + sym + '\mathrm{inf}}')
        result += f"$"

        if any(sym in result for sym in ['-', '+']):
            return result
    return ""

def valid_1D_nll_fname(fname):
    fname = fname.split('/')[-1].split('_')
    nvalidto = 0
    for part in fname:
        if "to" in part:
            minmax = part.split('to')
            for mm in minmax:
                try:
                    pmtofloat(mm)
                    break
                except ValueError:
                    return False
            nvalidto += 1
    return nvalidto == 1

def read_nll(points, directories, name, kinks, skip, rangex, insidex, zeropoint):
    result = [[], []]
    best_fit, fits = result
    pstr = "__".join(points)

    for ii, (directory, scenario, tag) in enumerate(directories):
        fexp = f"{directory}/{pstr}_{tag}_nll_{scenario}_{name}_*.root"
        files = [ifile for ifile in glob.glob(fexp) if valid_1D_nll_fname(ifile)]
        best_fit.append(None)

        zero = None
        if zeropoint != "minimum":
            zeropoint = float(zeropoint)
            zero = (sys.float_info.max, sys.float_info.max)

            for ifile in files:
                dfile = TFile.Open(ifile)
                dtree = dfile.Get("limit")

                for i in dtree:
                    if dtree.quantileExpected >= 0.:
                        value = getattr(dtree, name)
                        delta = abs(value - zeropoint)
                        if delta < zero[-1]:
                            zero = (dtree.nll, delta)
                dfile.Close()

        originals = []
        for ifile in files:
            dfile = TFile.Open(ifile)
            dtree = dfile.Get("limit")

            for i in dtree:
                value = getattr(dtree, name)
                dnll = dtree.deltaNLL if zero is None else dtree.deltaNLL + dtree.nll0 - zero[0]
                dnll *= 2.

                if dtree.quantileExpected >= 0.:
                    originals.append((value, dnll))
                elif best_fit[ii] is None and dtree.quantileExpected == -1.:
                    best_fit[ii] = (value, dnll)
            dfile.Close()
        originals.append(best_fit[ii])
        originals = sorted(originals, key = lambda tup: tup[0])
        intx = []
        inty = []
        dataset = []

        if kinks is not None:
            for kink in kinks:
                if ii == kink[0]:
                    values = []
                    dnlls = []
                    for vv, dd in originals:
                        inkink = any([kink[1][i] < vv < kink[1][i + 1] for i in range(0, len(kink[1]), 2)])
                        if not inkink:
                            values.append(vv)
                            dnlls.append(dd)

                    if len(values) > 6 and len(dnlls) == len(values):
                        spline = UnivariateSpline(np.array(values), np.array(dnlls))
                        for vv, dd in originals:
                            inkink = any([kink[1][i] < vv < kink[1][i + 1] for i in range(0, len(kink[1]), 2)])
                            if inkink:
                                intx.append(vv)
                                inty.append(spline(vv))

        for value, dnll in originals:
            if skip and value in intx:
                continue
            if insidex and not (rangex[0] <= value <= rangex[1]):
                continue

            data = inty[intx.index(value)] if value in intx else dnll
            dataset.append((value, data))
        fits.append(dataset)
    return result

def draw_nll(oname, points, directories, labels, kinks, skip, namelabel,
             rangex, rangey, insidex, zeropoint, legendloc, legendtitle, transparent, plotformat):
    if not hasattr(draw_nll, "colors"):
        draw_nll.settings = OrderedDict([
            (1, [["black"], ["solid"]]),
            (2, [["#cc0033", "#0033cc"], ["solid", "dashed"]]),
            (3, [["black", "#cc0033", "#0033cc"], ["solid", "dashed", "dotted"]]),
            (4, [["black", "#cc0033", "#0033cc", "#33cc00"], ["solid", "dashed", "dashdot", "dotted"]]),
            (5, [["black", "#cc0033", "#0033cc", "#33cc00", "#9966cc"], ["solid", "dashed", "dashdot", (0, (3, 5, 1, 5)), (0, (1, 1))]]),
            (6, [["black", "#cc0033", "#0033cc", "#33cc00", "#9966cc", "#555555"], ["solid", "dashed", (0, (1, 1)), (0, (3, 5, 1, 5)), "dashdot", (0, (1, 5))]]),
        ])

    ndir = len(directories)
    if ndir <= len(draw_nll.settings):
        colors, styles = draw_nll.settings[ndir]
    elif ndir <= 2 * len(draw_nll.settings):
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
    nlls = read_nll(points, directories, name, kinks, skip, rangex, insidex, zeropoint)

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
    parser.add_argument("--label", help = "labels to attach on plot for each tag triplet, semicolon separated", default = "", required = False,
                        type = lambda s: tokenize_to_list(s, token = ';' ))

    parser.add_argument("--kinks", help = "semicolon-separated list of kink to be smoothed. syntax: "
                        "idx:min0,max0,min1,max1,... where idx is the zero-based curve index, min and max denote the xmin and xmax where the kinks happen.",
                        default = "", required = False, type = lambda s: None if s == "" else tokenize_to_list(remove_spaces_quotes(s), token = ';'))
    parser.add_argument("--skip", help = "skip points in kinks, rather than interpolating its value",
                        dest = "skip", action = "store_true", required = False)

    parser.add_argument("--x-name-label", help = "semicolon-separated parameter name and label on the x-axis. empty label means same as name for NPs.",
                        dest = "namelabel", type = lambda s: tokenize_to_list(remove_spaces_quotes(s), token = ';'), default = ["g1"], required = False)
    parser.add_argument("--x-range", help = "comma-separated min and max values in the x-axis", dest = "rangex",
                        type = lambda s: tokenize_to_list( remove_spaces_quotes(s), astype = float), default = [0., 2.], required = False)
    parser.add_argument("--y-range", help = "comma-separated min and max values in the y-axis", dest = "rangey",
                        type = lambda s: tokenize_to_list( remove_spaces_quotes(s), astype = float), default = [0., 36.], required = False)
    parser.add_argument("--inside-x-range", help = "consider only points within the specified x range",
                        dest = "insidex", action = "store_true", required = False)
    parser.add_argument("--zero-point", help = "point to mark as 0 on the 2dNLL axis. can be the 2dNLL minimum, or a given value value on the x axis.", dest = "zeropoint",
                        default = "minimum", required = False)
    parser.add_argument("--legend-position", help = "where to put the legend. passed as-is to mpl loc.", dest = "legendloc",
                        default = "upper right", required = False, type = remove_quotes)
    parser.add_argument("--legend-title", help = "where to put the legend. passed as-is to mpl.", dest = "legendtitle",
                        default = "", required = False, type = remove_quotes)

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

    if args.kinks is not None:
        args.kinks = [kk.split(':') for kk in args.kinks]
        args.kinks = [[int(kk[0]), [float(mm) for mm in kk[1].split(',')]] for kk in args.kinks]

        if any([len(kk[1]) % 2 == 1 for kk in args.kinks]):
            raise RuntimeError("one or more of the kinks given don't correspond to list of minmaxes. aborting!")

    if len(args.namelabel) == 1 and args.namelabel[0] in ["g1", "g2"]:
        args.namelabel[1] = axes["coupling"] % str_point(points[0]) if args.namelabel[0] == "g1" else axes["coupling"] % str_point(points[1])

    dirs = [tag.split(':') for tag in args.itag]
    dirs = [tag + tag[:1] if len(tag) == 2 else tag for tag in dirs]
    dirs = [[f"{pstr}_{tag[0]}"] + tag[1:] for tag in dirs]

    draw_nll(f"{args.odir}/{pstr}_nll_{args.namelabel[0]}{args.ptag}{args.fmt}",
             points, dirs, args.label, args.kinks, args.skip, args.namelabel, args.rangex, args.rangey, args.insidex, args.zeropoint,
             args.legendloc, args.legendtitle, args.transparent, args.fmt)
    pass
