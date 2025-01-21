#!/usr/bin/env python3
# draw the nuisance pull/impact plot
# requires matplotlib > 3.3 e.g. source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_100 x86_64-centos7-gcc10-opt

from argparse import ArgumentParser
import os
import sys
import numpy as np
import math

import glob
from collections import OrderedDict
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mln
from matplotlib.legend_handler import HandlerErrorbar

from drawings import min_g, max_g, epsilon, axes, first, second, get_point
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes

nuisance_per_page = 33

def read_pull(directories, isimpact, onepoi, poiname, gvalue, rvalue, fixpoi):
    pulls = [OrderedDict() for directory in directories]
    for ii, [directory, tag] in enumerate(directories):
        if ii == 0:
            poiname = "EWK_const"
        else:
            poiname = "EWK_const"

        impacts = glob.glob("{dcd}/{pnt}_{tag}_impacts_{mod}{gvl}{rvl}{fix}*.json".format(
            dcd = directory,
            tag = tag,
            pnt = '_'.join(directory.split('_')[:3]),
            mod = poiname if poiname != "g" else "one-poi" if onepoi else "g-scan",
            gvl = "_g_" + str(gvalue).replace(".", "p") if gvalue >= 0. else "",
            rvl = "_r_" + str(rvalue).replace(".", "p") if rvalue >= 0. and not onepoi else "",
            fix = "_fixed" if fixpoi and (gvalue >= 0. or rvalue >= 0.) else "",
        ))

        for imp in impacts:
            with open(imp) as ff:
                result = json.load(ff)

            nuisances = result["params"]
            for nn in nuisances:
                if nn["name"] == "EWK_const" or nn["name"] == "CMS_EtaT_norm_13TeV":
                    continue
                elif nn["name"] == "EWK_yukawa":
                    if isimpact:
                        pulls[ii][nn["name"]] = nn["r"]
                    else:
                        dummy = nn["fit"]
                        if dummy[1] < nn["prefit"][1]:
                            dummy = [(dd - nn["prefit"][1]) / (nn["prefit"][1] - nn["prefit"][0]) for dd in dummy]
                        else:
                            dummy = [(dd - nn["prefit"][1]) / (nn["prefit"][2] - nn["prefit"][1]) for dd in dummy]
                        pulls[ii][nn["name"]] = dummy
                else:
                    pulls[ii][nn["name"]] = poiname if isimpact else nn["fit"]

    return pulls

def plot_pull(oname, labels, isimpact, pulls, nuisances, extra, point, reverse, transparent, plotformat):
    fig, ax = plt.subplots()
    xval = [np.zeros(nuisance_per_page) for pp in pulls]
    yval = [np.zeros(nuisance_per_page) for pp in pulls]
    err = [np.zeros((2, nuisance_per_page)) for pp in pulls]
    imu = [np.zeros((2, nuisance_per_page)) for pp in pulls]
    imd = [np.zeros((2, nuisance_per_page)) for pp in pulls]
    imc = [np.zeros((2, nuisance_per_page)) for pp in pulls]

    offset = [0.3, 0., -0.3] if len(pulls) == 3 else [0.2, -0.2] if len(pulls) == 2 else [0.]
    pcolors = ['0', "#cc0033", "#0033cc"] if len(pulls) == 3 else ["#cc0033", "#0033cc"] if len(pulls) == 2 else ["black"]
    icolors = ['0', "#cc0033", "#0033cc"]
    markers = ["o", "s", "^"] if len(pulls) == 3 else ["o", "s"] if len(pulls) == 2 else ["o"]
    counter = math.ceil(len(nuisances) / nuisance_per_page) - 1 # not floor, because that doesn't give 0, 1, 2, ... for integer multiples

    if reverse:
        nuisances = list(reversed(nuisances))

    if isimpact:
        impacts = [abs(pulls[0][nn][2] - pulls[0][nn][0]) if nn in pulls[0] else 0. for nn in nuisances]
        nuisances = [nn for ii, nn in sorted(zip(impacts, nuisances))]

    for ii, nn in enumerate(nuisances):
        for jj in range(len(pulls)):
            if isimpact:
                xval[jj][ii % nuisance_per_page] = 0.

                if nn in pulls[jj]:
                    if pulls[jj][nn][0] - pulls[jj][nn][1] > 0.:
                        imu[jj][0, ii % nuisance_per_page] = 0.
                        imu[jj][1, ii % nuisance_per_page] = pulls[jj][nn][0] - pulls[jj][nn][1]
                    else:
                        imu[jj][0, ii % nuisance_per_page] = pulls[jj][nn][1] - pulls[jj][nn][0]
                        imu[jj][1, ii % nuisance_per_page] = 0.

                    if pulls[jj][nn][2] - pulls[jj][nn][1] > 0.:
                        imd[jj][0, ii % nuisance_per_page] = 0.
                        imd[jj][1, ii % nuisance_per_page] = pulls[jj][nn][2] - pulls[jj][nn][1]
                    else:
                        imd[jj][0, ii % nuisance_per_page] = pulls[jj][nn][1] - pulls[jj][nn][2]
                        imd[jj][1, ii % nuisance_per_page] = 0.
                else:
                    imu[jj][0, ii % nuisance_per_page] = 0.
                    imd[jj][0, ii % nuisance_per_page] = 0.
                    imu[jj][1, ii % nuisance_per_page] = 0.
                    imd[jj][1, ii % nuisance_per_page] = 0.
            else:
                xval[jj][ii % nuisance_per_page] = pulls[jj][nn][1] if nn in pulls[jj] else 0.

                err[jj][0, ii % nuisance_per_page] = pulls[jj][nn][1] - pulls[jj][nn][0] if nn in pulls[jj] else 0.
                err[jj][1, ii % nuisance_per_page] = pulls[jj][nn][2] - pulls[jj][nn][1] if nn in pulls[jj] else 0.

            yval[jj][ii % nuisance_per_page] = (ii % nuisance_per_page) + offset[jj]

        if ii % nuisance_per_page == nuisance_per_page - 1 or ii == len(nuisances) - 1:
            ymax = (ii % nuisance_per_page) + 1 if ii == len(nuisances) - 1 else nuisance_per_page
            rmax = max([np.amax(imp[0:, 0:]) for imp in imu] +
                        [np.amax(imp[0:, 0:]) for imp in imd]) if isimpact else 3.
            rmax = math.ceil(rmax * 10.**int( math.ceil(abs(math.log10(rmax))) )) / 10.**int( math.ceil(abs(math.log10(rmax))) )
            lmax = -rmax if isimpact else -3.

            plots = []
            handles = []
            if isimpact:
                for jj in range(len(pulls)):
                    plots.append(ax.errorbar(xval[jj][:ymax], yval[jj][:ymax], xerr = imu[jj][0:, :ymax], ls = "none", elinewidth = 1.5,
                                             marker = markers[jj], ms = 7, capsize = 5, color = icolors[1], label = "Up"))
                    plots.append(ax.errorbar(xval[jj][:ymax], yval[jj][:ymax], xerr = imd[jj][0:, :ymax], ls = "none", elinewidth = 1.5,
                                             marker = markers[jj], ms = 7, capsize = 5, color = icolors[2], label = "Down"))
                    plots.append(ax.errorbar(xval[jj][:ymax], yval[jj][:ymax], xerr = imc[jj][0:, :ymax], ls = "none", elinewidth = 1.5,
                                             marker = markers[jj], ms = 7, capsize = 5, color = icolors[0], label = labels[jj]))

                    if jj == 0:
                        handles.append((mln.Line2D([0], [0], color = icolors[1], linestyle = "solid", linewidth = 1.5), "Up"))
                        handles.append((mln.Line2D([0], [0], color = icolors[2], linestyle = "solid", linewidth = 1.5), "Down"))

                    handles.append((mln.Line2D([0], [0], color = icolors[0], linestyle = "solid", linewidth = 1.5,
                                               marker = markers[jj], ms = 7), labels[jj]))

            else:
                ax.fill_between(np.array([-1, 1]), np.array([-0.5, -0.5]), np.array([ymax - 0.5, ymax - 0.5]), color = "silver", linewidth = 0)

                for jj in range(len(pulls)):
                    plots.append(ax.errorbar(xval[jj][:ymax], yval[jj][:ymax], xerr = err[jj][0:, :ymax], ls = "none", elinewidth = 1.5,
                                             marker = markers[jj], ms = 5, capsize = 5, color = pcolors[jj], label = labels[jj]))

            ax.set_yticks([kk for kk in range(ymax)])
            ax.set_yticklabels([nn + r"$\,$" for nn in nuisances[ii - ymax + 1 : ii + 1]])
            if isimpact:
                plt.xlabel(point[0] + '(' + str(int(point[1])) + ", " + str(point[2]) + "%) nuisance impacts", fontsize = 21, labelpad = 10)
            else:
                #plt.xlabel(point[0] + '(' + str(int(point[1])) + ", " + str(point[2]) + "%) nuisance pulls", fontsize = 21, labelpad = 10)
                plt.xlabel("nuisance pulls", fontsize = 21, labelpad = 10)
            ax.margins(x = 0, y = 0)
            plt.xlim((lmax, rmax))
            plt.ylim((-0.5, ymax - 0.5))

            if isimpact:
                legend = ax.legend(first(handles), second(handles),
	                           loc = "lower left", ncol = len(pulls) + 2, bbox_to_anchor = (0., 1.005, 1., 0.01),
                                   mode = "expand", borderaxespad = 0., fontsize = 15, frameon = False)
            else:
                if len(pulls) > 1:
                    legend = ax.legend(loc = "lower left", ncol = len(pulls), bbox_to_anchor = (0.05, 1.005, 0.9, 0.01),
                                       mode = "expand", borderaxespad = 0., handletextpad = 1.5, fontsize = 15, frameon = False,
                                       handler_map = {plots[0]: HandlerErrorbar(xerr_size = 1.5), plots[1]: HandlerErrorbar(xerr_size = 1.5)})

            ax.minorticks_on()
            ax.tick_params(axis = "both", which = "both", direction = "in", bottom = True, top = True, left = True, right = True)
            ax.tick_params(axis = "x", which = "major", width = 1, length = 8, labelsize = 18)
            ax.tick_params(axis = "y", which = "major", width = 1, length = 8, labelsize = 14)
            ax.tick_params(axis = "x", which = "minor", width = 1, length = 3)
            ax.tick_params(axis = "y", which = "minor", width = 0, length = 0)

            fig.set_size_inches(9., 16.)
            fig.tight_layout()
            for fmt in plotformat:
                if len(pulls) == 2:
                    fig.savefig(oname + extra + str(counter) + fmt, bbox_extra_artists = (legend,), transparent = transparent)
                else:
                    fig.savefig(oname + extra + str(counter) + fmt, transparent = transparent)
            fig.clf()

            fig, ax = plt.subplots()
            counter = counter - 1

def draw_pull(oname, directories, labels, isimpact, onepoi, poiname, gvalue, rvalue, fixpoi, mcstat, transparent, plotformat):
    pulls = read_pull(directories, isimpact, onepoi, poiname, gvalue, rvalue, fixpoi)
    point = get_point('_'.join(directories[0][0].split('_')[:3]))

    expth = []
    for pull in pulls:
        for nn in pull.keys():
            if "prop_bin" not in nn:
                expth.append(nn)
    plot_pull(oname, labels, isimpact, pulls, sorted(list(set(expth))), "_expth_", point, True, transparent, plotformat)

    if mcstat:
        mcstat = []
        for pull in pulls:
            for nn in pull.keys():
                if "prop_bin" in nn:
                    mcstat.append(nn)
        plot_pull(oname, labels, isimpact, pulls, sorted(list(set(mcstat))), "_mcstat_", point, True, transparent, plotformat)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "signal point to plot the pulls of", default = "", required = True, type = remove_spaces_quotes)
    parser.add_argument("--tag", help = "input tag-output-tag pairs to search. the pairs are semicolon separated, and tags colon-separated, "
                        "so e.g. when there are 2 tags: 't1:o1;t2:o2...", dest = "itag", default = "", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))
    parser.add_argument("--plot-tag", help = "extra tag to append to plot names", dest = "ptag", default = "", required = False, type = prepend_if_not_empty)
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False, type = remove_spaces_quotes)
    parser.add_argument("--label", help = "labels to attach on plot for each input tags, semicolon separated", default = "", required = False, type = lambda s: tokenize_to_list(s, ';' ))
    parser.add_argument("--draw", help = "what to draw, pulls or impacts", default = "pull", required = False, type = remove_spaces_quotes)

    parser.add_argument("--one-poi", help = "plot pulls obtained with the g-only model", dest = "onepoi", action = "store_true", required = False)

    parser.add_argument("--poi-name", help = "name of poi",
                        dest = "poiname", default = "g", required = False)

    parser.add_argument("--g-value",
                        help = "g to use when evaluating impacts/fit diagnostics/nll. "
                        "does NOT freeze the value, unless --fix-poi is also used. "
                        "note: semantically sets value of 'r' with --one-poi, as despite the name it plays the role of g.",
                        dest = "setg", default = "-1.", required = False, type = lambda s: float(remove_spaces_quotes(s)))
    parser.add_argument("--r-value",
                        help = "r to use when evaluating impacts/fit diagnostics/nll, if --one-poi is not used."
                        "does NOT freeze the value, unless --fix-poi is also used.",
                        dest = "setr", default = "-1.", required = False, type = lambda s: float(remove_spaces_quotes(s)))
    parser.add_argument("--fix-poi", help = "fix pois in the fit, through --g-value and/or --r-value",
                        dest = "fixpoi", action = "store_true", required = False)

    parser.add_argument("--with-mc-stats", help = "plot also the bb-lite nuisances",
                        dest = "mcstat", action = "store_true", required = False)

    parser.add_argument("--opaque-background", help = "make the background white instead of transparent",
                        dest = "transparent", action = "store_false", required = False)
    parser.add_argument("--plot-formats", help = "comma-separated list of formats to save the plots in", default = [".png"], dest = "fmt", required = False,
                        type = lambda s: [prepend_if_not_empty(fmt, '.') for fmt in tokenize_to_list(remove_spaces_quotes(s))])

    args = parser.parse_args()

    isimpact = "impact" in args.draw

    if len(args.itag) != len(args.label):
        raise RuntimeError("length of tags isnt the same as labels. aborting")

    dirs = [tag.split(':') for tag in args.itag]
    dirs = [tag + tag[:1] if len(tag) < 2 else tag for tag in dirs]
    dirs = [[f"{args.point}_{tag[0]}"] + tag[1:] for tag in dirs]

    draw_pull(args.odir + "/" + args.point + "_{drw}".format(drw = "impact" if isimpact else "pull") + args.ptag,
              dirs, args.label, isimpact, args.onepoi, args.poiname, args.setg, args.setr, args.fixpoi, args.mcstat, args.transparent, args.fmt)
    pass
