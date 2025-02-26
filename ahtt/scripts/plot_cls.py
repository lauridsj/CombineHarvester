#!/usr/bin/env python3
# draw the cls(g) for some signal points
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
import matplotlib.pyplot as plt

from drawings import min_g, max_g, epsilon, axes, first, second, get_point, str_point
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes

def read_cls(directory, otag):
    chunks = glob.glob("{dcd}/{pnt}_{tag}_limits_g-scan_n*_i*.json".format(
        dcd = directory,
        pnt = '_'.join(directory.split('_')[:3]),
        tag = otag))
    chunks.sort(key = lambda name: int(name.split('_')[-1].split('.')[0][1:]))

    limit = OrderedDict([
        ("exp-2", []),
        ("exp-1", []),
        ("exp0", []),
        ("exp+1", []),
        ("exp+2", []),
        ("obs", [])
    ])

    for nn in chunks:
        with open(nn) as ff:
            result = json.load(ff)

        for g, lmt in result.items():
            # FIXME in some cases combine gives strange cls values
            # not understood why, skip
            if any([round(cls, 3) > 1. or round(cls, 3) < 0. for cls in lmt.values()]):
                continue
            for quantile, cls in lmt.items():
                limit[quantile].append((round(float(g), 4), round(cls, 5)))
    return limit

def draw_cls(odir, directories, tags, xaxis, transparent, plotformat):
    stuff = OrderedDict([
        ("exp-2", (r"Expected $-2\sigma$", "cornflowerblue")),
        ("exp+2", (r"Expected $+2\sigma$", "lightcoral")),
        ("exp-1", (r"Expected $-1\sigma$", "darkblue")),
        ("exp+1", (r"Expected $+1\sigma$", "darkred")),
        ("exp0", (r"Expected", "dimgrey")),
        ("obs", ("Observed", "black"))
    ])

    for tt, directory in enumerate(directories):
        for jj, dcd in enumerate(directory):
            limits = read_cls(dcd, tags[tt])
            fig, ax = plt.subplots()

            ax.plot(np.array([min_g, max_g]), np.array([0.05, 0.05]), color = 'darkolivegreen', linestyle = '--', linewidth = 1.5)
            for quantile, cls in limits.items():
                ax.plot(np.array([g for g, c in cls]), np.array([c for g, c in cls]), label = stuff[quantile][0],
                        color = stuff[quantile][1], linestyle = '--', marker = 'o', markersize = 1, linewidth = 1.5)

            plt.xlim((min_g, max_g))
            plt.xlabel(xaxis, fontsize = 13, loc = "right")
            plt.ylabel(r"$\mathrm{CL}_{\mathrm{s}}$", fontsize = 13, loc = "top")
            ax.set_yscale('log')
            ax.legend(title = "${point}$".format(point = str_point('_'.join(dcd.split('_')[:3]))))
            fig.tight_layout()
            for fmt in plotformat:
                fig.savefig("{ooo}/{pnt}_{tag}_cls{fmt}".format(
                    ooo = odir,
                    pnt = '_'.join(dcd.split('_')[:3]),
                    tag = tags[tt],
                    fmt = fmt
                ), transparent = transparent)
            fig.clf()
    pass

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--tag", help = "input tag-output-tag pairs to search. the pairs are semicolon separated, and tags colon-separated, "
                        "so e.g. when there are 2 tags: 't1:o1;t2:o2...", dest = "itag", default = "", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))
    parser.add_argument("--plot-tag", help = "extra tag to append to plot names", dest = "ptag", default = "", required = False, type = prepend_if_not_empty)
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False, type = remove_spaces_quotes)
    parser.add_argument("--drop",
                        help = "comma separated list of points to be dropped. 'XX, YY' means all points containing XX or YY are dropped.",
                        default = "", required = False, type = lambda s: [] if s == "" else tokenize_to_list( remove_spaces_quotes(s) ) )
    parser.add_argument("--keep",
                        help = "comma separated list of points to be kept. 'XX, YY' means all points containing XX or YY are kept.",
                        default = "", required = False, type = lambda s: [] if s == "" else tokenize_to_list( remove_spaces_quotes(s) ) )
    parser.add_argument("--opaque-background", help = "make the background white instead of transparent",
                        dest = "transparent", action = "store_false", required = False)
    parser.add_argument("--plot-formats", help = "comma-separated list of formats to save the plots in", default = [".png"], dest = "fmt", required = False,
                        type = lambda s: [prepend_if_not_empty(fmt, '.') for fmt in tokenize_to_list(remove_spaces_quotes(s))])

    args = parser.parse_args()
    adir = [[pnt for pnt in sorted(glob.glob('A*_w*' + tag.split(':')[0])) if len(args.drop) == 0 or not any([dd in pnt for dd in args.drop])] for tag in args.itag]
    hdir = [[pnt for pnt in sorted(glob.glob('H*_w*' + tag.split(':')[0])) if len(args.drop) == 0 or not any([dd in pnt for dd in args.drop])] for tag in args.itag]
    if len(args.keep) > 0:
        adir = [[pnt for pnt in l if any(k in pnt for k in args.keep)] for l in adir]
        hdir = [[pnt for pnt in l if any(k in pnt for k in args.keep)] for l in hdir]
    otags = [tag.split(':')[1] if len(tag.split(':')) > 1 else tag.split(':')[0] for tag in args.itag]

    apnt = [[get_point(pnt) for pnt in directory] for directory in adir]
    hpnt = [[get_point(pnt) for pnt in directory] for directory in hdir]

    if not all([pnt == apnt[0]] for pnt in apnt):
        raise RuntimeError("A signal points are not the same between args.itag. aborting")

    if not all([pnt == hpnt[0]] for pnt in hpnt):
        raise RuntimeError("H signal points are not the same between args.itag. aborting")

    # keep only the points of the first tag, as they're all the same
    apnt = apnt[0]
    hpnt = hpnt[0]

    for ii in range(len(adir)):
        adir[ii] = [dd for dd, pnt in sorted(zip(adir[ii], apnt), key = lambda tup: (tup[1][1], tup[1][2]))]
    apnt.sort(key = lambda tup: (tup[1], tup[2]))
    for ii in range(len(hdir)):
        hdir[ii] = [dd for dd, pnt in sorted(zip(hdir[ii], hpnt), key = lambda tup: (tup[1][1], tup[1][2]))]
    hpnt.sort(key = lambda tup: (tup[1], tup[2]))

    if len(apnt) > 0:
        draw_cls("{ooo}".format(ooo = args.odir), adir, otags, axes["coupling"] % apnt[0][0], args.transparent, args.fmt)
    if len(hpnt) > 0:
        draw_cls("{ooo}".format(ooo = args.odir), hdir, otags, axes["coupling"] % hpnt[0][0], args.transparent, args.fmt)
    pass
