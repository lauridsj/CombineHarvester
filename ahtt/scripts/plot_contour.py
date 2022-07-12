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

        for gv in cc["g-grid"].keys():
            contours[ii]["g1"].append( gv.replace(" ", "").split(",")[0] )
            contours[ii]["g2"].append( gv.replace(" ", "").split(",")[1] )
            contours[ii]["eff"].append( cc["g-grid"][gv]["pass"] / cc["g-grid"][gv]["total"] )

    return contours

def draw_contour(ofile, pair, contours, labels, maxsigma, transparent):
    pass

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--contour", help = "the json files containing the contour information, comma separated. must be of the same signal pair.", default = "", required = True)
    parser.add_argument("--otag", help = "extra tag to append to plot names", default = "", required = False)
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False)
    parser.add_argument("--max-sigma", help = "max number of sigmas to be drawn on the contour", dest = "maxsigma", default = 2, type = int, required = False)
    parser.add_argument("--label", help = "labels to attach on plot for each json input, semicolon separated", default = "", required = False)
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

    pairs = [os.path.basename(cc).split(".")[0].split("__") for cc in contours.split(",")]
    pairs = [[pp[0], "_".join(pp[1].split("_")[:3])] for pp in pairs]

    if not all([pp == pairs[0] for pp in pairs]):
        raise RuntimeError("provided contours are not all of the same pair of points!!")

    draw_contour("{ooo}/{prs}_fc-contour_{tag}{fmt}".format(ooo = args.odir, tag = args.otag, fmt = args.fmt), pair[0], contours, labels, args.maxsigma, args.transparent)
    pass
