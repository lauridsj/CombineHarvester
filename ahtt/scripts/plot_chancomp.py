#!/usr/bin/env python3
# draw the cls(g) for some signal points
# requires matplotlib > 3.3 e.g. source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_100 x86_64-centos7-gcc10-opt

from argparse import ArgumentParser
import os
import sys
import numpy as np
import math

import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drawings import min_g, max_g, epsilon, axes, first, second, get_point, str_point, stock_labels
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes

from utilspy import recursive_glob

from ROOT import TTree, TFile

def summarize_chancomp(odir, directories, points, transparent, plotformat):
    for idir in directories:
        allobs = recursive_glob(idir[0], "*_chancomp_*_obs.root")
        for obs in allobs:
            print(obs)
            ofile = TFile.Open(obs)
            otree = ofile.Get("chancomp")
            pois = otree.GetListOfBranches()
            pois = [bb.GetName() for bb in pois]
            pois = [bb for bb in pois if "NLL" not in bb and "_error" not in bb]
            poiname = [bb for bb in pois if bb.endswith("_global")][0]
            igl = pois.index(poiname)
            pois[0], pois[igl] = pois[igl], pois[0]
            poiname = poiname.replace("_global", "")

            vals = []
            errs = []
            pvals = []
            for i in otree:
                pvals.append(getattr(otree, "ddNLL"))
                for poi in pois:
                    vals.append(getattr(otree, poi))
                    errs.append(getattr(otree, poi + "_error"))

            names = [poi.replace(poiname + "_", "") for poi in pois]
            fig, ax = plt.subplots()
            for ipnt in range(len(pois)):
                vhi = vals[ipnt] + errs[ipnt]
                vlo = vals[ipnt] - errs[ipnt]
                ax.fill_between([ipnt, ipnt + 1], [vlo, vlo], [vhi, vhi], color = "#607641", linewidth = 0, alpha = 1.)
                ax.plot([ipnt, ipnt + 1], [vals[ipnt], vals[ipnt]], color = 'black', linestyle = "dashed", linewidth = 1.5)
            plt.xticks([0.5 + ii for ii in range(len(pois))], names)
            plt.xlim(0, len(pois))
            plt.ylabel(stock_labels([poiname], points)[0], fontsize = 15, loc = "top")
            plt.xlabel("channels", fontsize = 15, loc = "right")
            if len(pois) > 25:
                fig.set_size_inches(25., 5.)
            elif len(pois) > 15:
                fig.set_size_inches(20., 5.)
            elif len(pois) > 8:
                fig.set_size_inches(15., 5.)
            else:
                fig.set_size_inches(8., 6.)
            fig.set_dpi(450)
            fig.tight_layout()
            for fmt in plotformat:
                fig.savefig("{ooo}/{fnm}{fmt}".format(
                    ooo = odir,
                    fnm = obs.split('/')[-1].replace("_obs.root", "_chancomp-result"),
                    fmt = fmt
                ), transparent = transparent)
            fig.clf()

            alltoys = recursive_glob(idir[0], obs.split('/')[-1].replace("_obs.root", "_toys*.root"))
            for toy in alltoys:
                tfile = TFile.Open(toy)
                ttree = tfile.Get("chancomp")
                for i in ttree:
                    pvals.append(getattr(ttree, "ddNLL"))

            fig, ax = plt.subplots()
            ax.hist(np.array(pvals[1:]), bins = 20, label = 'toys')
            plt.axvline(pvals[0], color = 'black', linestyle = "dashed", linewidth = 1.5, label = 'observed')
            pval = round(len([pv for pv in pvals if pv > pvals[0]]) / (len(pvals) - 1), 2)
            plt.legend(fontsize = 15, framealpha = 0, title = f'ntoy {len(pvals) - 1} p-value {pval}', title_fontsize = 15)
            plt.xlabel("test statistic for " + stock_labels([poiname], points)[0], fontsize = 15, loc = "right")
            plt.ylabel("counts", fontsize = 15, loc = "top")
            fig.set_size_inches(8., 6.)
            fig.set_dpi(450)
            fig.tight_layout()
            for fmt in plotformat:
                fig.savefig("{ooo}/{fnm}{fmt}".format(
                    ooo = odir,
                    fnm = obs.split('/')[-1].replace("_obs.root", "_chancomp-toys"),
                    fmt = fmt
                ), transparent = transparent)
            fig.clf()
    pass

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "signal point pair", default = "", required = True, type = lambda s: sorted(tokenize_to_list( remove_spaces_quotes(s) )))
    parser.add_argument("--tag", help = "input tag-output-tag pairs to search. the pairs are semicolon separated, and tags colon-separated, "
                        "so e.g. when there are 2 tags: 't1:o1;t2:o2...", dest = "itag", default = "", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False, type = remove_spaces_quotes)
    parser.add_argument("--opaque-background", help = "make the background white instead of transparent",
                        dest = "transparent", action = "store_false", required = False)
    parser.add_argument("--plot-formats", help = "comma-separated list of formats to save the plots in", default = [".png"], dest = "fmt", required = False,
                        type = lambda s: [prepend_if_not_empty(fmt, '.') for fmt in tokenize_to_list(remove_spaces_quotes(s))])

    args = parser.parse_args()
    points = args.point
    if len(points) != 2:
        raise RuntimeError("this script is to be used with exactly two A/H points!")
    pstr = "__".join(points)

    if len(args.itag) != 1:
        raise RuntimeError("unsupported len > 1. aborting")

    dirs = [tag.split(':') for tag in args.itag]
    dirs = [tag + tag[:1] if len(tag) == 2 else tag for tag in dirs]
    dirs = [[f"{pstr}_{tag[0]}"] + tag[1:] for tag in dirs]

    summarize_chancomp(args.odir, dirs, points, args.transparent, args.fmt)
    pass
