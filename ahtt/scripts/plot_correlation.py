#!/usr/bin/env python3
# plot a correlation matrix from combine's FitDiagnostics output

import numpy as np
import uproot
import argparse
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, help="Path to fitDiagnosticsTest root file")
parser.add_argument("--outfile", type=str, help="Output file")
parser.add_argument("--signal", type=str, choices=["A","H","EtaT"])
parser.add_argument("--only", type=int, help="Plot only the N highest NPs", default=-1)
parser.add_argument("--include_mcstats", action="store_true")
parser.add_argument("--nuisance_map", type=str, default=None)
args = parser.parse_args()

poi_names = {
    "A":"g1",
    "H":"g2",
    "EtaT":"CMS_EtaT_norm_13TeV"
}

poi_labels = {
    "A":"$\\mathrm{g_{A t \\bar{t}}}$",
    "H":"$\\mathrm{g_{H t \\bar{t}}}$",
    "EtaT": "$\\mathrm{\\eta_t}$ signal strength"
}

poi = poi_names[args.signal]
poi_label = poi_labels[args.signal]

def translate_name(name, ndict):

    if name == poi:
        return poi_label
    
    label = name
    nuisances = ndict["nuisances"]
    if name in nuisances:
        label = nuisances[name]
    else:
        for chankey, channame in ndict["channels"].items():
            if chankey in name:
                name_temp = name.replace(chankey, "$CHAN")
                if name_temp in nuisances:
                    label = nuisances[name_temp].replace("$CHAN", channame)
                    break

        for year in ndict["years"]:
            if year in name:
                name_temp = name.replace(year, "$YEAR")
                if name_temp in nuisances:
                    label = nuisances[name_temp].replace("$YEAR", year)
                    break
                else:
                    for chankey, channame in ndict["channels"].items():
                        if chankey in name:
                            name_temp = name_temp.replace(chankey, "$CHAN")
                            if name_temp in nuisances:
                                label = nuisances[name_temp].replace("$YEAR", year).replace("$CHAN", channame)
                                break

        for prockey, procname in ndict["processes"].items():
            if prockey in name:
                name_temp = name.replace(prockey, "$PROC")
                if name_temp in nuisances:
                    label = nuisances[name_temp].replace("$PROC", procname)
                    break
    
    label = label.replace("t\\bar{t}", "$\\mathrm{t\\bar{t}}$")
    label = label.replace("\\alpha_{s}", "$\\mathrm{\\alpha_{s}}$")
    label = label.replace("\\mu", "$\\mathrm{\\mu}$")
    label = label.replace("\\eta", "$\\mathrm{\\eta_t}$")
    return label



nuisance_map = None
if args.nuisance_map is not None:
    with open(args.nuisance_map) as f:
        nuisance_map = json.load(f)

with uproot.open(args.infile) as f:
    matrix = f["covariance_fit_s"]
    mvals = matrix.values()
    mvals = mvals[::-1, :]
    labels = matrix.axes[0].labels()
    labels = labels[::-1]

    if not args.include_mcstats:
        keep = ["prop" not in l for l in labels]
        mvals = mvals[keep, :][:, keep]
        labels = [l for l in labels if "prop" not in l]

    if not poi in labels:
        raise ValueError("POI not found")
    
    poi_ind = labels.index(poi)
    poi_corr = mvals[poi_ind, :]
    sorting = np.argsort(abs(poi_corr))[::-1]

    if args.only > 0:
        sorting = sorting[:args.only]

    mvals = mvals[sorting, :][:, sorting]
    labels = [labels[i] for i in sorting]

    if nuisance_map is not None:
        labels = [translate_name(l, nuisance_map) for l in labels]

    figsize = len(labels) / 5

    fig, ax = plt.subplots(dpi=100, figsize=(figsize,figsize))
    ax.xaxis.tick_top()

    im = plt.imshow(mvals, cmap="RdBu_r", vmin=-1., vmax=1.)

    tick_range = np.arange(0, len(labels))
    plt.minorticks_off()
    plt.xticks(tick_range, labels, fontsize="small", rotation=45, ha="left")
    plt.yticks(tick_range, labels, fontsize="small")

    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.05,ax.get_position().height])
    plt.colorbar(im, label="correlation coefficient", cax=cax)

    ax.annotate("CMS", (-0.04, 1.04), fontsize = 27, ha = 'right', va = 'bottom', usetex = True, xycoords="axes fraction")

    plt.savefig(args.outfile, bbox_inches="tight")
    plt.close()