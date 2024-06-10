#!/usr/bin/env python3
# analyze a correlation matrix from combine's FitDiagnostics output
# it's basically this lecture note: https://www2.tulane.edu/~PsycStat/dunlap/Psyc613/RI2.html
# environment: source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_102 x86_64-centos7-gcc11-opt

import numpy as np
import uproot
import argparse
from desalinator import tokenize_to_list, remove_spaces_quotes

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type = str, help = "path to fitDiagnostics root file")
parser.add_argument("--signal", type = str, choices = ["A", "H", "EtaT"])
parser.add_argument("--drop", dest = "drop", help = "comma-separated list of NPs to be dropped, greedy matched. signal is not droppable. default is dropping mc stats.",
                    type = lambda s: ["prop"] if s == "" else tokenize_to_list(remove_spaces_quotes(s)))
args = parser.parse_args()

poi_names = {
    "A": "g1",
    "H": "g2",
    "EtaT": "CMS_EtaT_norm_13TeV"
}
poi = poi_names[args.signal]

with uproot.open(args.infile) as f:
    matrix = f["covariance_fit_s"]
    mvals = matrix.values()
    mvals = mvals[::-1, :]
    labels = matrix.axes[0].labels()
    labels = labels[::-1]

    # assume only one signal for now
    keep = [l not in poi_names.values() or l == poi for l in labels]
    mvals = mvals[keep, :][:, keep]
    labels = [l for l in labels if l not in poi_names.values() or l == poi]

    for drop in args.drop:
        keep = [drop not in l or l == poi for l in labels]
        mvals = mvals[keep, :][:, keep]
        labels = [l for l in labels if drop not in l or l == poi]

    if not poi in labels:
        raise ValueError("POI not found")    
    idx = labels.index(poi)

    minvs = np.linalg.inv(mvals)
    det = np.linalg.det(mvals)
    cond = np.linalg.cond(mvals)
    multcorr = np.sqrt(1. - 1./minvs[idx][idx])

    print(f"reading matrix {args.infile}")
    print(f"determinant: {det}")
    print(f"condition: {cond}")
    print(f"multiple correlation coefficient of {poi}: {multcorr}")
