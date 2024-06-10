#!/usr/bin/env python3
# analyze a correlation matrix from combine's FitDiagnostics output
# it's basically this lecture note: https://www2.tulane.edu/~PsycStat/dunlap/Psyc613/RI2.html
# environment: source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_102 x86_64-centos7-gcc11-opt

import numpy as np
import uproot
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type = str, help = "path to fitDiagnostics root file")
parser.add_argument("--signal", type = str, choices = ["A", "H", "EtaT"])
parser.add_argument("--include_mcstats", action = "store_true")
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

    # first stab, assume only one signal
    keep = [l not in poi_names.values() or l == poi for l in labels]
    mvals = mvals[keep, :][:, keep]
    labels = [l for l in labels if l not in poi_names.values() or l == poi]

    if not args.include_mcstats:
        keep = ["prop" not in l for l in labels]
        mvals = mvals[keep, :][:, keep]
        labels = [l for l in labels if "prop" not in l]

    if not poi in labels:
        raise ValueError("POI not found")    
    idx = labels.index(poi)

    minvs = np.linalg.inv(mvals)
    vdet = np.linalg.det(mvals)
    multcorr = np.sqrt(1. - 1./minvs[poi_idx][poi_idx])

    print(f"reading matrix {args.infile}")
    print(f"determinant: {vdet}")
    print(f"multiple correlation coefficient of {poi}: {multcorr}")
