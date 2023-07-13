import ROOT
import argparse
import numpy as np
import os
import uproot
import matplotlib.pyplot as plt
import mplhep

parser = argparse.ArgumentParser()
parser.add_argument("rootfile")
parser.add_argument("inputtemplates")
args = parser.parse_args()

var_name = "g1"
category = "em_2018"
sig = "A_m400_w5p0"
parts = ["pos", "neg", "res"]
bgs = ["TT", "TQ", "TB", "TW", "TTV", "VV", "DY"]

procs = []
procs.extend(bgs)
for part in parts:
    procs.append(sig + "_" + part)

templates = {}

with uproot.open(args.inputtemplates) as rfile:
    for proc in procs:
        templates[proc] = rfile[category + "/" + proc].values()

n_bins = len(templates["TT"])

f = ROOT.TFile.Open(args.rootfile)
w = f.Get("w")

func = w.function("prop_bin" + category)
x_var = w.var("CMS_th1x")
y_var = w.var(var_name)

w.var("g2").setVal(0.)

def get_combine_events(var_val):
    n_exp = np.empty(n_bins)

    y_var.setVal(var_val)
    for i in range(n_bins):
        x_var.setVal(i + 0.5)
        n_exp[i] = func.getVal()

    return n_exp

def get_naive_events(g):
    n_exp = np.zeros(n_bins)

    for bg in bgs:
        n_exp += templates[bg]
    
    n_exp += g**2 * templates[sig + "_pos"]
    n_exp -= g**2 * templates[sig + "_neg"]
    n_exp += g**4 * templates[sig + "_res"]

    return n_exp

fig, (ax1, ax2) = plt.subplots(dpi=200, nrows=2)

colors = plt.get_cmap("tab10")
edges = np.arange(n_bins+1)

for i, g in enumerate([0., 0.5, 1., 2.]):
    n_combine = get_combine_events(g)
    n_naive = get_naive_events(g)

    print("Agreement for g = {g}: {a}".format(g=g, a=np.all(np.isclose(n_combine, n_naive))))

    mplhep.histplot(n_combine, edges, color=colors(i), linestyle="dotted", linewidth=1, edges=False, ax=ax1, label="g = {g}".format(g=g))
    mplhep.histplot(n_naive, edges, color=colors(i), linestyle="dashed", linewidth=1, edges=False, ax=ax1)

    mplhep.histplot(n_combine-n_naive, edges, color=colors(i), linewidth=1, edges=False, ax=ax2)

ax1.set_yscale("log")
ax1.autoscale()
ax2.autoscale()
ax1.legend()
plt.savefig("eventyields.png")
plt.close()