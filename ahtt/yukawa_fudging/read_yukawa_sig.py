import ROOT
import argparse
import numpy as np
import os
import uproot
import matplotlib.pyplot as plt
import mplhep
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser()
parser.add_argument("rootfile")
parser.add_argument("inputtemplates")
args = parser.parse_args()

var_name = "EWK_yukawa"
category = "em_2018"
procs = ["TT", "TQ", "TB", "TW", "TTV", "VV", "DY"]

dyt_up = 0.11
dyt_down = 0.12

yt_nom = 1.0
yt_up = yt_nom + dyt_up
yt_down = yt_nom - dyt_down

templates = {}
templates_up = {}
templates_down = {}

with uproot.open(args.inputtemplates) as rfile:
    rcat = rfile[category]
    for proc in procs:
        templates[proc] = rcat[proc].values()
        if proc + "_" + var_name + "Up" in rcat:
            templates_up[proc] = rcat[proc + "_" + var_name + "Up"].values()
            templates_down[proc] = rcat[proc + "_" + var_name + "Down"].values()

interpolators = {}

for proc in procs:
    if proc in templates_up:
        interpolators[proc] = interp1d(
            [yt_down, yt_nom, yt_up],
            [templates_down[proc], templates[proc], templates_up[proc]],
            axis=0, kind="quadratic", fill_value="extrapolate")

n_bins = len(templates["TT"])

f = ROOT.TFile.Open(args.rootfile)
w = f.Get("w")

func = w.function("prop_bin" + category)
x_var = w.var("CMS_th1x")
y_var = w.var("dyt")

w.var("g1").setVal(0.)
w.var("g2").setVal(0.)

def get_combine_events(var_val):
    n_exp = np.empty(n_bins)

    y_var.setVal(var_val)
    for i in range(n_bins):
        x_var.setVal(i + 0.5)
        n_exp[i] = func.getVal()

    return n_exp

def get_naive_events(yt):
    n_exp = np.zeros(n_bins)

    for proc in procs:
        if proc in interpolators:
            n_exp += interpolators[proc](yt)
        else:
            n_exp += templates[proc]

    return n_exp

def get_bg_events():
    n_exp = np.zeros(n_bins)

    for proc in procs:
        n_exp += templates[proc]

    return n_exp


fig, (ax1, ax2) = plt.subplots(dpi=200, nrows=2)

colors = plt.get_cmap("tab10")
edges = np.arange(n_bins+1)

for i, sigmayt in enumerate([-2, -1, -0.5, 0, 0.5, 1, 2]):
    dyt = sigmayt * dyt_up if sigmayt >= 0 else sigmayt * dyt_down
    yt = yt_nom + dyt
    n_combine = get_combine_events(dyt)
    n_naive = get_naive_events(yt)

    print("Agreement for yt = {g}: {a}".format(g=yt, a=np.all(np.isclose(n_combine, n_naive))))

    mplhep.histplot(n_combine, edges, color=colors(i), linestyle="dotted", linewidth=1, edges=False, ax=ax1, label="yt = {g}".format(g=yt))
    mplhep.histplot(n_naive, edges, color=colors(i), linestyle="dashed", linewidth=1, edges=False, ax=ax1)

    mplhep.histplot(n_combine/n_naive, edges, color=colors(i), linewidth=1, edges=False, ax=ax2)

ax1.set_yscale("log")
ax1.autoscale()
ax2.autoscale()
ax1.legend()
ax1.set_ylabel("event yield")
ax2.set_ylabel("combine / naive")
plt.title("gA = gH = 0")
plt.savefig("yukawayields_sig.png")
plt.close()