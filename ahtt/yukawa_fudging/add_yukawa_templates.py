import uproot
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("inputtemplates")
parser.add_argument("--plotout", default=None)
args = parser.parse_args()

proc = "TT"
var_name = "EWK_yukawa"

channels = ["em", "ee", "mm", "e3j", "m3j", "e4pj", "m4pj"]
years = ["2016pre", "2016post", "2017", "2018"]

dyt_up = 0.11
dyt_down = -0.12

yt_nom = 1.0
yt_up = yt_nom + dyt_up
yt_down = yt_nom + dyt_down

templates = {}
templates_up = {}
templates_down = {}

with uproot.open(args.inputtemplates) as rfile:
    for channel in channels:
        for year in years:
            cat = channel + "_" + year
            rcat = rfile[cat]
            templates[cat] = rcat[proc].values()
            if proc + "_" + var_name + "Up" in rcat:
                templates_up[cat] = rcat[proc + "_" + var_name + "Up"].values()
                templates_down[cat] = rcat[proc + "_" + var_name + "Down"].values()

# We define dyt = yt - 1 and set for the yield:
# N(EWK + TT) = N(TT) + a * dyt + b * dyt^2
# where a and b are input templates for the linear and quadratic parts
#
# To get a and b, we use the existing Yukawa variation N(TT)_up and N(TT)_down, and set
#
# N(TT) + a * dyt_up + b * dyt_up^2 == N_up
# N(TT) + a * dyt_down + b * dyt_down^2 == N_down
#
# where dyt_up = +0.11 and dyt_down = -0.12 are the Yukawa values where the variation was evaluated.
# We then solve this system of two equations for a and b, which leads to the formula below.

output_templates = {}

for cat in templates.keys():

    dN_up = templates_up[cat] - templates[cat]
    dN_down = templates_down[cat] - templates[cat]

    denom = dyt_up * dyt_down * (dyt_up - dyt_down)

    a = (dN_down * dyt_up**2 - dN_up * dyt_down**2) / denom
    b = -(dN_down * dyt_up - dN_up * dyt_down) / denom

    # we need to split a and b into negative and positive parts

    a_pos = np.where(a>=0 , a, 0.)
    a_neg = np.where(a<=0 , -a, 0.)
    b_pos = np.where(b>=0 , b, 0.)
    b_neg = np.where(b<=0 , -b, 0.)

    # Check that this reproduces the input templates

    def predict_yukawa(dyt):
        return templates[cat] + \
            dyt * a_pos - dyt * a_neg + \
            dyt**2 * b_pos - dyt**2 * b_neg

    assert np.all(np.isclose(predict_yukawa(0.), templates[cat])) # nominal
    assert np.all(np.isclose(predict_yukawa(dyt_up), templates_up[cat])) # up
    assert np.all(np.isclose(predict_yukawa(dyt_down), templates_down[cat])) # down

    # Add bin edges so that uproot saves it as a TH1D
    bin_edges = np.arange(len(templates[cat])+1)

    output_templates[cat + "/EWK_TT_quad_pos"] = (b_pos, bin_edges)
    output_templates[cat + "/EWK_TT_quad_neg"] = (b_neg, bin_edges)
    output_templates[cat + "/EWK_TT_lin_pos"] = (a_pos, bin_edges)
    output_templates[cat + "/EWK_TT_lin_neg"] = (a_neg, bin_edges)

if args.plotout is not None:
    import os
    import matplotlib.pyplot as plt
    import mplhep

    if not os.path.isdir(args.plotout):
        os.makedirs(args.plotout)
    for name, (hist, edges) in output_templates.items():
        plt.figure(dpi=200)
        mplhep.histplot(hist, edges)
        plt.title(name)
        plt.savefig(args.plotout + "/" + name.replace("/", "_") + ".png")
        plt.close()

#oldfile = {}
#with uproot.open(args.inputtemplates) as rfile:
#    for key in rfile.keys():
#        oldfile[key[:-2]] = rfile[key]
#
with uproot.update(args.inputtemplates) as rfile:
    #for key, hist in oldfile.items():
    #    rfile[key] = hist
    for key, hist in output_templates.items():
        rfile[key] = hist

