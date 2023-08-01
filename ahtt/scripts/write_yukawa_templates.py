import uproot
import argparse
import numpy as np
from collections import defaultdict
import hist

parser = argparse.ArgumentParser()
parser.add_argument("inputtemplates")
parser.add_argument("outputfile")
parser.add_argument("--plotout", default = None)
args = parser.parse_args()

proc = "TT"
var_name = "EWK_yukawa"

channels = ["em", "ee", "mm", "e3jets", "mu3jets", "e4pjets", "mu4pjets"]
years = ["2016pre", "2016post", "2017", "2018"]

# Weight-based systematics to copy to the EWK templates
# It is assumed that EWK_yukawa and the other nuisance are uncorellated. Since they are both
# weight-based, the template for both systematics varied by 1sigma is then just
# (EW_yukawa up/down) * (other up/down) / (nominal)
# Then, the same quadratic equation is solved for the systematic as for the nominal to get
# the variation templates for the EWK signal.

systs_to_copy = [
    "EWK_scheme",
    "CMS_eff_e_reco",
    "CMS_eff_e_id",
    "CMS_eff_m_id_stat",
    "CMS_eff_m_id_syst",
    "CMS_eff_m_iso_stat",
    "CMS_eff_m_iso_syst",
    "CMS_eff_trigger_ee",
    "CMS_eff_trigger_em",
    "CMS_eff_trigger_mm",
    "CMS_L1_prefire",
    "CMS_eff_trigger_m_syst",
    "CMS_eff_trigger_m_stat",
    "CMS_eff_trigger_e",
]

dyt_up = 0.11
dyt_down = -0.12

yt_nom = 1.0
yt_up = yt_nom + dyt_up
yt_down = yt_nom + dyt_down

templates = defaultdict(dict)
templates_yukawa_up = {}
templates_yukawa_down = {}

with uproot.open(args.inputtemplates) as rfile:
    for channel in channels:
        for year in years:
            cat = channel + "_" + year
            print("Loading templates for " + cat)
            if cat in rfile:
                rcat = rfile[cat]
                templates[cat]["nominal"] = rcat[proc].values()
                templates_yukawa_up[cat] = rcat[proc + "_" + var_name + "Up"].values()
                templates_yukawa_down[cat] = rcat[proc + "_" + var_name + "Down"].values()
                for sys in systs_to_copy:
                    for sysdir in ["Up", "Down"]:
                        syskey = proc + "_" + sys + sysdir
                        if syskey in rcat:
                            templates[cat][sys + sysdir] = rcat[syskey].values()

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
    for sys in templates[cat].keys():
        print("Calculating templates for " + cat + " " + sys)

        template_yukawa_nominal = templates[cat][sys]

        template_yukawa_up = templates_yukawa_up[cat] * templates[cat][sys] / templates[cat]["nominal"]
        template_yukawa_down = templates_yukawa_down[cat] * templates[cat][sys] / templates[cat]["nominal"]

        dN_up = template_yukawa_up - template_yukawa_nominal
        dN_down = template_yukawa_down - template_yukawa_nominal

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
            return template_yukawa_nominal + \
                dyt * a_pos - dyt * a_neg + \
                dyt**2 * b_pos - dyt**2 * b_neg

        assert np.all(np.isclose(predict_yukawa(0.), template_yukawa_nominal)) # nominal
        assert np.all(np.isclose(predict_yukawa(dyt_up), template_yukawa_up)) # up
        assert np.all(np.isclose(predict_yukawa(dyt_down), template_yukawa_down)) # down

        # Add bin edges so that uproot saves it as a TH1D
        bin_edges = np.arange(len(template_yukawa_nominal)+1)

        syskey = "" if sys == "nominal" else "_" + sys

        variances = np.zeros_like(a_pos)

        hist_b_pos = hist.Hist(hist.axis.Variable(bin_edges), "Weight", data=np.stack([b_pos, variances], axis=-1))
        hist_b_neg = hist.Hist(hist.axis.Variable(bin_edges), "Weight", data=np.stack([b_neg, variances], axis=-1))
        hist_a_pos = hist.Hist(hist.axis.Variable(bin_edges), "Weight", data=np.stack([a_pos, variances], axis=-1))
        hist_a_neg = hist.Hist(hist.axis.Variable(bin_edges), "Weight", data=np.stack([a_neg, variances], axis=-1))

        output_templates[cat + "/EWK_TT_quad_pos" + syskey] = hist_b_pos
        output_templates[cat + "/EWK_TT_quad_neg" + syskey] = hist_b_neg
        output_templates[cat + "/EWK_TT_lin_pos" + syskey] = hist_a_pos
        output_templates[cat + "/EWK_TT_lin_neg" + syskey] = hist_a_neg

if args.plotout is not None:
    import os
    import matplotlib.pyplot as plt
    import mplhep
    import warnings
    warnings.filterwarnings("ignore", "invalid value encountered in divide")

    if not os.path.isdir(args.plotout):
        os.makedirs(args.plotout)
    for cat in templates.keys():
        for part in ["EWK_TT_quad_pos", "EWK_TT_quad_neg", "EWK_TT_lin_pos", "EWK_TT_lin_neg"]:

            # nominal
            name = cat + "/" + part
            hist, edges = output_templates[name]

            print("Plotting " + name)

            plt.figure(dpi=200)
            mplhep.histplot(hist, edges)
            plt.title(name)
            plt.savefig(args.plotout + "/" + name.replace("/", "_") + ".png")
            plt.close()

            # variations
            for sys in systs_to_copy:
                if name + "_" + sys + "Up" in output_templates:
                    name_sys = name + "_" + sys
                    hist_up, edges_up = output_templates[name_sys + "Up"]
                    hist_down, edges_down = output_templates[name_sys + "Down"]

                    ratio_up = np.nan_to_num(hist_up / hist, nan = 1.)
                    ratio_down = np.nan_to_num(hist_down / hist, nan = 1.)

                    plt.figure(dpi=200)
                    mplhep.histplot(ratio_up, edges, edges = False, label = "up", color = "orangered")
                    mplhep.histplot(ratio_down, edges, edges = False, label = "down", color = "royalblue")
                    plt.legend()
                    plt.title(name_sys)
                    ratiolim = max(np.amax(np.abs(ratio_up - 1)), np.amax(np.abs(ratio_down - 1))) * 1.1
                    plt.ylim(1-ratiolim, 1+ratiolim)
                    plt.savefig(args.plotout + "/" + name_sys.replace("/", "_") + ".png")
                    plt.close()

print("Writing to disk")
with uproot.recreate(args.outputfile) as rfile:
    for key, hist in output_templates.items():
        rfile[key] = hist

