#!/usr/bin/env python3
# environment: source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_102 x86_64-centos7-gcc11-opt
# should work without any additional packages hopefully

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt # noqa:E402
import mplhep # noqa:E402
import uproot # noqa:E402
import itertools
import traceback

import warnings
warnings.filterwarnings("ignore", "invalid value encountered in true_divide")

parser = argparse.ArgumentParser()
parser.add_argument("input", help="The ahtt_input.root file to plot")
parser.add_argument("outfolder", help="Output folder for plotting")
parser.add_argument("--process", "-p", default=None, help="Plot only this process")
parser.add_argument("--syst", "-s", default=None, help="Plot only this systematic")
parser.add_argument("--channel", "-c", default=None, help="Plot only this channel")
parser.add_argument("--year", "-y", default=None, help="Plot only this year")
parser.add_argument("--ext", default="pdf", help="File extension")

args = parser.parse_args()

if args.channel is not None:
    channels = [args.channel]
else:
    channels = ["ee", "em", "mm", "e3j", "e4pj", "m3j", "m4pj"]

if args.year is not None:
    years = [args.year]
else:
    years = ["2016pre", "2016post", "2017", "2018"]

def plot_syst(subf, proc, systname, ann, outfile):
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 1]}, figsize=(8,6))

    vals_nom = subf[proc].values()
    err_nom = subf[proc].errors()

    if systname == "EWK_yukawa":
        vals_nom = subf["TT"].values()
        err_nom = subf["TT"].errors()
        vals_up = vals_nom.copy()
        vals_down = vals_nom.copy()

        for proc in ["quad_pos", "quad_neg", "lin_pos", "lin_neg", "const_pos", "const_neg"]:
            sign = -1 if "neg" in proc else 1
            hi = 1.11**2 if "quad" in proc else 1.11 if "lin" in proc else 1
            lo = 0.88**2 if "quad" in proc else 0.88 if "lin" in proc else 1
            vals_nom += sign * subf[f'EWK_TT_{proc}'].values()
            vals_up += hi * sign * subf[f'EWK_TT_{proc}'].values()
            vals_down += lo * sign * subf[f'EWK_TT_{proc}'].values()
            err_nom = (err_nom**2 + subf[f'EWK_TT_{proc}'].errors()**2)**0.5
        proc = "TT"
    elif systname == "EWK_scheme":
        vals_nom = subf["TT"].values()
        err_nom = subf["TT"].errors()
        vals_up = vals_nom.copy()
        vals_down = vals_nom.copy()

        for proc in ["quad_pos", "quad_neg", "lin_pos", "lin_neg", "const_pos", "const_neg"]:
            sign = -1 if "neg" in proc else 1
            vals_nom += sign * subf[f'EWK_TT_{proc}'].values()
            vals_up += sign * subf[f'EWK_TT_{proc}_{systname}Up'].values()
            vals_down += sign * subf[f'EWK_TT_{proc}_{systname}Down'].values()
            err_nom = (err_nom**2 + subf[f'EWK_TT_{proc}'].errors()**2)**0.5
        proc = "TT"
    else:
        vals_up = subf[f"{proc}_{systname}Up"].values()
        vals_down = subf[f"{proc}_{systname}Down"].values()

    edges = np.arange(len(vals_nom)+1)
    mplhep.histplot(
            vals_nom,
            edges,
            yerr=0.,
            histtype="step", 
            edges=False,
            linewidth=1,
            label=f"{proc} nominal",
            color="black",
            ax=ax1
        )
    
    abs_unc_down = vals_nom - err_nom
    abs_unc_down = np.concatenate([abs_unc_down, [abs_unc_down[-1]]])
    abs_unc_up = vals_nom + err_nom
    abs_unc_up = np.concatenate([abs_unc_up, [abs_unc_up[-1]]])
    ax1.fill_between(edges, abs_unc_down, abs_unc_up, step="post", color="lightgrey", label="MC stat unc.", zorder=-50)

    ratio_unc = err_nom / vals_nom
    ratio_unc = np.concatenate([ratio_unc, [ratio_unc[-1]]])
    #ax2.fill_between(edges, 1-ratio_unc, 1+ratio_unc, step="post", color="lightgrey", label="MC stat unc.", zorder=-50)
    
    ax2.hlines(1., edges[0], edges[-1], color="black", linestyle="dashed", linewidth=0.7)

    ratiolim = 1e-5

    for vals, label, color in zip(
            [vals_down, vals_up],
            ["down", "up"],
            ["royalblue", "orangered"]
        ):

        mplhep.histplot(
            vals,
            edges,
            yerr=0.,
            histtype="step", 
            edges=False,
            linewidth=1,
            label=f"{proc} {systname} {label}",
            color=color,
            ax=ax1
        )

        ratio_vals = vals / vals_nom
        mplhep.histplot(
            ratio_vals,
            edges,
            yerr=0.,
            histtype="step", 
            edges=False,
            linewidth=1,
            label=f"{proc} {systname} {label}",
            color=color,
            ax=ax2
        )

        ratio_vals = np.nan_to_num(ratio_vals, nan=1., posinf=1., neginf=1.)
        ratiolim = max(ratiolim, np.amax(abs(ratio_vals-1.)))

    ax1.set_xlabel("")
    ax1.set_ylabel("Event yield")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.minorticks_on()
    ax1.tick_params(direction="in", top=True, bottom=True, left=True, right=True, which='both')
    ax2.tick_params(direction="in", top=True, bottom=True, left=True, right=True, which='both')

    ax2.set_ylabel("Ratio")
    ax2.set_xlim(edges[0], edges[-1])
    ax2.set_ylim(1.-ratiolim*1.05, 1.+ratiolim*1.05)
    ax2.set_xlabel("bin index")

    ax1.legend(frameon=False)

    ax1.annotate(ann, (0.02, 0.98), xycoords="axes fraction", va="top")

    fig.align_ylabels()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    fig.savefig(outfile)
    plt.close()


with uproot.open(args.input) as f:
    for (channel, year) in itertools.product(channels, years):
        key = channel + "_" + year
        if key in f:
            print(f"Plotting channel {channel} for year {year}")
            subf = f[key]
            all_templates = [k[:-2] for k in subf.keys()]
            if args.process is not None:
                if args.process not in subf:
                    print(f"Nominal for process {args.process} not found. Skipping channel/year")
                    continue
                procs = [args.process]
            else:
                procs = [p for p in all_templates if not p.endswith("Down") and not p.endswith("Up")]

            for p in procs:
                if p == "data_obs":
                    continue

                outfolder = os.path.join(args.outfolder, key + "/" + p)
                os.makedirs(outfolder, exist_ok=True)

                if args.syst is not None:
                    if not (f"{p}_{args.syst}Down" in subf and f"{p}_{args.syst}Up" in subf) and "EWK_" not in args.syst:
                        print(f"Systematic {args.syst} for process {p} not found. Skipping process")
                        continue
                    systs = [args.syst]
                else:
                    systs = [s for s in all_templates if s.startswith(p + "_") and s.endswith("Up")]
                    systs = [s[len(p)+1:-2] for s in systs]

                for sys in systs:
                    outfile = os.path.join(outfolder, f"{p}_{sys}_{key}.{args.ext}")
                    try:
                        plot_syst(subf, p, sys, key, outfile)
                    except ValueError as ex:
                        print(f"Error for systematic {sys}, process {p}, category {key}:")
                        traceback.print_exc()

            

