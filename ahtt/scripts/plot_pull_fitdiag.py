from argparse import ArgumentParser
import json
import numpy as np
import math
import ROOT
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes
from glob import glob

def translate_name(name, ndict):
    
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
    label = label.replace("$\\mathrm{\\mu}$_{R}", "$\\mathrm{\\mu_R}$")
    label = label.replace("$\\mathrm{\\mu}$_{F}", "$\\mathrm{\\mu_F}$")
    label = label.replace("\\gamma", "$\\mathrm{\\gamma}$")
    label = label.replace("\\eta_{t}", "$\\mathrm{\\eta_t}$")
    label = label.replace("\\geq", "$\\geq$")
    label = label.replace("p_{T}", "$p_{\\mathrm{T}}$")
    return label

parser = ArgumentParser()
parser.add_argument("--point", help = "signal point pair", default = "", required = True, type = lambda s: sorted(tokenize_to_list( remove_spaces_quotes(s) )))
parser.add_argument("--tag", help = "input tag-output-tag pairs to search. the pairs are semicolon separated, and tags colon-separated, "
                    "so e.g. when there are 2 tags: 't1:o1;t2:o2...", dest = "itag", default = "", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))
parser.add_argument("--outfile", "-o", type=str, required=True)
parser.add_argument("--label", "-l", help = "labels to attach on plot for each input tags, semicolon separated", default = "", required = False, type = lambda s: tokenize_to_list(s, ';' ))
parser.add_argument("--nuisance_map", "-t", type=str, default=None, help="NP label translation map")
parser.add_argument("--impacts", "-i", type=str, default=None, help="Impact json for sorting")
parser.add_argument("--sort_by", "-s", type=str, default=None, help="list of NPs to sort by")
parser.add_argument("--poi",  type=str, default="CMS_EtaT_norm_13TeV")
parser.add_argument("--per_page",  type=int, default=20)
parser.add_argument("--per_line",  type=int, default=1)
parser.add_argument("--add_poi", action="store_true")

args = parser.parse_args()

filelabels = args.label

points = args.point
if len(points) != 2:
    raise RuntimeError("this script is to be used with exactly two A/H points!")
pstr = "__".join(points)
dirs = [tag.split(':') for tag in args.itag]
dirs = [tag + tag[:1] if len(tag) == 2 else tag for tag in dirs]
dirs = [[f"{pstr}_{tag[0]}"] + tag[1:] for tag in dirs]

infiles = [
    glob(f"{d[0]}/{pstr}_{d[2]}_fitdiagnostics_result*_{d[1]}.root") for d in dirs
]

if not all(len(f) == 1 for f in infiles):
    print(infiles)
    raise RuntimeError("Wrong number of input files")

if len(args.itag) != len(args.label):
    raise RuntimeError("length of tags isnt the same as labels. aborting")

infiles = [f[0] for f in infiles]

print(infiles)

nuisance_map = None
if args.nuisance_map is not None:
    with open(args.nuisance_map) as f:
        nuisance_map = json.load(f)

params_infiles = []

if args.impacts is not None and args.sort_by is not None:
    raise RuntimeError("please give either impacts or sort_by, not both")

if args.impacts is not None:
    with open(args.impacts) as jsonfile:
        impact_data = json.load(jsonfile)
    impact_data = {p['name']: p for p in impact_data['params']}

if args.sort_by is not None:
    with open(args.sort_by) as jsonfile:
        sortby = json.load(jsonfile)

for infile in infiles:

    fittype = "b" if infile.split(".root")[0].endswith("_b") else "s"

    rfile = ROOT.TFile(infile)

    fitres = rfile.Get("fit_" + fittype)
    fitparams = fitres.floatParsFinal()

    params = {}

    for i in range(fitparams.getSize()):
        realvar = fitparams.at(i)
        param = realvar.GetName()

        if param.startswith("prop_bin"):
            continue

        # shitty hack for by-year comparisons
        if param == "lumi_13TeV":
            param = "lumi_13TeV_correlated"

        central = realvar.getVal()
        errup = realvar.getErrorHi()
        errdown = realvar.getErrorLo()

        #print(infile, param, central, errup, errdown)
        if param == "EWK_yukawa":
            central -= 1.0
            if central < 0:
                central /= 0.12
            else:
                central /= 0.11
            errup /= 0.11
            errdown /= 0.12

        params[param] = (central, errup, errdown)

    rfile.Close()

    params_infiles.append(params)

if args.sort_by is not None:
    all_params = sortby
else:
    all_params = []
    for p in params_infiles:
        all_params.extend(p.keys())
    all_params = list(set(all_params))
    if args.poi in all_params:
        all_params.remove(args.poi)

    def get_impact(p):
        if p == args.poi:
            return np.inf
        if not p in impact_data:
            print(f"Missing impact: {p}")
            return 0.
        else:
            return abs(impact_data[p]['impact_' + args.poi])

    if args.impacts is not None:
        all_params = sorted(all_params, reverse=True, 
            key=get_impact)
    else:
        all_params = sorted(all_params, reverse=False)

params = params_infiles[0]

nparams = min(len(all_params), args.per_page)

colors = ["#cc0033", "#0033cc", "forestgreen", "rebeccapurple", "orange"]
markers = ["o", "D", "s"]

pdf = PdfPages(args.outfile)

npages = int(np.ceil(len(all_params) / args.per_page))

for ipage in range(npages):

    print(f"Page {ipage+1} of {npages}")

    paramkeys = all_params[nparams*ipage:nparams*(ipage+1)]

    y = np.arange(len(paramkeys))

    if args.nuisance_map is not None:
        labels = [translate_name(p, nuisance_map) for p in paramkeys]
    else:
        labels = paramkeys


    figsize=(5, 8*len(paramkeys)/20)
    plt.figure(dpi=200, figsize=figsize)

    for i, params in enumerate(params_infiles):

        central = [params[k][0] if k in params else np.nan for k in paramkeys]
        errup = [abs(params[k][1]) if k in params else np.nan for k in paramkeys]
        errdown = [abs(params[k][2]) if k in params else np.nan for k in paramkeys]

        imarker = i % args.per_line
        icolor = (i-imarker) // args.per_line
        totlines = int(np.ceil(len(infiles) / args.per_line))

        offset = 0.45 - 0.9 * (icolor+1) / (totlines + 1)

        fl=filelabels[i]
        if args.add_poi:
            if args.poi == "CMS_EtaT_norm_13TeV":
                poiname = "\\mu(\\eta_t)"
            else:
                poiname = args.poi

            if dirs[i][1] == "s" and args.poi in params:
                poi_central, poi_up, poi_down = params[args.poi]
                if round(abs(poi_up), 2) == round(abs(poi_down), 2):
                    fl += f", ${poiname} = {poi_central:.2f} \\pm {abs(poi_up):.2f}$"
                else:
                    fl += f", ${poiname} = {poi_central:.2f} + {abs(poi_up):.2f} - {abs(poi_down):.2f}$"
                
            elif dirs[i][1] == "b":
                fl += f", ${poiname} = 0$ (fixed)"
            else:
                print(f"Could not get value for poi {args.poi} for infile {infiles[i]}")

        

        plt.errorbar(central, y[::-1]+offset, xerr=(errdown, errup), yerr=None, linestyle="none", marker=markers[imarker//2], markersize=5,
                    label=fl, color=colors[icolor], markeredgecolor=colors[icolor], markerfacecolor=colors[icolor] if imarker % 2 == 0 else "white",
                    capsize=3. if imarker % 2 == 1 else 0., capthick=1.0,
                    elinewidth=2. if imarker % 2 == 0 else 1.0)

    xlim = max(abs(v[0]) for d in params_infiles for v in d.values())
    xlim = max(3, np.ceil(xlim))

    plt.xlim(-xlim,xlim)
    plt.yticks(y, labels[::-1])
    plt.ylim(-0.5, len(paramkeys)-0.5)
    plt.xlabel("Nuisance pull")

    for i in range(len(paramkeys)):
        if i % 2 ==0:
            plt.fill_between([-xlim,xlim], [i-0.5, i-0.5], [i+0.5, i+0.5], facecolor="lightgrey", linewidth=0., zorder=-20)

    plt.vlines(np.arange(-xlim, xlim+1), -1, len(paramkeys), colors="black", linestyles="dashed", linewidth=1.0, zorder=-10)

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = handles[::2] + handles[1::2]
    labels = labels[::2] + labels[1::2]
    plt.legend(handles, labels, frameon=False, ncol=2, bbox_to_anchor=(-0.5, 1.0, 1.5, 0.1 * (len(infiles)//2+1)), mode="expand", loc="lower left")

    #plt.savefig(args.outfile, bbox_inches="tight")
    pdf.savefig(plt.gcf(), bbox_inches="tight")
    plt.close()

pdf.close()