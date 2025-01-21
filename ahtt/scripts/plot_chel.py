import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mplhep
import math
import os
mplhep.style.use(mplhep.style.CMS)

# currently hardcoded to read the A/H prefit signals from...
ah_signals_dir = "/data/dust/user/afiqaize/cms/ahtt_run2ul_stat_200803/combine/CMSSW_10_2_13/src/CombineHarvester/ahtt/cleanup_1D_240205"

def unc_from_covar(total_covar, ittmin, ittmax, which="chel"):
    variance = np.zeros((3,3))

    channels = ["ee", "em", "mm"]
    if len(total_covar) // (180*4) == 7:
        allchannels = ["e3j", "e4pj", "ee", "em", "m3j", "m4pj", "mm"]
    elif len(total_covar) // (180*4) == 3:
        allchannels = channels
    else:
        raise ValueError("Invalid dimension of covariance matrix")

    def index_in_sum(i):
        ich = math.floor(i / (180*4))
        if allchannels[ich] not in channels:
            return -1
        itemplate = i % 180
        if itemplate % 20 < imttmin or itemplate % 20 >= imttmax:
            return -1
        elif which=="chel":
            return int(math.floor((itemplate % 60) / 20))
        elif which=="chan":
            return int(math.floor(itemplate / 60))
        else:
            raise ValueError()
        

    for i in range(len(total_covar)):
        isum = index_in_sum(i)
        if isum == -1:
            continue
        for j in range(len(total_covar)):
            jsum = index_in_sum(j)
            if jsum == -1:
                continue
            variance[isum,jsum] += total_covar[i,j]

    err = np.sqrt(np.diag(variance))
    return err

def plot(yields_bg, errs_bg, yields_data, yields_signals, labels_signals, colors_signals, fittype, whichvar, outfile):
    fig, (ax1,ax2) = plt.subplots(nrows=2,dpi=600, figsize=(8,10), sharex=True)

    edges = np.linspace(-1,1,4)
    widths = (edges[1:]-edges[:-1])
    centers = edges[:-1] +widths/2

    evfac = 1e-3

    handles = []
    labels = []

    handles.append(Rectangle((0,0), 0, 0, facecolor="white", edgecolor="white", alpha=0.))
    labels.append("")

    h = mplhep.histplot(yields_bg*evfac, edges, edges=False, color="black", linestyle="dashed", ax=ax1)
    handles.append(h[0][0])
    labels.append("BG only")
    ax2.hlines(1,-1,1, color="black", linestyle="dashed")

    for s, ys in yields_signals.items():
        h = mplhep.histplot((yields_bg+ys)*evfac, edges, edges=False, color=colors_signals[s], ax=ax1)
        handles.append(h[0][0])
        labels.append("BG + " + labels_signals[s])
        mplhep.histplot(ys/yields_bg+1, edges, edges=False, color=colors_signals[s], ax=ax2)

    h = mplhep.histplot(yields_bg*evfac, edges, yerr=errs_bg*evfac, edges=False, histtype="band", ax=ax1, zorder=-90, facecolor="black", alpha=0.3, linewidth=0., hatch=None)
    handles.append(h[0][0])
    labels.append("Prefit uncertainty" if fittype == "prefit" else "Postfit uncertainty")
    mplhep.histplot(np.ones_like(yields_bg), edges, yerr=errs_bg/yields_bg, edges=False, histtype="band", ax=ax2, zorder=-90, facecolor="black", alpha=0.3, linewidth=0., hatch=None)

    h= mplhep.histplot(yields_data*evfac, edges, yerr=np.sqrt(yields_data)*evfac, edges=False, histtype="errorbar", color="black", ax=ax1)
    handles.append(h[0])
    labels.append("Data")
    mplhep.histplot(yields_data/yields_bg, edges, yerr=np.sqrt(yields_data)/yields_bg, edges=False, histtype="errorbar", color="black", ax=ax2)

    legendtitle = ("Prefit" if fittype=="prefit" else "Postfit")
    legendtitle += ", $\\ell\\ell$, "
    if args.mtt_min == 320:
        legendtitle += "$m_{t \\bar{t}} < "+ str(args.mtt_max) + "$ GeV"
    else:
        legendtitle += "$" + str(args.mtt_min) + " < m_{t \\bar{t}} < "+ str(args.mtt_max) + "$ GeV"
    ax1.annotate(legendtitle, (0.04, 0.96), xycoords="axes fraction", va="top", fontsize=20)

    ax1.legend(handles=handles, labels=labels, frameon=False, borderpad=0.5, labelspacing=0.3, handletextpad=0.4, fontsize=18,)# title=legendtitle, title_fontsize=18)
    #ax1.set_title("reconstructed $m_{t \\bar{t}} < 360$ GeV")
    #ax1.set_title("$m_{t \\bar{t}} < 360$ GeV, prefit")

    ax1.set_xlabel("")
    varlabel = "reconstructed $c_{\\mathrm{hel}}$" if whichvar == "chel" else "reconstructed $c_{\\mathrm{han}}$"
    ax2.set_xlabel(varlabel)
    ax1.set_ylabel("<Events> / $10^3$")
    ax2.set_ylabel("Ratio to background")
    ax1.set_ylim(np.amin(yields_data)*0.9*evfac, None)
    ax2.set_xlim(-1,1)
    if fittype == "prefit":
        ax2.set_ylim(0.79, 1.21)
        ax2.set_yticks([0.8, 1.0, 1.2])
    else:
        ax2.set_ylim(0.895, 1.105)
        ax2.set_yticks([0.9, 1.0, 1.1])
    #ax1.annotate("Prefit" if fittype=="prefit" else "Postfit", (0.03, 0.97), xycoords="axes fraction", va="top", fontweight="bold", fontsize=20)
    mplhep.cms.label(ax=ax1, data=True, lumi=138, label="Preliminary")
    plt.subplots_adjust(hspace=0.03, right=0.95, top=0.94, bottom=0.1)
    plt.savefig(outfile)
    plt.close()



parser = argparse.ArgumentParser()
parser.add_argument("infile", help="FitDiagnostics output root file")
parser.add_argument("--which", choices=('chel', 'chan'), default='chel')
parser.add_argument("--outdir", "-o", default=".")
parser.add_argument("--ext", choices=('pdf', 'svg', 'png'), default="pdf")
parser.add_argument("--mtt_min", type=int, default=320)
parser.add_argument("--mtt_max", type=int, default=1700)
parser.add_argument("--signals", type=str, default="EtaT")
parser.add_argument("--mass", "-m", default=400, type=int, help="A/H mass point")
parser.add_argument("--width", "-w", default=2.0, type=float, help="A/H relative width point")
parser.add_argument("--gA", "-gA", default=1.0, type=float, help="A/H coupling modifier")
parser.add_argument("--gH", "-gH", default=1.0, type=float, help="A/H coupling modifier")
args = parser.parse_args()

signals_to_plot = args.signals.split(",")

bgs = ['TT', 'DY', 'VV', 'TTV', 'TB', 'TQ', 'TW']
bgs.extend(f"EWK_TT_{part}_{sign}" for part in ["const", "lin", "quad"] for sign in ["pos", "neg"])
sigEtaT = 'EtaT'
sigA = f'A_m{args.mass}_w{args.width:.1f}'.replace('.', 'p')
sigH = f'H_m{args.mass}_w{args.width:.1f}'.replace('.', 'p')

channels = ["ee", "em", "mm"]
years = ["2016pre", "2016post", "2017", "2018"]

for fittype in ["prefit", "postfit"]:

    yield_bg = np.zeros(180)
    #var_bg = np.zeros(180)
    yield_EtaT = np.zeros(180)
    yield_A = np.zeros(180)
    yield_H = np.zeros(180)
    yield_data = np.zeros(180)

    muetat = 1.


    shapes = "shapes_prefit" if fittype == "prefit" else \
        ("shapes_fit_b" if "result_b" in args.infile else "shapes_fit_s")
    with uproot.open(args.infile) as f:
        total_covar = f[f"{shapes}/overall_total_covar"].values()

        if fittype == "postfit":
            ftree = f["tree_fit_sb"]
            if "EtaT" in signals_to_plot:
                muetat = ftree["CMS_EtaT_norm_13TeV"].array()[0]
            if "A" in signals_to_plot:
                gAbest = ftree["g1"].array()[0] if "g1" in ftree else 0.
            if "H" in signals_to_plot:
                gHbest = ftree["g2"].array()[0] if "g2" in ftree else 0.

        for c in channels:
            for y in years:
                for p in bgs:
                    v_bg = f[f"{shapes}/{c}_{y}/{p}"].values()
                    yield_bg += v_bg
                    #var_bg += f[f"{shapes}/{c}_{y}/{p}"].variances()
                if "EtaT" in signals_to_plot:
                    yield_EtaT += f[f"{shapes}/{c}_{y}/{sigEtaT}"].values()
                if "A" in signals_to_plot:
                    yield_A += f[f"{shapes}/{c}_{y}/{sigA}_res"].values()
                    yield_A += f[f"{shapes}/{c}_{y}/{sigA}_pos"].values()
                    yield_A += f[f"{shapes}/{c}_{y}/{sigA}_neg"].values()
                if "H" in signals_to_plot:
                    yield_H += f[f"{shapes}/{c}_{y}/{sigH}_res"].values()
                    yield_H += f[f"{shapes}/{c}_{y}/{sigH}_pos"].values()
                    yield_H += f[f"{shapes}/{c}_{y}/{sigH}_neg"].values()
                
                yield_data += f[f"{shapes}/{c}_{y}/data"].values()[1] 

    sum_axis = 2 if args.which == 'chel' else 1

    mtt_binning = [320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 845,
                    890, 935, 985, 1050, 1140, 1300, 1700]
    imttmin = mtt_binning.index(args.mtt_min)
    imttmax = mtt_binning.index(args.mtt_max)

    yield_bg_sum = yield_bg.reshape(3,3,20).T.sum(axis=sum_axis)
    yield_bg_sl = yield_bg_sum[imttmin:imttmax].sum(axis=0)
    #var_bg_sum = var_bg.reshape(3,3,20).T.sum(axis=sum_axis)
    #var_bg_sl = var_bg_sum[imttmin:imttmax].sum(axis=0)

    if "EtaT" in signals_to_plot:
        yield_EtaT_sum = yield_EtaT.reshape(3,3,20).T.sum(axis=sum_axis)
        yield_EtaT_sl = yield_EtaT_sum[imttmin:imttmax].sum(axis=0)
    if "A" in signals_to_plot:  
        yield_A_sum = yield_A.reshape(3,3,20).T.sum(axis=sum_axis)
        yield_A_sl = yield_A_sum[imttmin:imttmax].sum(axis=0)
    if "H" in signals_to_plot:
        yield_H_sum = yield_H.reshape(3,3,20).T.sum(axis=sum_axis)
        yield_H_sl = yield_H_sum[imttmin:imttmax].sum(axis=0)
    yield_data_sum = yield_data.reshape(3,3,20).T.sum(axis=sum_axis)
    yield_data_sl = yield_data_sum[imttmin:imttmax].sum(axis=0)

    err_bg_sl = unc_from_covar(total_covar, imttmin, imttmax, which=args.which)

    if fittype == "prefit" and ("A" in signals_to_plot or "H" in signals_to_plot):
        yield_A = np.zeros(180)
        yield_H = np.zeros(180)
        for sig, ysig in zip([sigA, sigH], [yield_A, yield_H]):
            infile_sig = f"{ah_signals_dir}/{sig}_lx_smtt/ahtt_input.root"
            g = args.gA if sig == sigA else args.gH
            with uproot.open(infile_sig) as f:
                for c in channels:
                    for y in years:
                        for p in ["res", "pos", "neg"]:
                            if p == "res":
                                fac = g**4
                            elif p == "pos":
                                fac = g**2
                            else:
                                fac = -g**2
                            ysig += fac * f[f"{c}_{y}/{sig}_{p}"].values()
        yield_A_sum = yield_A.reshape(3,3,20).T.sum(axis=sum_axis)
        yield_H_sum = yield_H.reshape(3,3,20).T.sum(axis=sum_axis)
        yield_A_sl = yield_A_sum[imttmin:imttmax].sum(axis=0)
        yield_H_sl = yield_H_sum[imttmin:imttmax].sum(axis=0)

    outfile = os.path.join(args.outdir, f"{args.which}_mtt_{args.mtt_min}_to_{args.mtt_max}_{fittype}.{args.ext}")

    signals = {}
    labels_signals = {}

    if "EtaT" in signals_to_plot:
        signals['EtaT'] = yield_EtaT_sl
        labels_signals['EtaT'] = f"$\\eta_t$, $\\mu(\\eta_t) = {'1' if fittype == 'prefit' else f'{muetat:.2f}'}$"
    if "A" in signals_to_plot:
        signals['A'] = yield_A_sl
        labels_signals['A'] = f"A({args.mass}, {args.width:.0f}%), $g_A = {(args.gA if fittype == 'prefit' else gAbest):.2f}$"
    if "H" in signals_to_plot:
        signals['H'] = yield_H_sl
        labels_signals['H'] = f"H({args.mass}, {args.width:.0f}%), $g_H = {(args.gH if fittype == 'prefit' else gHbest):.1f}$"

    plot(
        yields_bg = yield_bg_sl,
        errs_bg = err_bg_sl,
        yields_data = yield_data_sl,
        yields_signals = signals,
        labels_signals = labels_signals,
        colors_signals = {
            'EtaT': 'forestgreen',
            'A': "#cc0033",
            'H': "#0033cc"
        },
        fittype=fittype,
        whichvar=args.which,
        outfile=outfile
    )
