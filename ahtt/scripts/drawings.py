#!/usr/bin/env python3
# utilities containing functions to be imported - plotting version

import ROOT

from utilspy import pmtofloat, coinflip
from desalinator import tokenize_to_list, remove_spaces_quotes
from scipy import interpolate

min_g = 0.
max_g = 3.
epsilon = 2.**-17
axes = {
    "mass" :      r"$\mathrm{m}_{\mathrm{\mathsf{%s}}}$ [GeV]",
    "width":      r"$\Gamma_{\mathrm{\mathsf{%s}}}$ [%% m$_{\mathrm{\mathsf{%s}}}$]",
    "coupling":   r"$\mathrm{g}_{\mathrm{\mathsf{%s}}}$",
    "ttcoupling": r"$\mathrm{g}_{\mathrm{\mathsf{%s}t\bar{t}}}$",
    "dnll":       r"$-2\,\ln\,\dfrac{\mathcal{L}(g_{\mathrm{\mathsf{%s}}})}{\mathcal{L}_{\mathrm{SM}}}$",
    #"muah":       r"$\mu^{\mathrm{\mathsf{%s}}}_{\mathrm{%s}}$",
    "muah":       r"$\sigma\left(\mathrm{\mathsf{%s}}\right)$ [%s pb]",
    "muetat":     r"$\mu(\eta_{\mathrm{t}})$",
    "muchit":     r"$\mu(\chi_{\mathrm{t}})$",
    "sigmaetat":  r"$\sigma\left(\eta_{\mathrm{t}}\right)$ [pb]",
    "sigmachit":  r"$\sigma\left(\chi_{\mathrm{t}}\right)$ [pb]",
    "yukawa":     r"$y_{\mathrm{t}}$",
    "ll":         r"$\ell\ell$",
    "l3j":        r"$\ell$, 3j",
    "l4pj":       r"$\ell$, $\geq$ 4j",
    "lj":         r"$\ell$j",
    "lx":         r"$\ell\ell$, $\ell$j"
}
channels = ["ee", "em", "mm", "e4pj", "m4pj", "e3j", "m3j"]
years = ["2016pre", "2016post", "2017", "2018"]
sm_procs = {
    "TTV": r"Other",
    "VV": r"Other",
    "EWQCD": r"Other",
    "TW": "tX",
    "TB": "tX",
    "TQ": "tX",
    "DY": r"Other",
    "EtaT": r"$\eta_{\mathrm{t}}$",
    "ChiT": r"$\chi_{\mathrm{t}}$",
    "PsiT": r"$\psi_{\mathrm{t}}$",
    "TT": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_const_pos": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_const_neg": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_lin_pos": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_lin_neg": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_quad_pos": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "EWK_TT_quad_neg": r"$\mathrm{t}\bar{\mathrm{t}}$",
}
proc_colors = {
    "tX": "C0", #"#5790fc",
    r"Other": "C1", #"#f89c20",
    r"$\mathrm{t}\bar{\mathrm{t}}$":  "#F3E5AB", #"#e42536",
    #r"EW + QCD": "#58A279", #"#964a8b",
    r"$\eta_{\mathrm{t}}$": "#cc0033",
    r"$\chi_{\mathrm{t}}$": "#0033cc",
    r"$\psi_{\mathrm{t}}$": "#009000",

    "A": "#cc0033",
    "H": "#0033cc",
    "Total": "#3B444B"
}
signal_zorder = {
    r"$\eta_{\mathrm{t}}$": 1,
    r"$\chi_{\mathrm{t}}$": 2,
    r"$\psi_{\mathrm{t}}$": 3,
    "Total": 0,
    "A": 1,
    "H": 2
}
binnings = {
    ("ee", "em", "mm"): {
        r"$m_{\mathrm{t}\bar{\mathrm{t}}}$ (GeV)":
            [320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 845, 890, 935, 985, 1050, 1140, 1300, 1460],
        r"$m_{\mathrm{b}\mathrm{b}\ell\ell}$ (GeV)":
            [0, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 530, 560, 600, 660, 750, 1000],
        r"$c_{\mathrm{hel}}$": ["-1", r"-$\frac{1}{3}$", r"$\frac{1}{3}$", "1"],
        r"$c_{\mathrm{han}}$": ["-1", r"-$\frac{1}{3}$", r"$\frac{1}{3}$", "1"],
    },
    ("e4pj", "m4pj", "e3j", "m3j"): {
        r"$m_{\mathrm{t}\bar{\mathrm{t}}}$ (GeV)":
            [320, 360, 400, 440, 480, 520, 560, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100,  1150, 1200, 1300, 1500, 1700],
        r"$\left|\cos(\theta_{\mathrm{t}_{\ell}}^{*})\right|$": [0.0, 0.4, 0.6, 0.75, 0.9, 1.0],
    }
}
ratiolabels = {
    "b": "Ratio to background",
    "s": "Ratio to background",
    "p": "Ratio to background",
}
lumis = {
    "2016pre": "19.5",
    "2016post": "16.8",
    "2017": "41.5",
    "2018": "59.9",
    "Run 2": "138",
}
hatchstyle = dict(
    color = "black",
    alpha = 0.3,
    linewidth = 0,
)
datastyle = dict(
    marker = "o",
    markersize = 3,
    elinewidth = 0.75,
    linestyle = "none",
    color = "black"
)

def ith(iterable, idx):
    return [item[idx] for item in iterable]

first = lambda iterable: ith(iterable, 0)
second = lambda iterable: ith(iterable, 1)
third = lambda iterable: ith(iterable, 2)

def pruned(iterable, keep = None, dropout = 0.02):
    result = []
    for item in iterable:
        if (keep is not None and (item == keep or item in keep)) or coinflip(1. - dropout):
            result.append(item)
    return result

def withinerror(v0, v1, epsilon = 2.**-3):
    relativeok = abs((v0 - v1) / v0) < epsilon if v0 != 0 else True
    absoluteok = abs(v0 - v1) < epsilon
    return relativeok or absoluteok

def get_point(sigpnt):
    pnt = sigpnt.split('_')
    return (pnt[0][0], float(pnt[1][1:]), float(pnt[2][1:].replace('p', '.')))

def str_point(sigpnt, spinstate = False):
    term = {
        'A': '\mathrm{{}^{1} S_{0}}',
        'H': '\mathrm{{}^{3} P_{0}}'
    }
    pnt = sigpnt.split('_')
    parity = term[pnt[0][0]] if spinstate else pnt[0][0]
    return parity + '(' + pnt[1][1:] + ',\, ' + pnt[2][1:].replace('p0', '').replace('p', '.') + ' \%)'

def default_etat_measurement(arg = ""):
    result = tokenize_to_list(remove_spaces_quotes(arg), astype = float)
    defaults = [0, 6.43, 0.64, 0.64]
    while len(result) < len(defaults):
        result.append(defaults[len(result)])
    return result

def etat_blurb(cfg):
    if cfg[0]:
        blurb = [
            r"$\mathbf{Including~{}^{1} S_{0}^{[1]}~t\bar{t}~bound~state~\eta_{\mathrm{t}}}$",
            r"PRD 104, 034023 ($\mathbf{2021}$)"
        ]
        # disabled because adding the profiled number depends on signal point
        #blurb += [r"Best fit $\sigma_{\eta_{\mathrm{t}}}$: $" + "{val}".format(val = cfg[1])]
        #if len(cfg) > 3:
        #    blurb[-1] += r"_{-" + "{ulo}".format(ulo = cfg[2]) + r"}^{+" + "{uhi}".format(uhi = cfg[3])
        #else:
        #    blurb[-1] += r" \pm " + "{unc}".format(unc = cfg[2])
        #blurb[-1] += r"$ pb ($\mathrm{g}_{\mathrm{\mathsf{A/H}}} = 0$)"
    else:
        blurb = [
            r"$\mathbf{No~t\bar{t}~bound~states}$",
            #r"PRD 104, 034023 ($\mathbf{2021}$)"
        ]
    return blurb

def stock_labels(parameters, points, resxsecpb = 5, toxsec = False):
    labels = []
    for ii, pp in enumerate(parameters):
        if pp in ["g1", "g2"]:
            labels.append(axes["coupling"] % str_point(points[ii]))
        elif pp in ["r1", "r2"]:
            labels.append(axes["muah"] % (str_point(points[ii], spinstate = True), str(resxsecpb)))
        elif pp == "CMS_EtaT_norm_13TeV":
            labels.append(axes["sigmaetat"] if toxsec else axes["muetat"])
        elif pp == "CMS_ChiT_norm_13TeV":
            labels.append(axes["sigmachit"] if toxsec else axes["muchit"])
        elif pp == "EWK_yukawa":
            labels.append(axes["yukawa"])
        else:
            labels.append(pp)
    return labels

def valid_nll_fname(fname, tag, ninterval = 1):
    fname = fname.split('/')[-1].replace(tag, "").split('_')
    nvalidto = 0
    for part in fname:
        if "to" in part:
            minmax = part.split('to')
            for mm in minmax:
                try:
                    pmtofloat(mm)
                    break
                except ValueError:
                    return False
            nvalidto += 1
    return nvalidto == ninterval

def get_poi_values(fname, signals, tname):
    signals = list(signals.keys())
    signals = [sig for sig in signals if sig != ("Total", None, None)]
    twing = len(signals) == 2 and sum([1 if sig[0] == "A" or sig[0] == "H" else 0 for sig in signals]) == 2
    oneg = len(signals) == 1 and sum([1 if sig[0] == "A" or sig[0] == "H" else 0 for sig in signals]) == 1
    onepoi = "one-poi" in fname
    etat = (r"$\eta_{\mathrm{t}}$", r"$\eta_{\mathrm{t}}$" in [sig[0] for sig in signals])
    chit = (r"$\chi_{\mathrm{t}}$", r"$\chi_{\mathrm{t}}$" in [sig[0] for sig in signals])
    psit = (r"$\psi_{\mathrm{t}}$", r"$\psi_{\mathrm{t}}$" in [sig[0] for sig in signals])

    if not twing and not oneg and not onepoi and not etat:
        raise NotImplementedError()

    tfile = None
    tres = None
    if tname != "":
        if tname == "default":
            tname = fname.replace("fitdiagnostics_result", "single_obs")
            if "_fixed_" in tname:
                pois = tname.split("_fixed_")[0].split("_g1_")[0].split("_obs_")[1]
                pois = pois.replace(".root", "")
                tname = tname.replace("_" + pois, "").replace("_s.root", f"_{pois}.root")
        tfile = ROOT.TFile.Open(tname, "read")
        tres = tfile.Get("limit")

    ffile = ROOT.TFile.Open(fname, "read")
    if "fit_s" not in ffile.GetListOfKeys():
        return {sig: 0 for sig in signals}

    fres = ffile.Get("fit_s")
    result = {}

    if onepoi:
        if fres and tres is None:
            gg = fres.floatParsFinal().find('g')
            result = {signals[0]: (round(gg.getValV(), 3), round(gg.getError(), 3))}
        elif tres is not None:
            values = [0., 0., 0.]
            for i in tres:
                qq = 0 if tres.quantileExpected == -1 else 1 if tres.quantileExpected > 0 else 2
                values[qq] = tres.g
            result = {signals[0]: (round(values[0], 3), round(abs(values[1] - values[0]), 3), round(abs(values[2] - values[0]), 3))}
        else:
            result = {signals[0]: (0., 0.)}
    elif twing:
        if fres and tres is None:
            g1 = fres.floatParsFinal().find('r1' if args.onlyres else 'g1')
            g2 = fres.floatParsFinal().find('r2' if args.onlyres else 'g2')
            result[signals[0]] = (round(g1.getValV(), 2), round(g1.getError(), 2))
            result[signals[1]] = (round(g2.getValV(), 2), round(g2.getError(), 2))
        elif tres is not None:
            values = [[0., 0., 0.], [0., 0., 0.]]
            for i in tres:
                qq = 0 if tres.quantileExpected == -1 else 1 if tres.quantileExpected > 0 else 2
                if qq != 0:
                    tmp = tres.g1 if args.onlyres else tres.r1
                    if tmp != values[0][0]:
                        values[0][qq] = tmp
                    tmp = tres.g2 if args.onlyres else tres.r2
                    if tmp != values[1][0]:
                        values[1][qq] = tmp
                else:
                    values[0][qq] = tres.g1 if args.onlyres else tres.r1
                    values[1][qq] = tres.g2 if args.onlyres else tres.r2
            result[signals[0]] = (round(values[0][0], 3), round(abs(values[0][1] - values[0][0]), 3), round(abs(values[0][2] - values[0][0]), 3))
            result[signals[1]] = (round(values[1][0], 3), round(abs(values[1][1] - values[1][0]), 3), round(abs(values[1][2] - values[1][0]), 3))
        else:
            result = {signals[0]: (0., 0.), signals[1]: (0., 0.)}
    elif oneg:
        if fres and tres is None:
            allparams = [p.GetName() for p in fres.floatParsFinal()]
            if signals[0][0] == "A" and "g1" in allparams:
                parname = "g1"
            elif signals[0][0] == "H" and "g2" in allparams:
                parname = "g2"
            elif "g" in allparams:
                parname = "g"
            else:
                raise ValueError()
            gg = fres.floatParsFinal().find(parname)
            result = {signals[0]: (round(gg.getValV(), 3), round(gg.getError(), 3))}
        elif tres is not None:
            raise NotImplementedError()
        else:
            result = {signals[0]: (0., 0.)}

    for flag, name in [(etat, 'CMS_EtaT_norm_13TeV'), (chit, 'CMS_ChiT_norm_13TeV'), (psit, 'CMS_PsiT_norm_13TeV')]:
        if flag[1]:
            if fres and tres is None:
                muhat = fres.floatParsFinal().find(name)
                result = result | {(flag[0], None, None): (round(muhat.getValV(), 3), round(muhat.getError(), 3))}
            elif tres is not None:
                values = [0., 0., 0.]
                for i in tres:
                    qq = 0 if tres.quantileExpected == -1 else 1 if tres.quantileExpected > 0 else 2
                    tmp = getattr(tres, name)
                    if qq == 0 or (qq != 0 and tmp != values[0]):
                        values[qq] = tmp
                result = result | {(flag[0], None, None): (round(values[0], 3), round(abs(values[1] - values[0]), 3), round(abs(values[2] - values[0]), 3))}
            else:
                result = result | {(flag[0], None, None): (0., 0.)}
    result = {k: v if len(v) < 3 or abs(v[1] - v[2]) / v[1] > 0.1 else (v[0], v[1]) for k, v in result.items()}
    return result

def get_model_at_minimum(fname):
    '''
    fname is the filename of the *fitdiagnostics_result* file in some other setting, that is passed as --ifile to plot_prepost*.py scripts
    '''
    ffile = ROOT.TFile.Open(fname, "read")
    if "fit_s" not in ffile.GetListOfKeys() and "fit_b" not in ffile.GetListOfKeys():
        return {}
    fres = ffile.Get("fit_s")
    fcov = ffile.Get("covariance_fit_s")
    if fres is None:
        fres = ffile.Get("fit_b")
        fcov = ffile.Get("covariance_fit_b")

    values = {pp.GetName(): (pp.getValV(), pp.getError()) for pp in fres.floatParsFinal()}
    values = {k: v for k, v in result.items() if all([abs(vv) > 1e-7 for vv in v])}
    correlation = {}
    params = list(values.keys())
    nbins = fcov.GetXaxis().GetNbins()
    for i0, n0 in enumerate(params):
        for i1 in range(i0 + 1, len(params)):
            ix = iy = -1
            for ibin in range(1, nbins + 1):
                label = fcov.GetXaxis().GetBinLabel(ibin)
                if label == n0:
                    ix = ibin
                elif label == params[i1]:
                    iy == ibin
                if ix != -1 and iy != -1:
                    break
            correlation[ (n0, params[i1]) ] = cov[ix][iy] / math.sqrt(cov[ix][ix] * cov[iy][iy])
    return {"value": values, "correlation": correlations}

def combine_shape_impl(value, dup, ddo):
    ds = dup + ddo
    dd = dup - ddo
    t0 = dd * value
    t1 = ds / 8.
    t2 = (3. * value**6.) - (10. * value**4.) + (15 * value**2.)
    return (t0 + (t1 * t2)) / 2.

def combine_shape_interpolation(value, nominal, up, down):
    # implements 'shape' interpolation ala https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part2/settinguptheanalysis/#binned-shape-analyses
    # value is the desired np value to interpolate to
    # remaining args are the template shapes, assumed to be of the form (norm, binwise fractions), passed to combine
    result = (0., np.zeros(len(nominal[1])))
    inorm = interpolate.interp1d([-1, 0, 1], [math.log(down[0]), math.log(nominal[0]), math.log(up[0])], fill_value = "extrapolate")
    result[0] = max(float(inorm(value)), 0.)
    for ii in range(len(nominal[1])):
        if value > 1 or value < -1:
            delta = up[1][ii] - nominal[1][ii] if value > 1 else down[1][ii] - nominal[1][ii]
            result[1][ii] = delta * value
        else:
            du = up[1][ii] - nominal[1][ii]
            dd = down[1][ii] - nominal[1][ii]
            result[1][ii] = max(combine_shape_impl(value, du, dd), 0.)
    return result

def apply_model(templates, model, prefit):
    # where templates is the dict containing postfit templates, for all channels and years before summing
    # model is the output of get_model_at_minimum()
    # and prefit is the file name with prefit templates ie ahtt_input.root
    model = get_model_at_minimum(model)
    #hnew = hist.Hist(*[ax.copy() for ax in h.axes], storage="Weight")
    ccyy = f"{templates[channel]}_{templates[year]}"
    shapes = {}
    with uproot.open(prefit) as ff:
        procs = list(sm_procs.keys())
        ahtt = [f"{ss[0]}_m{ss[1]}_w{str(ss[2]).replace('.', 'p')}" for ss in signals if ss[0] != "Total"]
        procs += [f"{ah}_{pp}" for ah in ahtt for pp in ["res", "pos", "neg"]]

        tmp = {proc: ff["{ccyy}"][proc] for proc in procs}
        tmp = {k: v for k, v in tmp.items() if v is not None}
        tmp = {k: (sum(v.values()), v / sum(v.values())) for k, v in tmp.items()}
        shapes["nominal"] = tmp

        for param in model["value"].keys():
            shapes[param] = {}
            for dd in zip(["u", "d"], ["Up", "Down"]):
                tmp = {proc: ff["{ccyy}"][f"{proc}_{param}{dd[1]}"] for proc in procs}
                tmp = {k: v for k, v in tmp.items() if v is not None}
                tmp = {k: (sum(v.values()), v / sum(v.values())) for k, v in tmp.items()}
                shapes[param][dd[0]] = tmp

    # TODO actual application of the "model"
    # probably start from prefit nominal, and then shift by deltas as per NP value
    # and the variance is added based on further shifts by NP value +- unc
    # to check: ordering of application +- unc for shape (as above), lnN (overall scaling, on prefit nominal or final?), rateparam (same as lnN?)...
    pass
