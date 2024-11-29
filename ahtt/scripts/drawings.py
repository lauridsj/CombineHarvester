#!/usr/bin/env python3
# utilities containing functions to be imported - plotting version

from utilspy import pmtofloat, coinflip
from desalinator import tokenize_to_list, remove_spaces_quotes

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
    "yukawa":     r"$y_{\mathrm{t}}$",
    "ll":         r"$\ell\bar{\ell}$",
    "l3j":        r"$\ell$, 3j",
    "l4pj":       r"$\ell$, $\geq$ 4j",
    "lj":         r"$\ell$j",
    "lx":         r"$\ell\bar{\ell}$, $\ell$j"
}

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

def stock_labels(parameters, points, resxsecpb = 5):
    labels = []
    for ii, pp in enumerate(parameters):
        if pp in ["g1", "g2"]:
            labels.append(axes["coupling"] % str_point(points[ii]))
        elif pp in ["r1", "r2"]:
            labels.append(axes["muah"] % (str_point(points[ii], spinstate = True), str(resxsecpb)))
        elif pp == "CMS_EtaT_norm_13TeV":
            labels.append(axes["muetat"])
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
