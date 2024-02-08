#!/usr/bin/env python3
# utilities containing functions to be imported - plotting version

min_g = 0.
max_g = 3.
epsilon = 2.**-17
axes = {
    "mass" :    r"$\mathrm{m}_{\mathrm{\mathsf{%s}}}$ [GeV]",
    "width":    r"$\Gamma_{\mathrm{\mathsf{%s}}}$ [%% m$_{\mathrm{\mathsf{%s}}}$]",
    "coupling": r"$\mathrm{g}_{\mathrm{\mathsf{%s}}}$",
    "dnll":     r"$-2\,\ln\,\dfrac{\mathcal{L}(g_{\mathrm{\mathsf{%s}}})}{\mathcal{L}_{\mathrm{SM}}}$",
}

first  = lambda vv: [ii for ii, _ in vv]
second = lambda vv: [ii for _, ii in vv]

def get_point(sigpnt):
    pnt = sigpnt.split('_')
    return (pnt[0][0], float(pnt[1][1:]), float(pnt[2][1:].replace('p', '.')))

def str_point(sigpnt):
    pnt = sigpnt.split('_')
    return pnt[0][0] + '(' + pnt[1][1:] + ',\, ' + pnt[2][1:].replace('p0', '').replace('p', '.') + ' \%)'
