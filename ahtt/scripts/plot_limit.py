#!/usr/bin/env python3
# draw the model-independent limit on gAH plot
# requires matplotlib > 3.3 e.g. source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_100 x86_64-centos7-gcc10-opt

from argparse import ArgumentParser
import os
import sys
import numpy as np
from scipy.interpolate import UnivariateSpline
import math

import glob
from collections import OrderedDict
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpt
import matplotlib.lines as mln
import matplotlib.colors as mcl
import matplotlib.ticker as mtc

from drawings import min_g, max_g, epsilon, axes, first, second, get_point
from drawings import default_etat_measurement, etat_blurb
from desalinator import prepend_if_not_empty, append_if_not_empty, tokenize_to_list, remove_spaces_quotes

def ahtt_width_coupling_helper(parity, mah):
    sqrt2 = math.sqrt(2)
    pi = math.pi
    sqrtpi = math.sqrt(pi)

    aew = 1. / 132.50698
    mZ = 91.188
    Gf = 1.16639e-5
    mt = 172.5

    mah2 = mah * mah
    mt2 = mt * mt
    mZ2 = mZ *mZ
    mZ4 = mZ2 * mZ2

    term1 = (aew * pi * mZ2) / (Gf * sqrt2)
    mW = math.sqrt((0.5 * mZ2) + math.sqrt((0.25 * mZ4) - term1))
    mW2 = mW * mW

    sw = math.sqrt(1. - (mW2 / mZ2))
    ee = 2. * math.sqrt(aew) * sqrtpi
    vev = 2. * mW * sw / ee
    vev2 = vev * vev
    term2 = 8. * pi * vev2

    term3 = 0.75 * Gf / (pi * sqrt2)
    
    if mah >= 2 * mt:
        beta = math.sqrt(1. - (4. * mt2 / mah2))
    else:
        beta = 0.
    factor = mah * mt2 * term3 * beta
    return factor if parity == 'A' else factor * beta * beta if parity == 'H' else 0.

def ahtt_partial_width(parity, mah, gah, relwidth = True):
    wah = gah * gah * ahtt_width_coupling_helper(parity, mah)
    return wah / mah if relwidth else wah

def ahtt_max_coupling(parity, mah, wah, relwidth = True):
    wah = wah * mah if relwidth else wah
    return math.sqrt(wah / ahtt_width_coupling_helper(parity, mah))

def cleanup_multi_crossing(g_and_cls):
    ### this block here is some attempt to smooth out the raw cls numbers from combine
    ### to deal with unphysical jumps at close by g values
    ### scenario with possibly multiple crossing (used for observed)
    result = []
    cutoffs = []
    cutoff = []

    for gg, cc in g_and_cls:
        if 0.01 < cc < 0.2:
            cutoff.append((gg, cc))
        else:
            if cutoff != []:
                cutoffs.append(cutoff)
                cutoff = []

    ## merge intervals that are nearby
    minmax = [(first(cutoff)[0], first(cutoff)[-1]) for cutoff in cutoffs if len(cutoff) != 1]
    imm = 0
    while imm < len(minmax) - 1:
        if minmax[imm + 1][0] - minmax[imm][1] < 0.1:
            cutoffs = cutoffs[:imm] + [cutoffs[imm] + cutoffs[imm + 1]] + cutoffs[imm + 2:]
            minmax = [(first(cutoff)[0], first(cutoff)[-1]) for cutoff in cutoffs]
            imm = 0
        else:
            imm += 1

    ## ensure the intervals are monotonous
    for icut in range(len(cutoffs)):
        clamp = (cutoffs[icut][0][1], cutoffs[icut][-1][1])
        which = 1 if clamp[-1] > clamp[0] else -1
        mono = [cutoffs[icut][0]]
        for cut in cutoffs[icut][1:]:
            ismono= second(mono) + [cut[1]]
            upward = which > 0 and ismono == sorted(ismono)
            downward = which < 0 and ismono == list(reversed(sorted(ismono)))
            if upward or downward:
                mono.append(cut)
            cutoffs[icut] = mono

    ## build splines to interpolate
    for cutoff in cutoffs:
        if len(cutoff) < 5:
            continue
        spline = UnivariateSpline(np.array(first(cutoff)), np.array(second(cutoff)), s = len(cutoff))
        minmax = (first(cutoff)[0], first(cutoff)[-1])
        for idx in range(len(g_and_cls)):
            if minmax[0] <= g_and_cls[idx][0] <= minmax[1]:
                result.append((g_and_cls[idx][0], float(spline(g_and_cls[idx][0]))))
            else:
                result.append((g_and_cls[idx][0], g_and_cls[idx][1]))
    ### end of block
    return result

def cleanup_single_crossing(g_and_cls):
    ### this block here is some attempt to smooth out the raw cls numbers from combine
    ### to deal with unphysical jumps at close by g values
    ### scenario with possibly multiple crossing (used for expected)
    result = []
    g = [gg for gg, cc in g_and_cls]
    cls = [cc for gg, cc in g_and_cls]

    vmin = min([(abs(cc - 0.05), gg, cc) for gg, cc in g_and_cls])
    vmin = (vmin[1], vmin[2])
    imin = g_and_cls.index(vmin)

    if imin > 0 and len(cls) - imin > 0:
        cmin = 5e-3
        cmax = 0.5

        left = [cc for cc in cls if cc < 0.05]
        right = [cc for cc in cls if cc > 0.05]

        left = sum(left) / len(left) if len(left) > 0 else -1.
        right = sum(right) / len(right) if len(right) > 0 else -1.

        g = []
        cls = []

        condition = lambda x, y, le_if_true_else_ge: x < y if le_if_true_else_ge else x > y
        if left >= 0. and right >= 0.:
            for ii in range(1, len(g_and_cls)):
                gg, cc = g_and_cls[ii]
                if cmin < cc < cmax:
                    if (len(g) == 0 and len(cls) == 0) or condition(cc, cls[-1], vmin[1] < cls[0]):
                        g.append(gg)
                        cls.append(cc)
                    elif len(g) > 1 and len(cls) > 1 and ii < len(g_and_cls) - 1:
                        cprev = condition(cc, cls[-2], vmin[1] < cls[0])
                        nprev = condition(g_and_cls[ii + 1][1], cls[-2], vmin[1] < cls[0])
                        if cprev and nprev:
                            g.pop()
                            cls.pop()
                            g.append(gg)
                            cls.append(cc)
    else:
        g = []
        cls = []

    if len(g) > 3:
        spline = UnivariateSpline(np.array(g), np.array(cls), s = len(g))

        smin = min([(abs(cc - 0.05), gg, cc) for gg, cc in zip(g, cls)])
        smin = (smin[1], smin[2])

        min_factor = 1.
        abs_tolerance = 0.0005
        need_checking = False
        crossing = smin[0]
        residual = abs(spline(crossing) - 0.05)
        factor = 2.**9 if smin[1] > 0.05 else -2.**9

        while residual > abs_tolerance and crossing < max_g and crossing > min_g:
            if need_checking or abs(factor) < min_factor:
                break

            crossing += factor * epsilon
            if abs(spline(crossing) - 0.05) > residual or crossing >= g[-1] or crossing <= g[0]:
                if crossing >= g[-1]:
                    crossing = g[-1] - (min_factor * epsilon / 2.)
                if crossing <= g[0]:
                    crossing = g[0] + (min_factor * epsilon / 2.)
                factor /= -2.

            if abs(factor) < min_factor and residual > abs_tolerance:
                need_checking = True
            residual = abs(spline(crossing) - 0.05)

        if need_checking:
            #print("in " + dcd + ", quantile " + quantile + ", achieved cls residual is " +
            #      str(residual) + " at g = " + str(crossing))
            print("achieved cls residual is " + str(residual) + " at g = " + str(crossing))
            print("g and cls values used to build the spline:")
            print(g)
            print(cls)
            print("g, cls point with minimum distance to cls = 0.05 from a raw search on sampled points: ", vmin)
            print("g, cls point with minimum distance to cls = 0.05 from points considered for the spline: ", smin)
            print("\n")

        ## FIXME to be reimplemented post refactoring
        #if dump_spline or need_checking:
        #    qstr = quantile.replace('+', 'pp').replace('-', 'm')
        #    fig, ax = plt.subplots()
        #    ax.plot(g, spline(g), 'g', lw = 3)
        #    fig.tight_layout()
        #    fig.savefig("{dcd}/{pnt}_{tag}_spline_{qua}.png".format(
        #        dcd = odir,
        #        tag = otags[tt],
        #        pnt = '_'.join(dcd.split('_')[:3]),
        #        qua = qstr
        #    ), transparent = True)
        #    fig.clf()
        result = [[crossing, max_g]] if crossing < max_g else []
    else:
        #print("in " + dcd + ", quantile " + quantile + ", following g and cls are insufficient to form a spline:")
        print("following g and cls are insufficient to form a spline:")
        print(g)
        print(cls)
        print("g, cls point with minimum distance to cls = 0.05 from a raw search on sampled point: ", vmin)
        print("\n")
        result = []
    return result

def read_limit(directories, otags, xvalues, onepoi):
    limits = [OrderedDict() for tag in directories]

    for tt, directory in enumerate(directories):
        for jj, dcd in enumerate(directory):
            pnt = '_'.join(tokenize_to_list(dcd, '/')[-1].split('_')[:3])
            limit = OrderedDict([
                ("exp-2", []),
                ("exp-1", []),
                ("exp0", []),
                ("exp+1", []),
                ("exp+2", []),
                ("obs", [])
            ])

            if onepoi:
                with open("{dcd}/{pnt}_{tag}_limits_one-poi.json".format(
                        dcd = dcd,
                        pnt = pnt,
                        tag = otags[tt])) as ff:
                    result = json.load(ff)

                    for mm, lmt in result.items():
                        for quantile, g in lmt.items():
                            limit[quantile].append([g, max_g])
            else:
                chunks = glob.glob("{dcd}/{pnt}_{tag}_limits_g-scan_n*_i*.json".format(
                    dcd = dcd,
                    pnt = pnt,
                    tag = otags[tt]))
                chunks.sort(key = lambda name: int(name.split('_')[-1].split('.')[0][1:]))
                for nn in chunks:
                    with open(nn) as ff:
                        result = json.load(ff)

                    for g, lmt in result.items():
                        # FIXME in some cases combine gives strange cls values
                        # not understood why, skip
                        cls = list(lmt.values())
                        if any([round(cc, 3) > 1. or round(cc, 3) < 0. for cc in cls]):
                            continue

                        for quantile, cls in lmt.items():
                            limit[quantile].append((float(g), round(cls, 7)))
            limits[tt][xvalues[jj]] = limit
    return limits

def draw_1D(onames, limits, labels, xaxis, yaxis, ltitle, gcurve, interpolate, drawband, observed, formal, cmsapp, luminosity, a343bkg, transparent):
    if len(limits) > 6:
        raise RuntimeError("current plotting code is not meant for more than 6 tags. aborting")

    if len(limits) > 1 and not all([list(ll) == list(limits[0]) for ll in limits]):
        raise RuntimeError("limits in the tags are not over the same x points for plot " + onames[0] + ". aborting")

    if not hasattr(draw_1D, "colors"):
        draw_1D.colors = OrderedDict([
            (1    , [{"exp2": "#f5bb54", "exp1": "#607641", "exp0": "0", "expl": "dashed", "obsf": "#385cb4", "obsl": "#0033a0", "alpe": 1., "alpo": 0.25}]),

            (2    , [{"exp2": "#ff6699", "exp1": "#ff3366", "exp0": "#cc0033", "expl": "dashed", "obsf": "#ffffff", "obsl": "#cc0033", "alpe": 0.4, "alpo": 0.},
                     {"exp2": "#6699ff", "exp1": "#3366ff", "exp0": "#0033cc", "expl": (0, (1, 1)), "obsf": "#ffffff", "obsl": "#0033cc", "alpe": 0.4, "alpo": 0.}]),

            (3    , [{"exp2": "#ffcc00", "exp1": "#00cc00", "exp0": "0", "expl": "dashed", "obsf": "#0033cc", "obsl": "#0033cc", "alpe": 1., "alpo": 0.25},
                     {"exp2": "#ff6699", "exp1": "#ff3366", "exp0": "#cc0033", "expl": "dashdot", "obsf": "#ffffff", "obsl": "#cc0033", "alpe": 0.25, "alpo": 0.},
                     {"exp2": "#6699ff", "exp1": "#3366ff", "exp0": "#0033cc", "expl": (0, (1, 1)), "obsf": "#ffffff", "obsl": "#0033cc", "alpe": 0.25, "alpo": 0.},]),

            (4    , [{"exp2": "#ffcc00", "exp1": "#00cc00", "exp0": "0", "expl": "dashed", "obsf": "#0033cc", "obsl": "#0033cc", "alpe": 1., "alpo": 0.25},
                     {"exp2": "#ff6699", "exp1": "#ff3366", "exp0": "#cc0033", "expl": "dashdot", "obsf": "#ffffff", "obsl": "#cc0033", "alpe": 0.25, "alpo": 0.},
                     {"exp2": "#6699ff", "exp1": "#3366ff", "exp0": "#0033cc", "expl": (0, (3, 5, 1, 5)), "obsf": "#ffffff", "obsl": "#0033cc", "alpe": 0.25, "alpo": 0.},
                     {"exp2": "#99ff66", "exp1": "#66ff33", "exp0": "#33cc00", "expl": (0, (1, 1)), "obsf": "#ffffff", "obsl": "#33cc00", "alpe": 0.25, "alpo": 0.},]),

            (5    , [{"exp2": "#ffcc00", "exp1": "#00cc00", "exp0": "0", "expl": "solid", "obsf": "#0033cc", "obsl": "#0033cc", "alpe": 1., "alpo": 0.25},
                     {"exp2": "#ff6699", "exp1": "#ff3366", "exp0": "#cc0033", "expl": "dashed", "obsf": "#ffffff", "obsl": "#cc0033", "alpe": 0.25, "alpo": 0.},
                     {"exp2": "#6699ff", "exp1": "#3366ff", "exp0": "#0033cc", "expl": "dashdot", "obsf": "#ffffff", "obsl": "#0033cc", "alpe": 0.25, "alpo": 0.},
                     {"exp2": "#99ff66", "exp1": "#66ff33", "exp0": "#33cc00", "expl": (0, (3, 5, 1, 5)), "obsf": "#ffffff", "obsl": "#33cc00", "alpe": 0.25, "alpo": 0.},
                     {"exp2": "#ffcc33", "exp1": "#cc99ff", "exp0": "#9966cc", "expl": (0, (1, 1)), "obsf": "#ffffff", "obsl": "#cc0033", "alpe": 0.25, "alpo": 0.},]),

            (6    , [{"exp2": "#ffcc00", "exp1": "#00cc00", "exp0": "0", "expl": "solid", "obsf": "#0033cc", "obsl": "#0033cc", "alpe": 1., "alpo": 0.25},
                     {"exp2": "#ff6699", "exp1": "#ff3366", "exp0": "#cc0033", "expl": "dashed", "obsf": "#ffffff", "obsl": "#cc0033", "alpe": 0.25, "alpo": 0.},
                     {"exp2": "#6699ff", "exp1": "#3366ff", "exp0": "#0033cc", "expl": (0, (1, 1)), "obsf": "#ffffff", "obsl": "#0033cc", "alpe": 0.25, "alpo": 0.},
                     {"exp2": "#99ff66", "exp1": "#66ff33", "exp0": "#33cc00", "expl": (0, (3, 5, 1, 5)), "obsf": "#ffffff", "obsl": "#33cc00", "alpe": 0.25, "alpo": 0.},
                     {"exp2": "#ffcc33", "exp1": "#cc99ff", "exp0": "#9966cc", "expl": "dashdot", "obsf": "#ffffff", "obsl": "#cc0033", "alpe": 0.25, "alpo": 0.},
                     {"exp2": "#ffffff", "exp1": "#ffffff", "exp0": "#555555", "expl": (0, (1, 5)), "obsf": "#ffffff", "obsl": "#cc0033", "alpe": 0.25, "alpo": 0.},]),
        ])

    yvalues = []
    xvalues = np.array(list(limits[0]))

    for tt, tag in enumerate(limits):
        limit = OrderedDict([
            ("exp-2", []),
            ("exp-1", []),
            ("exp0", []),
            ("exp+1", []),
            ("exp+2", []),
            ("obs", [])
        ])

        for xx, lmt in tag.items():
            for quantile, exclusion in lmt.items():
                if quantile == "obs":
                    cleaned = cleanup_multi_crossing(exclusion)
                    limit[quantile].append(cleaned if cleaned != [] else exclusion)
                else:
                    cleaned = cleanup_single_crossing(exclusion)
                    if len(cleaned) > 1:
                        print(quantile, cleaned)
                        raise RuntimeError("tag number " + str(tt) + ", xvalue " + str(xx) + ", quantile " + quantile + ", plot " + onames[0] +
                                       ", current plotting code is meant to handle only 1 expected exclusion interval. aborting")
                    if len(cleaned) > 0:
                        limit[quantile].append(cleaned[0][0])
                        if cleaned[0][1] != max_g:
                            print(quantile, cleaned)
                            print("tag number " + str(tt) + ", xvalue " + str(xx) + ", quantile " + quantile + ", plot " + onames[0] +
                                  " strange exclusion interval. recheck.\n")
                    else:
                        limit[quantile].append(max_g)
        yvalues.append(limit)

    #with open(onames[0].replace(".pdf", ".json").replace(".png", ".json"), "w") as jj:
    #    json.dump(limits, jj, indent = 1)

    fig, ax = plt.subplots(dpi=600)
    handles = []
    ymin = 0.
    ymax = 0.

    for ii, yy in enumerate(yvalues):
        if drawband or ii == 0:
            ax.fill_between(xvalues, np.array(yy["exp-2"]), np.array(yy["exp+2"]),
                            color = draw_1D.colors[len(limits)][ii]["exp2"], linewidth = 0, alpha = draw_1D.colors[len(limits)][ii]["alpe"])

            label = "95% expected" if labels[ii] == "" else "95% exp."
            handles.append((mpt.Patch(color = draw_1D.colors[len(limits)][ii]["exp2"], alpha = draw_1D.colors[len(limits)][ii]["alpe"]),
                            label + " " + labels[ii]))
            ymin = min(ymin, min(yy["exp-2"]))
            ymax = max(ymax, max(yy["exp+2"]))
            ymax1 = math.ceil(ymax * 2.) / 2.

    for ii, yy in enumerate(yvalues):
        if drawband or ii == 0:
            ax.fill_between(xvalues, np.array(yy["exp-1"]), np.array(yy["exp+1"]),
                            color = draw_1D.colors[len(limits)][ii]["exp1"], linewidth = 0, alpha = draw_1D.colors[len(limits)][ii]["alpe"])

            label = "68% expected" if labels[ii] == "" else "68% exp."
            handles.append((mpt.Patch(color = draw_1D.colors[len(limits)][ii]["exp1"], alpha = draw_1D.colors[len(limits)][ii]["alpe"]),
                            label + " " + labels[ii]))

    for ii, yy in enumerate(yvalues):
        ax.plot(xvalues, np.array(yy["exp0"]), color = draw_1D.colors[len(limits)][ii]["exp0"], linestyle = draw_1D.colors[len(limits)][ii]["expl"], linewidth = 1.5)

        label = "Median expected" if labels[ii] == "" else "Median exp."
        handles.append((mln.Line2D([0], [0], color = draw_1D.colors[len(limits)][ii]["exp0"], linestyle = draw_1D.colors[len(limits)][ii]["expl"], linewidth = 1.5),
                        label + " " + labels[ii]))

        ymin = min(ymin, min(yy["exp0"]))
        ymax = max(ymax, max(yy["exp0"]))
        ymax1 = math.ceil(ymax * 2.) / 2.

    if '_m' in onames[0] or '_w' in onames[0]:
        fixed_value = onames[0].split('/')[-1]
        fixed_value = fixed_value.split('_')
        parity = fixed_value[0]
        fixed_value = [ff for ff in fixed_value if ff.startswith('m') or ff.startswith('w') ][0]
        ismass = 'm' in fixed_value
        fixed_value = float(fixed_value.replace('m', '').replace('w', '').replace('p', '.'))

        max_partial_g = [ahtt_max_coupling(parity, fixed_value, xx / 100.) if ismass else ahtt_max_coupling(parity, xx, fixed_value / 100.) for xx in xvalues]
        max_partial_g = [(xx, gg) for xx, gg in zip(xvalues, max_partial_g)]
        xmaxg = [xx for xx, gg in max_partial_g]
        max_partial_g = [gg for xx, gg in max_partial_g]
        may_partial_g = [min(gg + ((ymax1 / 50.) * (max_g - ymin)), max_g) for gg in max_partial_g]

        ax.fill_between(np.array(xmaxg), np.array(max_partial_g), np.array(may_partial_g), facecolor = 'none', hatch = '||', edgecolor = '#848482', linewidth = 0.)
        ax.plot(np.array(xmaxg), np.array(max_partial_g), color = '#848482', linestyle = "solid", linewidth = 1.5)

    if observed:
        for i1, yy in enumerate(yvalues):
            ymin = min(ymin, min([min(first(oo)) for oo in yy["obs"]]))
            ymax = max(ymax, max([g for oo in yy["obs"] for g, cls in oo if g < ymax or cls > 0.05]))
            ymax1 = math.ceil(ymax * 2.) / 2.

            ydots = np.arange(0., ymax1 + epsilon, 0.005)
            xv, yv = np.meshgrid(xvalues, ydots)
            zv = np.zeros_like(xv)
            gs = [first(gc) for gc in yy["obs"]]
            cls = [second(gc) for gc in yy["obs"]]

            for ir, xr in enumerate(xv):
                for ic, xc in enumerate(xr):
                    gg = yv[ir][ic]
                    mm = xv[ir][ic]

                    if gg in gs[ic]:
                        zv[ir][ic] = cls[ic][gs[ic].index(gg)]
                    else:
                        for g in gs[ic]:
                            i2 = -1
                            if g > gg:
                                i2 = gs[ic].index(g)
                                break
                        zv[ir][ic] = 0.5 * (cls[ic][i2] + cls[ic][i2 - 1]) if i2 > 0 else 0.

            cols = [draw_1D.colors[len(limits)][i1]["obsf"]]
            cmap = mcl.ListedColormap(cols)
            cf = ax.contourf(xv, yv, zv, [-1., 0.05], colors = cols, alpha = draw_1D.colors[len(limits)][i1]["alpo"])
            ax.contour(cf, colors = draw_1D.colors[len(limits)][i1]["obsl"], linewidths = 2)

            #ax.plot(xvalues, np.array(yy["obs"]), color = draw_1D.colors[len(limits)][i1]["obs"], linestyle = 'solid', linewidth = 2)

            label = "Observed" if labels[i1] == "" else "Obs."
            handles.append((mpt.Patch(facecolor = mcl.to_rgba(draw_1D.colors[len(limits)][i1]["obsf"], draw_1D.colors[len(limits)][i1]["alpo"]),
                                      edgecolor = mcl.to_rgba(draw_1D.colors[len(limits)][i1]["obsl"], 1.),
                                      linewidth = 2, linestyle = 'solid'), label + " " + labels[i1]))

    ymax2 = math.ceil(ymax1 * 5.5) / 4.
    if ymax1 / ymax2 < 0.6:
        ymax1 += 0.25
    elif ymax1 / ymax2 > 0.7:
        ymax2 += 0.25

    plt.ylim((ymin, ymax2))
    ax.fill_between([xvalues[0], xvalues[-1]], [ymax1, ymax1], [ymax2, ymax2], facecolor = 'white', linewidth = 0., zorder = 2.001)
    ax.plot([xvalues[0], xvalues[-1]], [ymax1, ymax1], color = "black", linestyle = 'solid', linewidth = 2.002)
    plt.xlabel(xaxis, fontsize = 26, loc = "right")
    plt.ylabel(yaxis, fontsize = 26, loc = "top")
    ax.margins(x = 0, y = 0)

    # resorting to get a columnwise fill in legend
    handles = [hh for label in labels for hh in handles if str(label + " ") in str(hh[1] + " ")] if len(limits) > 1 else handles
    handles.append((mpt.Patch(hatch = '||', facecolor = 'none', edgecolor = '#848482', linewidth = 1.), gcurve))

    lheight = (ymax2 - ymax1) / (ymax2 - ymin)
    lmargin = 0.04 if len(limits) == 1 else 0.02
    lwidth = 1. - (2. * lmargin)
    legend = ax.legend(first(handles), second(handles),
	               loc = "upper right", ncol = 2 if len(limits) < 3 else 3, bbox_to_anchor = (lmargin, 1. - lheight, lwidth, lheight - 0.025),
                       mode = "expand", borderaxespad = 0., handletextpad = 0.35, fontsize = 21 if len(limits) < 3 else 15, frameon = False,
                       title = "95% CL exclusion" + ltitle, title_fontsize = 21, facecolor = 'white')
    ax.add_artist(legend)

    if formal:
        xwindow = xvalues[-1] - xvalues[0]
        ctxt = "{cms}".format(cms = r"$\textbf{CMS}$")
        ax.text(0.02 * xwindow + xvalues[0], 1.0005 * ymax2, ctxt, fontsize = 31, ha = 'left', va = 'bottom', usetex = True)

        if cmsapp != "":
            atxt = "{app}".format(app = r" $\textit{" + cmsapp + r"}$")
            ax.text(0.18 * xwindow + xvalues[0], 1.005 * ymax2, atxt, fontsize = 26, ha = 'left', va = 'bottom', usetex = True)

        ltxt = "{lum}{ifb}".format(lum = luminosity, ifb = r" fb$^{\mathrm{\mathsf{-1}}}$ (13 TeV)")
        ax.text(0.99 * xwindow + xvalues[0], 1.005 * ymax2, ltxt, fontsize = 26, ha = 'right', va = 'bottom', usetex = True)

        btxt = etat_blurb(a343bkg)
        bbln = [matplotlib.patches.Rectangle((0, 0), 1, 1, fc = "white", ec = "white", lw = 0, alpha = 0)] * len(btxt)
        ax.legend(bbln, btxt, loc = 'lower right',
                  bbox_to_anchor = (0.85, 0.001 if len(btxt) > 1 else 0.005, 0.15, 0.1),
                  fontsize = 17 if len(btxt) > 1 else 17, frameon = False,
                  handlelength = 0, handletextpad = 0, borderaxespad = 0.4,
                  labelspacing=0.2)

    if ymax2 > 1.75:
        ax.yaxis.set_major_locator(mtc.MultipleLocator(0.5))
    ax.minorticks_on()
    ax.tick_params(axis = "both", which = "both", direction = "in", bottom = True, top = False, left = True, right = True)
    ax.tick_params(axis = "both", which = "major", width = 1, length = 8, labelsize = 22)
    ax.tick_params(axis = "both", which = "minor", width = 1, length = 3)

    fig.set_size_inches(8., 8.)
    fig.tight_layout()
    for oname in onames:
        fig.savefig(oname, transparent = transparent)
    plt.close()

def parse_point_quantile(pqs, var = "mass"):
    other = 2 if var == "mass" else 1
    result = [tokenize_to_list(pq, token = ':') for pq in pqs]
    result = {get_point(pq[0])[other]: tokenize_to_list(pq[1]) for pq in result}
    return result

def draw_natural(onames, points, directories, labels, xaxis, yaxis, interpolate, onepoi, drawband, observed, formal, cmsapp, luminosity, a343bkg, transparent):
    masses = [pnt[1] for pnt in points]
    if len(set(masses)) != len(masses):
        raise RuntimeError("producing " + onames[0] + ", --function natural expects unique mass points only. aborting")
    if len(masses) < 2:
        print("There are less than 2 masses points. skipping")

    draw_1D(onames, read_limit(directories, masses, onepoi), labels, xaxis, yaxis, "",
            parse_point_quantile(interpolate), drawband, observed, formal, cmsapp, luminosity, a343bkg, transparent)

def draw_variable(var1, onames, points, directories, otags, labels, yaxis, interpolate, onepoi, drawband, observed, formal, cmsapp, luminosity, a343bkg, transparent, dump_spline):
    if not hasattr(draw_variable, "settings"):
        draw_variable.settings = OrderedDict([
            ("mass",  {"var2": "width", "iv1": 1, "iv2": 2, "label": r", $\Gamma_{\mathrm{\mathsf{%s}}}\,=$ %.1f%% m$_{\mathrm{\mathsf{%s}}}$"}),
            ("width", {"var2": "mass", "iv1": 2, "iv2": 1, "label": r", m$_{\mathrm{\mathsf{%s}}}\,=$ %d GeV"})
        ])

    var2s = set([pnt[draw_variable.settings[var1]["iv2"]] for pnt in points])

    for vv in var2s:
        print(f"running {draw_variable.settings[var1]['var2']} {vv}")
        var1s = [pnt[draw_variable.settings[var1]["iv1"]] for pnt in points if pnt[draw_variable.settings[var1]["iv2"]] == vv]
        dirs = [[dd for dd, pnt in zip(directory, points) if pnt[draw_variable.settings[var1]["iv2"]] == vv] for directory in directories]

        if len(var1s) < 2 or not all([len(dd) == len(var1s) for dd in dirs]):
            print(f"{draw_variable.settings[var1]['var2']} {vv} has too few {var1}, or inconsistent input. skipping")
            continue

        axislabel = points[0][0] if var1 == "mass" else (points[0][0], points[0][0])
        legendtext = (points[0][0], vv, points[0][0]) if var1 == "mass" else (points[0][0], vv)
        draw_1D([oname.format(www = 'w' + str(vv).replace('.', 'p') if var1 == "mass" else 'm' + str(int(vv))) for oname in onames],
                read_limit(dirs, otags, var1s, onepoi),
                labels, axes[var1] % axislabel, yaxis,
                draw_variable.settings[var1]["label"] % legendtext,
                r'$\Gamma_{\mathrm{\mathsf{%s} t\bar{t}}} \,>\, \Gamma_{\mathrm{\mathsf{%s}}}$' % (points[0][0], points[0][0]),
                parse_point_quantile(interpolate), drawband, observed, formal, cmsapp, luminosity, a343bkg, transparent)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--function", help = "plot limit as a function of?", default = "mass",
                        choices = ["natural", "mass", "width"], required = False)
    parser.add_argument("--tag", help = "input tag-output-tag pairs to search. the pairs are semicolon separated, and tags colon-separated, "
                        "so e.g. when there are 2 tags: 't1:o1;t2:o2...", dest = "itag", default = "", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))
    parser.add_argument("--plot-tag", help = "extra tag to append to plot names", dest = "ptag", default = "", required = False, type = prepend_if_not_empty)
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False, type = remove_spaces_quotes)
    parser.add_argument("--label", help = "labels to attach on plot for each input tags, semicolon separated", default = "", required = False, type = lambda s: tokenize_to_list(s, ';' ))
    parser.add_argument("--drop",
                        help = "comma separated list of points to be dropped. 'XX, YY' means all points containing XX or YY are dropped.",
                        default = "", required = False, type = lambda s: [] if s == "" else tokenize_to_list( remove_spaces_quotes(s) ) )
    parser.add_argument("--interpolate",
                        help = "semicolon-separated point: quantiles to linearly interpolate based on neighboring points.\n"
                        "e.g. A_m425_w10p0: exp-1,exp0 to interpolate these two quantiles"
                        "the neighboring points are based on --function.",
                        default = "", required = False, type = lambda s: [] if s == "" else tokenize_to_list( remove_spaces_quotes(s), token = ';' ) )

    parser.add_argument("--one-poi", help = "plot limits set with the g-only model", dest = "onepoi", action = "store_true", required = False)
    parser.add_argument("--observed", help = "draw observed limits as well", dest = "observed", action = "store_true", required = False)

    parser.add_argument("--formal", help = "plot is for formal use - put the CMS text etc",
                        dest = "formal", action = "store_true", required = False)
    parser.add_argument("--cms-append", help = "text to append to the CMS text, if --formal is used", dest = "cmsapp", default = "", required = False)
    parser.add_argument("--luminosity", help = "integrated luminosity applicable for the plot, written if --formal is used", default = "138", required = False)
    parser.add_argument("--A343-background",
                        help = "a comma-separated list of 4 values for etat background text, written if --formal is used"
                        "syntax: (bool, 1 or 0, whether it is included as bkg, best fit xsec, xsec uncertainty (lo/hi)",
                        dest = "a343bkg", default = default_etat_measurement(), required = False,
                        type = default_etat_measurement)

    parser.add_argument("--opaque-background", help = "make the background white instead of transparent",
                        dest = "transparent", action = "store_false", required = False)
    parser.add_argument("--skip-secondary-bands", help = "do not draw the +-1, 2 sigma bands except for first tag",
                        dest = "drawband", action = "store_false", required = False)
    parser.add_argument("--dump-spline", help = "dump the splines used to obtain the cls = 0.05 crossing",
                        dest = "dump_spline", action = "store_true", required = False)
    parser.add_argument("--plot-formats", help = "comma-separated list of formats to save the plots in", default = [".png"], dest = "fmt", required = False,
                        type = lambda s: [prepend_if_not_empty(fmt, '.') for fmt in tokenize_to_list(remove_spaces_quotes(s))])

    parser.add_argument("--read-from", help = "this is the location where workspace dirs are searched for",
                        dest = "basedir", default = ".", required = False, type = append_if_not_empty)

    args = parser.parse_args()
    if len(args.itag) != len(args.label):
        if len(args.itag) == 1 and len(args.label) == 0:
            args.label =[""]
        else:
            raise RuntimeError("length of tags isnt the same as labels. aborting")

    adir = [[pnt for pnt in sorted(glob.glob(f'{args.basedir}A*_w*' + tag.split(':')[0])) if len(args.drop) == 0 or not any([dd in pnt for dd in args.drop])] for tag in args.itag]
    hdir = [[pnt for pnt in sorted(glob.glob(f'{args.basedir}H*_w*' + tag.split(':')[0])) if len(args.drop) == 0 or not any([dd in pnt for dd in args.drop])] for tag in args.itag]
    otags = [tag.split(':')[1] if len(tag.split(':')) > 1 else tag.split(':')[0] for tag in args.itag]

    apnt = [[get_point(tokenize_to_list(pnt, '/')[-1]) for pnt in directory] for directory in adir]
    hpnt = [[get_point(tokenize_to_list(pnt, '/')[-1]) for pnt in directory] for directory in hdir]
    #print(apnt)
    #raise RuntimeError("xxx")
    if not all([pnt == apnt[0]] for pnt in apnt):
        raise RuntimeError("A signal points are not the same between args.itag. aborting")

    if not all([pnt == hpnt[0]] for pnt in hpnt):
        raise RuntimeError("H signal points are not the same between args.itag. aborting")

    # keep only the points of the first tag, as they're all the same
    apnt = apnt[0]
    hpnt = hpnt[0]

    for ii in range(len(adir)):
        adir[ii] = [dd for dd, pnt in sorted(zip(adir[ii], apnt), key = lambda tup: (tup[1][1], tup[1][2]))]
    apnt.sort(key = lambda tup: (tup[1], tup[2]))
    for ii in range(len(hdir)):
        hdir[ii] = [dd for dd, pnt in sorted(zip(hdir[ii], hpnt), key = lambda tup: (tup[1][1], tup[1][2]))]
    hpnt.sort(key = lambda tup: (tup[1], tup[2]))

    if args.function == "natural":
        if len(apnt) > 0:
            draw_natural(["{ooo}/A_limit_natural_{mod}{tag}{f}".format(ooo = args.odir, mod = "one-poi" if args.onepoi else "g-scan", tag = args.ptag, f = f) for f in args.fmt],
                         apnt, adir, otags, args.label, axes["mass"] % apnt[0][0], axes["ttcoupling"] % apnt[0][0], args.interpolate, args.onepoi, args.drawband, args.observed, args.formal, args.cmsapp, args.luminosity, args.a343bkg, args.transparent)
        if len(hpnt) > 0:
            draw_natural(["{ooo}/H_limit_natural_{mod}{tag}{f}".format(ooo = args.odir, mod = "one-poi" if args.onepoi else "g-scan", tag = args.ptag, f = f) for f in args.fmt],
                         hpnt, hdir, otags, args.label, axes["mass"] % hpnt[0][0], axes["ttcoupling"] % hpnt[0][0], args.interpolate, args.onepoi, args.drawband, args.observed, args.formal, args.cmsapp, args.luminosity, args.a343bkg, args.transparent)
    else:
        if len(apnt) > 0:
            draw_variable(args.function,
                          ["{ooo}/A_limit_{www}_{mod}{tag}{f}".format(ooo = args.odir, www = r"{www}", mod = "one-poi" if args.onepoi else "g-scan",
                                                                      tag = args.ptag, f = f) for f in args.fmt],
                          apnt, adir, otags, args.label, axes["ttcoupling"] % apnt[0][0], args.interpolate, args.onepoi, args.drawband, args.observed, args.formal, args.cmsapp, args.luminosity, args.a343bkg, args.transparent, args.dump_spline)
        if len(hpnt) > 0:
            draw_variable(args.function,
                          ["{ooo}/H_limit_{www}_{mod}{tag}{f}".format(ooo = args.odir, www = r"{www}", mod = "one-poi" if args.onepoi else "g-scan",
                                                                      tag = args.ptag, f = f) for f in args.fmt],
                          hpnt, hdir, otags, args.label, axes["ttcoupling"] % hpnt[0][0], args.interpolate, args.onepoi, args.drawband, args.observed, args.formal, args.cmsapp, args.luminosity, args.a343bkg, args.transparent, args.dump_spline)
    pass
