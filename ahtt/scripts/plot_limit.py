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

from drawings import min_g, max_g, epsilon, axes, first, second, get_point

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

    beta = math.sqrt(1. - (4. * mt2 / mah2))
    factor = mah * mt2 * term3 * beta
    return factor if parity == 'A' else factor * beta * beta if parity == 'H' else 0.

def ahtt_partial_width(parity, mah, gah, relwidth = True):
    wah = gah * gah * ahtt_width_coupling_helper(parity, mah)
    return wah / mah if relwidth else wah

def ahtt_max_coupling(parity, mah, wah, relwidth = True):
    wah = wah * mah if relwidth else wah
    return math.sqrt(wah / ahtt_width_coupling_helper(parity, mah))

def read_limit(directories, xvalues, onepoi, dump_spline, odir):
    limits = [OrderedDict() for tag in directories]

    for tt, tag in enumerate(directories):
        for jj, dd in enumerate(tag):
            #print(dd)
            limit = OrderedDict([
                ("exp-2", []),
                ("exp-1", []),
                ("exp0", []),
                ("exp+1", []),
                ("exp+2", []),
                ("obs", [])
            ])

            exclusion = OrderedDict([
                ("exp-2", False),
                ("exp-1", False),
                ("exp0", False),
                ("exp+1", False),
                ("exp+2", False),
            ])

            if onepoi:
                with open("{dd}/{pnt}_limits_one-poi.json".format(dd = dd, pnt = '_'.join(dd.split('_')[:3]))) as ff:
                    result = json.load(ff)

                    for mm, lmt in result.items():
                        for quantile, g in lmt.items():
                            limit[quantile].append([g, max_g])
            else:
                chunks = glob.glob("{dd}/{pnt}_limits_g-scan_n*_i*.json".format(dd = dd, pnt = '_'.join(dd.split('_')[:3])))
                chunks.sort()
                for nn in chunks:
                    with open(nn) as ff:
                        result = json.load(ff)

                    for g, lmt in result.items():
                        # FIXME in some cases combine gives strange cls values
                        # not understood why, skip
                        if any([round(cls, 3) > 1. or round(cls, 3) < 0. for cls in lmt.values()]):
                            continue

                        for quantile, cls in lmt.items():
                            limit[quantile].append((round(float(g), 4), round(cls, 5)))

                for quantile in limit.keys():
                    if quantile == "obs":
                        continue

                    g = [gg for gg, cc in limit[quantile]]
                    cls = [cc for gg, cc in limit[quantile]]
                    vmin = min([(abs(cc - 0.05), gg, cc) for gg, cc in limit[quantile]])
                    vmin = (vmin[1], vmin[2])
                    imin = limit[quantile].index(vmin)

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
                            for ii in range(1, len(limit[quantile])):
                                gg, cc = limit[quantile][ii]
                                if cmin < cc < cmax:
                                    if (len(g) == 0 and len(cls) == 0) or condition(cc, cls[-1], vmin[1] < cls[0]):
                                        g.append(gg)
                                        cls.append(cc)
                                    elif len(g) > 1 and len(cls) > 1 and ii < len(limit[quantile]) - 1:
                                        cprev = condition(cc, cls[-2], vmin[1] < cls[0])
                                        nprev = condition(limit[quantile][ii + 1][1], cls[-2], vmin[1] < cls[0])

                                        if cprev and nprev:
                                            g.pop()
                                            cls.pop()

                                            g.append(gg)
                                            cls.append(cc)
                    else:
                        g = []
                        cls = []

                    if len(g) > 3:
                        spline = UnivariateSpline(np.array(g), np.array(cls))

                        smin = min([(abs(cc - 0.05), gg, cc) for gg, cc in zip(g, cls)])
                        smin = (smin[1], smin[2])

                        min_factor = 1.
                        abs_tolerance = 0.0005
                        need_checking = False
                        crossing = smin[0]
                        residual = abs(spline(crossing) - 0.05)
                        factor = 2.**9 if smin[1] > 0.05 else -2.**9

                        while residual > abs_tolerance and crossing < max_g and crossing > min_g:
                            #if "m750" in dd and "w21p0" in dd and quantile == "exp+2":
                            #    print(crossing, residual, factor, vmin, smin)

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
                            print("in " + dd + ", quantile " + quantile + ", achieved cls residual is " +
                                  str(residual) + " at g = " + str(crossing))
                            print("g and cls values used to build the spline:")
                            print(g)
                            print(cls)
                            print("g, cls point with minimum distance to cls = 0.05 from a raw search on sampled points: ", vmin)
                            print("g, cls point with minimum distance to cls = 0.05 from points considered for the spline: ", smin)
                            print("\n")

                        if dump_spline or need_checking:
                            qstr = quantile.replace('+', 'pp').replace('-', 'm')
                            fig, ax = plt.subplots()

                            ax.plot(g, spline(g), 'g', lw = 3)
                            fig.tight_layout()
                            fig.savefig("{dd}/{pnt}_spline_{qua}.png".format(dd = odir, pnt = '_'.join(dd.split('_')[:3]), qua = qstr), transparent = True)
                            fig.clf()

                        limit[quantile] = [[crossing, max_g]] if crossing < max_g else []

                    else:
                        print("in " + dd + ", quantile " + quantile + ", following g and cls are insufficient to form a spline:")
                        print(g)
                        print(cls)
                        print("g, cls point with minimum distance to cls = 0.05 from a raw search on sampled point: ", vmin)
                        print("\n")

                        limit[quantile] = []

            limits[tt][xvalues[jj]] = limit

    return limits

def draw_1D(oname, limits, labels, xaxis, yaxis, ltitle, gcurve, drawband, observed, transparent):
    if len(limits) > 6:
        raise RuntimeError("current plotting code is not meant for more than 6 tags. aborting")

    if len(limits) > 1 and not all([list(ll) == list(limits[0]) for ll in limits]):
        raise RuntimeError("limits in the tags are not over the same x points for plot " + oname + ". aborting")

    if not hasattr(draw_1D, "colors"):
        draw_1D.colors = OrderedDict([
            (1    , [{"exp2": "#ffcc00", "exp1": "#00cc00", "exp0": "0", "expl": "dashed", "obsf": "#0033cc", "obsl": "#0033cc", "alpe": 1., "alpo": 0.25}]),

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
                    limit[quantile].append(exclusion)
                else:
                    if len(exclusion) > 1:
                        print(quantile, exclusion)
                        raise RuntimeError("tag number " + str(tt) + ", xvalue " + str(xx) + ", quantile " + quantile + ", plot " + oname +
                                       ", current plotting code is meant to handle only 1 expected exclusion interval. aborting")

                    if len(exclusion) > 0:
                        limit[quantile].append(exclusion[0][0])
                        if exclusion[0][1] != max_g:
                            print(quantile, exclusion)
                            print("tag number " + str(tt) + ", xvalue " + str(xx) + ", quantile " + quantile + ", plot " + oname +
                                  " strange exclusion interval. recheck.\n")
                    else:
                        limit[quantile].append(max_g)

        yvalues.append(limit)

    #with open(oname.replace(".pdf", ".json").replace(".png", ".json"), "w") as jj: 
    #    json.dump(limits, jj, indent = 1)

    fig, ax = plt.subplots()
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

        label = "Expected" if labels[ii] == "" else "Exp."
        handles.append((mln.Line2D([0], [0], color = draw_1D.colors[len(limits)][ii]["exp0"], linestyle = draw_1D.colors[len(limits)][ii]["expl"], linewidth = 1.5),
                        label + " " + labels[ii]))

        ymin = min(ymin, min(yy["exp0"]))
        ymax = max(ymax, max(yy["exp0"]))
        ymax1 = math.ceil(ymax * 2.) / 2.

    if '_m' in oname or '_w' in oname:
        fixed_value = oname.split('/')[-1]
        fixed_value = fixed_value.split('_')
        parity = fixed_value[0]
        fixed_value = [ff for ff in fixed_value if ff.startswith('m') or ff.startswith('w') ][0]
        ismass = '_m' in fixed_value
        fixed_value = float(fixed_value.replace('m', '').replace('w', '').replace('p', '.'))

        max_partial_g = [ahtt_max_coupling(parity, fixed_value, xx) if ismass else ahtt_max_coupling(parity, xx, fixed_value / 100.) for xx in xvalues]

        if fixed_value == 8.:
            print(max_partial_g)

        max_partial_g = [gg for gg in max_partial_g if gg < ymax1]
        xmaxg = [xx for gg, xx in zip(max_partial_g, xvalues) if gg < ymax1]
        may_partial_g = [min(gg + 0.05 * (ymax1 - ymin), ymax1) for gg in max_partial_g]

        if fixed_value == 8.:
            print(xmaxg)
            print(max_partial_g)

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

    ymax2 = math.ceil(ymax1 * 2.8) / 2.
    plt.ylim((ymin, ymax2))
    ax.plot([xvalues[0], xvalues[-1]], [ymax1, ymax1], color = "black", linestyle = 'solid', linewidth = 2)
    plt.xlabel(xaxis, fontsize = 21, loc = "right")
    plt.ylabel(yaxis, fontsize = 21, loc = "top")
    ax.margins(x = 0, y = 0)

    # resorting to get a columnwise fill in legend
    handles = [hh for label in labels for hh in handles if str(label + " ") in str(hh[1] + " ")] if len(limits) > 1 else handles
    handles.append((mpt.Patch(hatch = '||', facecolor = 'none', edgecolor = '#848482', linewidth = 1.), gcurve))

    lheight = (ymax2 - ymax1) / (ymax2 - ymin)
    lmargin = 0.06 if len(limits) == 1 else 0.02
    lwidth = 1. - (2. * lmargin) 
    legend = ax.legend(first(handles), second(handles),
	               loc = "upper right", ncol = 2 if len(limits) < 3 else 3, bbox_to_anchor = (lmargin, 1. - lheight, lwidth, lheight - 0.025),
                       mode = "expand", borderaxespad = 0., handletextpad = 0.5, fontsize = 21 if len(limits) < 3 else 15, frameon = False,
                       title = "95% CL exclusion" + ltitle, title_fontsize = 21)
    #fontprop = matplotlib.font_manager.FontProperties()
    #fontprop.set_size(21)
    #legend.set_title(title = "95% CL exclusion", prop = fontprop)

    ax.minorticks_on()
    ax.tick_params(axis = "both", which = "both", direction = "in", bottom = True, top = False, left = True, right = True)
    ax.tick_params(axis = "both", which = "major", width = 1, length = 8, labelsize = 18)
    ax.tick_params(axis = "both", which = "minor", width = 1, length = 3)

    fig.set_size_inches(8., 8.)
    fig.tight_layout()
    fig.savefig(oname, transparent = transparent)
    fig.clf()

def draw_natural(oname, points, directories, labels, xaxis, yaxis, onepoi, drawband, observed, transparent):
    masses = [pnt[1] for pnt in points]
    if len(set(masses)) != len(masses):
        raise RuntimeError("producing " + oname + ", --function natural expects unique mass points only. aborting")

    if len(masses) < 2:
        print("There are less than 2 masses points. skipping")

    draw_1D(oname, read_limit(directories, masses, onepoi), labels, xaxis, yaxis, "", drawband, observed, transparent)

def draw_variable(var1, oname, points, directories, labels, yaxis, onepoi, drawband, observed, transparent, dump_spline):
    if not hasattr(draw_variable, "settings"):
        draw_variable.settings = OrderedDict([
            ("mass",  {"var2": "width", "iv1": 1, "iv2": 2, "label": r", $\Gamma_{\mathrm{\mathsf{%s}}}\,=$ %.1f%% m$_{\mathrm{\mathsf{%s}}}$"}),
            ("width", {"var2": "mass", "iv1": 2, "iv2": 1, "label": r", m$_{\mathrm{\mathsf{%s}}}\,=$ %d GeV"})
        ])

    var2s = set([pnt[draw_variable.settings[var1]["iv2"]] for pnt in points])

    for vv in var2s:
        print("running", draw_variable.settings[var1]["var2"], vv)
        var1s = [pnt[draw_variable.settings[var1]["iv1"]] for pnt in points if pnt[draw_variable.settings[var1]["iv2"]] == vv]
        dirs = [[dd for dd, pnt in zip(tag, points) if pnt[draw_variable.settings[var1]["iv2"]] == vv] for tag in directories]

        if len(var1s) < 2 or not all([len(dd) == len(var1s) for dd in dirs]):
            print(draw_variable.settings[var1]["var2"] + str(vv) + " has too few " + var1 + ", or inconsistent input. skipping")
            continue

        axislabel = points[0][0] if var1 == "mass" else (points[0][0], points[0][0])
        legendtext = (points[0][0], vv, points[0][0]) if var1 == "mass" else (points[0][0], vv)
        draw_1D(oname.format(www = 'w' + str(vv).replace('.', 'p') if var1 == "mass" else 'm' + str(int(vv))),
                read_limit(dirs, var1s, onepoi, dump_spline, os.path.dirname(oname)),
                labels, axes[var1] % axislabel, yaxis,
                draw_variable.settings[var1]["label"] % legendtext,
                r'$\Gamma_{\mathrm{\mathsf{%s} t\bar{t}}} \,>\, \Gamma_{\mathrm{\mathsf{%s}}}$' % (points[0][0], points[0][0]),
                drawband, observed, transparent)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--function", help = "plot limit as a function of?", default = "mass",
                        choices = ["natural", "mass", "width"], required = False)
    parser.add_argument("--itag", help = "input directory tags to search, semicolon separated", default = "", required = False)
    parser.add_argument("--otag", help = "extra tag to append to plot names", default = "", required = False)
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False)
    parser.add_argument("--label", help = "labels to attach on plot for each input tags, semicolon separated", default = "", required = False)
    parser.add_argument("--drop",
                        help = "comma separated list of points to be dropped. 'XX, YY' means all points containing XX or YY are dropped.",
                        default = "", required = False)
    parser.add_argument("--one-poi", help = "plot limits set with the g-only model", dest = "onepoi", action = "store_true", required = False)
    parser.add_argument("--observed", help = "draw observed limits as well", dest = "observed", action = "store_true", required = False)
    parser.add_argument("--transparent-background", help = "make the background transparent instead of white",
                        dest = "transparent", action = "store_true", required = False)
    parser.add_argument("--skip-secondary-bands", help = "do not draw the +-1, 2 sigma bands except for first tag",
                        dest = "drawband", action = "store_false", required = False)
    parser.add_argument("--dump-spline", help = "dump the splines used to obtain the cls = 0.05 crossing",
                        dest = "dump_spline", action = "store_true", required = False)
    parser.add_argument("--plot-format", help = "format to save the plots in", default = "png", dest = "fmt", required = False)

    args = parser.parse_args()
    if (args.otag != "" and not args.otag.startswith("_")):
        args.otag = "_" + args.otag

    if (args.fmt != "" and not args.fmt.startswith(".")):
        args.fmt = "." + args.fmt

    tags = args.itag.replace(" ", "").split(';')
    labels = args.label.split(';')

    if len(tags) != len(labels):
        raise RuntimeError("length of tags isnt the same as labels. aborting")

    drops = args.drop.replace(" ", "").split(',') if args.drop != "" else []
    adir = [[pnt for pnt in sorted(glob.glob('A*_w*' + tag)) if len(drops) == 0 or not any([dd in pnt for dd in drops])] for tag in tags]
    hdir = [[pnt for pnt in sorted(glob.glob('H*_w*' + tag)) if len(drops) == 0 or not any([dd in pnt for dd in drops])] for tag in tags]

    apnt = [[get_point(pnt) for pnt in tag] for tag in adir]
    hpnt = [[get_point(pnt) for pnt in tag] for tag in hdir]

    if not all([pnt == apnt[0]] for pnt in apnt):
        raise RuntimeError("A signal points are not the same between tags. aborting")

    if not all([pnt == hpnt[0]] for pnt in hpnt):
        raise RuntimeError("H signal points are not the same between tags. aborting")

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
            draw_natural("{ooo}/A_limit_natural_{mod}{tag}{fmt}".format(ooo = args.odir, mod = "one-poi" if args.onepoi else "g-scan", tag = args.otag, fmt = args.fmt),
                         apnt, adir, labels, axes["mass"] % apnt[0][0], axes["coupling"] % apnt[0][0], args.onepoi, args.drawband, args.observed, args.transparent)
        if len(hpnt) > 0:
            draw_natural("{ooo}/H_limit_natural_{mod}{tag}{fmt}".format(ooo = args.odir, mod = "one-poi" if args.onepoi else "g-scan", tag = args.otag, fmt = args.fmt),
                         hpnt, hdir, labels, axes["mass"] % hpnt[0][0], axes["coupling"] % hpnt[0][0], args.onepoi, args.drawband, args.observed, args.transparent)
    else:
        if len(apnt) > 0:
            draw_variable(args.function,
                          "{ooo}/A_limit_{www}_{mod}{tag}{fmt}".format(ooo = args.odir, www = r"{www}", mod = "one-poi" if args.onepoi else "g-scan",
                                                                       tag = args.otag, fmt = args.fmt),
                          apnt, adir, labels, axes["coupling"] % apnt[0][0], args.onepoi, args.drawband, args.observed, args.transparent, args.dump_spline)
        if len(hpnt) > 0:
            draw_variable(args.function,
                          "{ooo}/H_limit_{www}_{mod}{tag}{fmt}".format(ooo = args.odir, www = r"{www}", mod = "one-poi" if args.onepoi else "g-scan",
                                                                       tag = args.otag, fmt = args.fmt),
                          hpnt, hdir, labels, axes["coupling"] % hpnt[0][0], args.onepoi, args.drawband, args.observed, args.transparent, args.dump_spline)
    pass
