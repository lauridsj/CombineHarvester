import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator, make_interp_spline
import glob
import os
from tqdm import tqdm
import math
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict

from drawings import min_g, max_g, epsilon, axes, first, second, get_point
from drawings import default_etat_measurement, etat_blurb
from desalinator import prepend_if_not_empty, append_if_not_empty, tokenize_to_list, remove_spaces_quotes

from plot_limit import ahtt_partial_width, draw_1D, parse_point_quantile

def interpolate_limit(parity, dirs, points, otag, observed=False):

    limitdict = {}
    for d, pnt in zip(dirs, points):
        mass = pnt[1]
        width = pnt[2]
        
        jsons = glob.glob(d + f"/*_{otag}_limits_g-scan*.json")
        for jf in jsons:
            with open(jf) as f:
                lim = json.load(f)
            for g, dd in lim.items():
                for sc, pval in dd.items():
                    limitdict[(mass, width, float(g), sc)] = pval
                    if width == 0.5:
                       limitdict[(mass, 0., float(g), sc)] = pval

    masses = sorted(set(p[1] for p in points))
    gplot = np.linspace(min_g, max_g, 300)
    #scenarios = ["exp0", "exp+1", "exp-1", "exp+2", "exp-2"]
    scenarios = ["exp0"]
    if observed:
        scenarios.append("obs")

    pvals = OrderedDict()
    for i, m in enumerate(masses):
        pvals[m] = OrderedDict()
        for sc in scenarios:

            #if parity == "H" and m == 800 and sc == "exp0":
            #    breakpoint()
            
            parr = np.array([[k[1], k[2], v] for k,v in limitdict.items() if k[0] == m and k[3] == sc])
            interp = CloughTocher2DInterpolator(parr[:,0:2], parr[:,2])
        
            w = ahtt_partial_width(parity, m, gplot) * 100
            w = np.where(w > 25., 25., w)
            w = np.where(w < 0.5, 0.5, w)
            
            pval = interp(w, gplot)
            pval[0] = 1.
            pval = np.nan_to_num(pval, nan=0.)
            pvals[m][sc] = list(zip(gplot, pval))

    return pvals

def draw_natural(onames, points, directories, otags, labels, xaxis, yaxis, interpolate, onepoi, drawband, observed, formal, cmsapp, luminosity, a343bkg, transparent):

    pvals = []
    for tagdirs, tagpnts, otag in zip(directories, points, otags):
        print(f"Interpolating limit for {otag}")
        pval = interpolate_limit(parity, tagdirs, tagpnts, otag, observed=observed)
        pvals.append(pval)
    
    draw_1D(onames, pvals, labels, xaxis, yaxis, " (median expected)", None,
            parse_point_quantile(interpolate), drawband, observed, formal, cmsapp, luminosity, a343bkg, transparent)

    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--tag", help = "input tag-output-tag pairs to search. the pairs are semicolon separated, and tags colon-separated, "
                        "so e.g. when there are 2 tags: 't1:o1;t2:o2...", dest = "itag", default = "", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))
    parser.add_argument("--plot-tag", help = "extra tag to append to plot names", dest = "ptag", default = "", required = False, type = prepend_if_not_empty)
    parser.add_argument("--odir", help = "output directory to dump plots in", default = ".", required = False, type = remove_spaces_quotes)
    parser.add_argument("--label", help = "labels to attach on plot for each input tags, semicolon separated", default = "", required = False, type = lambda s: tokenize_to_list(s, ';' ))
    parser.add_argument("--drop",
                        help = "comma separated list of points to be dropped. 'XX, YY' means all points containing XX or YY are dropped.",
                        default = "", required = False, type = lambda s: [] if s == "" else tokenize_to_list( remove_spaces_quotes(s) ) )
    parser.add_argument("--keep",
                        help = "comma separated list of points to be kept. 'XX, YY' means all points containing XX or YY are kept.",
                        default = "", required = False, type = lambda s: [] if s == "" else tokenize_to_list( remove_spaces_quotes(s) ) )

    parser.add_argument("--observed", help = "draw observed limits as well", dest = "observed", action = "store_true", required = False)
    parser.add_argument("--interpolate",
                        help = "semicolon-separated point: quantiles to linearly interpolate based on neighboring points.\n"
                        "e.g. A_m425_w10p0: exp-1,exp0 to interpolate these two quantiles"
                        "the neighboring points are based on --function.",
                        default = "", required = False, type = lambda s: [] if s == "" else tokenize_to_list( remove_spaces_quotes(s), token = ';' ) )
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
    if len(args.keep) > 0:
        adir = [[pnt for pnt in l if any(k in pnt for k in args.keep)] for l in adir]
        hdir = [[pnt for pnt in l if any(k in pnt for k in args.keep)] for l in hdir]
    otags = [tag.split(':')[1] if len(tag.split(':')) > 1 else tag.split(':')[0] for tag in args.itag]

    apnt = [[get_point(tokenize_to_list(pnt, '/')[-1]) for pnt in directory] for directory in adir]
    hpnt = [[get_point(tokenize_to_list(pnt, '/')[-1]) for pnt in directory] for directory in hdir]

    for parity, dirs, pnts in zip(["A", "H"], [adir, hdir], [apnt, hpnt]):
        draw_natural(["{ooo}/{par}_limit_natural_{tag}{f}".format(
            par = parity, ooo = args.odir, tag = args.ptag, f = f) for f in args.fmt],
                    pnts, dirs, otags, args.label, axes["mass"] % parity, axes["ttcoupling"] % parity,
                    args.interpolate, False, args.drawband, args.observed, args.formal, args.cmsapp, args.luminosity, args.a343bkg, args.transparent)