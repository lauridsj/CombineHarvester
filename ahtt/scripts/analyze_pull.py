#!/usr/bin/env python
# go through the pull json, and report parameters with large difference (in std dev sense) between fits

import sys
import glob
import json
import math
from collections import OrderedDict
from argparse import ArgumentParser
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes

def measurement_format(central, upper, lower):
    vlog = int(abs(math.floor(min(math.log10(upper) - 1, math.log10(lower) - 1))))
    return "{cc} +{uu} -{ll}".format(cc = round(central, vlog), uu = round(upper, vlog), ll = round(lower, vlog))

def names_and_values(results, key):
    values = []
    allnames = {param["name"] for result in results for param in result[key]}

    for name in allnames:
        value = {
            "central": [],
            "upper": [],
            "lower": [],
        }

        for result in results:
            names = [param["name"] for param in result[key]]
            idx = names.index(name) if name in names else None
            central = None if idx is None else result[key][idx]["fit"][1]
            upper = None if idx is None else abs(result[key][idx]["fit"][2] - central)
            lower = None if idx is None else abs(result[key][idx]["fit"][0] - central)

            value["central"].append(central)
            value["upper"].append(upper)
            value["lower"].append(lower)

        values.append(value)

    return [allnames, values]

def report_discrepancy_wrt_reference(directories, parameters, always_print = False, threshold = 3):
    tags = ["/".join(directory) for directory in directories]
    names, values = parameters

    for nn, name in enumerate(names):
        for i0, tag0 in enumerate(tags):
            v0 = values[nn]["central"][i0]
            u0 = values[nn]["upper"][i0]
            l0 = values[nn]["lower"][i0]

            if None not in [v0, u0, l0]:
                large = (v0 < 0. and abs(v0) / u0 > threshold) or (v0 > 0. and abs(v0) / l0 > threshold)
                if always_print or large:
                    print("analyze_pull :: {pp} in tag {t0} = {m0} deviates by {th} sigma from 0.".format(
                        pp = name,
                        t0 = tag0,
                        m0 = measurement_format(v0, u0, l0),
                        th = threshold,
                    ))

                for i1, tag1 in enumerate(tags):
                    if i1 != i0:
                        v1 = values[nn]["central"][i1]
                        u1 = values[nn]["upper"][i1]
                        l1 = values[nn]["lower"][i1]

                        if None not in [v1, u1, l1]:
                            discrepant = (v1 > v0 and abs(v1 - v0) / threshold > u0) or (v1 < v0 and abs(v1 - v0) / threshold > l0)
                            if always_print or discrepant:
                                print("analyze_pull :: {pp} with tag {t1} = {m1} differ by {th} sigma wrt tag {t0} = {m0}".format(
                                    pp = name,
                                    t1 = tag1,
                                    m1 = measurement_format(v1, u1, l1),
                                    th = threshold,
                                    t0 = tag0,
                                    m0 = measurement_format(v0, u0, l0),
                                ))

def analyze(directories, onepoi, gvalue, rvalue, fixpoi, otag, always_print_poi = False):
    results = []
    for directory, tag in directories:
        iname = "{dcd}/{pnt}_{tag}_impacts_{mod}{gvl}{rvl}{fix}_{otg}.json".format(
            dcd = directory,
            tag = tag,
            pnt = '_'.join(directory.split('_')[:3]),
            mod = "one-poi" if onepoi else "g-scan",
            gvl = "_g_" + str(gvalue).replace(".", "p") if gvalue >= 0. else "",
            rvl = "_r_" + str(rvalue).replace(".", "p") if rvalue >= 0. and not onepoi else "",
            fix = "_fixed" if fixpoi and (gvalue >= 0. or rvalue >= 0.) else "",
            otg = otag
        )
        with open(iname) as ii:
            result = json.load(ii)
            results.append(result)

    print("analyze_pull :: checking POIs")
    pois = names_and_values(results, "POIs")
    report_discrepancy_wrt_reference(directories, pois, always_print_poi)
    sys.stdout.flush()
    print("\n\nanalyze_pull :: checking NPs")
    nuisances = names_and_values(results, "params")
    report_discrepancy_wrt_reference(directories, nuisances)
    sys.stdout.flush()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "signal point to plot the pulls of", default = "", required = True, type = remove_spaces_quotes)
    parser.add_argument("--tag", help = "input tag-output-tag pairs to search. the pairs are semicolon separated, and tags colon-separated, "
                        "so e.g. when there are 2 tags: 't1:o1;t2:o2...", dest = "itag", default = "", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))
    parser.add_argument("--one-poi", help = "analyze pulls obtained with the g-only model", dest = "onepoi", action = "store_true", required = False)

    parser.add_argument("--impact-tag", help = "tag assigned to the merged impact json file",
                        dest = "otag", default = "all", required = False)

    parser.add_argument("--g-value",
                        help = "g to use when evaluating impacts/fit diagnostics/nll. "
                        "does NOT freeze the value, unless --fix-poi is also used. "
                        "note: semantically sets value of 'r' with --one-poi, as despite the name it plays the role of g.",
                        dest = "setg", default = "-1.", required = False, type = lambda s: float(remove_spaces_quotes(s)))
    parser.add_argument("--r-value",
                        help = "r to use when evaluating impacts/fit diagnostics/nll, if --one-poi is not used."
                        "does NOT freeze the value, unless --fix-poi is also used.",
                        dest = "setr", default = "-1.", required = False, type = lambda s: float(remove_spaces_quotes(s)))
    parser.add_argument("--fix-poi", help = "fix pois in the fit, through --g-value and/or --r-value",
                        dest = "fixpoi", action = "store_true", required = False)

    parser.add_argument("--always-print-poi", help = "always print poi value, regardless of deviation",
                        dest = "alwayspoi", action = "store_true", required = False)

    args = parser.parse_args()
    dirs = [[args.point + '_' + tag.split(':')[0], tag.split(':')[1] if len(tag.split(':')) > 1 else tag.split(':')[0]] for tag in args.itag]
    analyze(dirs, args.onepoi, args.setg, args.setr, args.fixpoi, args.otag, args.alwayspoi)
