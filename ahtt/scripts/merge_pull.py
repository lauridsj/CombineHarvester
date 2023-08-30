#!/usr/bin/env python
# merge the pull jsons for use by HIG-style plotImpacts.py
# find . -type f -name '*impact*all.json' | xargs -I % sh -c 'fname="$(basename %)"; echo %; plotImpacts.py -i % -o "impact/${fname//.json/}"; echo;'
# find . -type f -name '*all.pdf' | xargs -I % bash -c 'fname="%"; inkscape --export-filename=${fname//.pdf/}.png --export-dpi=300 ${fname} --pdf-page=1'
# find . -type f -name '*expth.pdf' | xargs -I % bash -c 'fname="%"; gs -dSAFER -dUseCropBox -r200 -sDEVICE=pngalpha -o ${fname//.pdf/.png} ${fname}'

from argparse import ArgumentParser
import os
import sys

import glob
from collections import OrderedDict
import json

from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes

def dump_pull(directories, onepoi, gvalue, rvalue, fixpoi, nuisances, otag):
    for directory, tag in directories:
        pulls = OrderedDict()

        for group in ["expth", "mcstat"]:
            impacts = glob.glob("{dcd}/{pnt}_{tag}_impacts_{mod}{gvl}{rvl}{fix}_{grp}_*.json".format(
                dcd = directory,
                tag = tag,
                pnt = '_'.join(directory.split('_')[:3]),
                mod = "one-poi" if onepoi else "g-scan",
                gvl = "_g_" + str(gvalue).replace(".", "p") if gvalue >= 0. else "",
                rvl = "_r_" + str(rvalue).replace(".", "p") if rvalue >= 0. and not onepoi else "",
                fix = "_fixed" if fixpoi and (gvalue >= 0. or rvalue >= 0.) else "",
                grp = group
            ))

            for ii, imp in enumerate(impacts):
                print("merge_pull :: opening file " + imp)
                with open(imp) as ff:
                    result = json.load(ff)

                names = None
                if len(nuisances) > 0:
                    names = [param["name"] for param in result["params"] for nuisance in nuisances if nuisance in param["name"]]
                    names = set(names)
                params = [param for param in result["params"] if names is None or param["name"] in names]

                if not pulls:
                    pulls["POIs"] = result["POIs"]
                    pulls["method"] = result["method"]
                    pulls["params"] = params
                else:
                    if pulls["POIs"] != result["POIs"]:
                        print("merge_pull :: WARNING :: incompatible POI best fit between input jsons!!")

                    pulls["params"] += params

        with open("{dcd}/{pnt}_{tag}_impacts_{mod}{gvl}{rvl}{fix}_{out}.json".format(
                dcd = directory,
                tag = tag,
                pnt = '_'.join(directory.split('_')[:3]),
                mod = "one-poi" if onepoi else "g-scan",
                gvl = "_g_" + str(gvalue).replace(".", "p") if gvalue >= 0. else "",
                rvl = "_r_" + str(rvalue).replace(".", "p") if rvalue >= 0. and not onepoi else "",
                fix = "_fixed" if fixpoi and (gvalue >= 0. or rvalue >= 0.) else "",
                out = otag
        ), "w") as jj:
            json.dump(pulls, jj, indent = 1)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--point", help = "signal point to plot the pulls of", default = "", required = True, type = remove_spaces_quotes)
    parser.add_argument("--tag", help = "input tag-output-tag pairs to search. the pairs are semicolon separated, and tags colon-separated, "
                        "so e.g. when there are 2 tags: 't1:o1;t2:o2...", dest = "itag", default = "", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' ))
    parser.add_argument("--one-poi", help = "plot pulls obtained with the g-only model", dest = "onepoi", action = "store_true", required = False)

    parser.add_argument("--nuisances", help = "comma-separated list of nuisances to consider. greedy matching: XX,YY means any nuisances containing "
                        "either XX or YY in the name is included.",
                        dest = "nuisance", default = "", required = False, type = lambda s: [] if s == "" else tokenize_to_list( remove_spaces_quotes(s) ))
    parser.add_argument("--impact-tag", help = "tag to attach to the merged impact json file",
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

    args = parser.parse_args()
    dirs = [[args.point + '_' + tag.split(':')[0], tag.split(':')[1] if len(tag.split(':')) > 1 else tag.split(':')[0]] for tag in args.itag]
    dump_pull(dirs, args.onepoi, args.setg, args.setr, args.fixpoi, args.nuisance, args.otag)
