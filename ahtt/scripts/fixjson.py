#!/usr/bin/env python
# 2023/04/20 twin_point_ahtt::sum_up was buggy in that it does not propagate the dnll
# this script is written to salvage the jsons so created, by reading those fields from a previous json
# assumed to have been produced by --fc-mode add/brim
# usage: in the directory A_mxx_wyy__H_... with affected jsons, run this script
# output: new json files with index + 1, same pass/fail as the latest json for each exp, and also an extra dnll in each g pair

import glob
import json
from collections import OrderedDict

if __name__ == '__main__':
    for exp in ["exp-b", "exp-s", "exp-01", "exp-10", "obs"]:
        ggrid = glob.glob("*_fc-scan{exp}_*.json".format(exp = "_" + exp))
        ggrid.sort()
        idx1 = 0 if len(ggrid) == 0 else int(ggrid[-1].split("_")[-1].split(".")[0])
        idx = idx1 + 1

        if len(ggrid) < 2:
            continue

        with open(ggrid[-1]) as ff1:
            last1 = json.load(ff1, object_pairs_hook = OrderedDict)

            with open(ggrid[-2]) as ff2:
                last2 = json.load(ff2, object_pairs_hook = OrderedDict)
                
                for gv in last1['g-grid']:
                    last1['g-grid'][gv]['dnll'] = last2['g-grid'][gv]['dnll']

                last1['g-grid'] = OrderedDict(sorted(last1["g-grid"].items()))
                with open("{jjj}".format(jjj = ggrid[-1].replace("_" + str(idx1), "_" + str(idx))), "w") as jj:
                    json.dump(last1, jj, indent = 1)
