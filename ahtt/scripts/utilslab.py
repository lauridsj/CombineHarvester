#!/usr/bin/env python
# utilities containing functions used throughout - lab-specific file ie those pertaining to lxplus/desy differences

import os
import platform
from utilspy import syscall

cluster = None
if "desy" in platform.node():
    cluster = "naf"
elif "cern" in platform.node():
    cluster = "lxplus"
else:
    raise NotImplementedError("unknown cluster! can't provide a default input storage base!")

def input_storage_base_directory():
    if cluster == "naf":
        return "/nfs/dust/cms/group/exotica-desy/HeavyHiggs/"
    if cluster == "lxplus":
        return "/eos/cms/store/user/afiqaize/"

input_base = input_storage_base_directory()
condordir = "/nfs/dust/cms/user/afiqaize/cms/sft/condor/" if "desy" in input_base else "/afs/cern.ch/work/a/afiqaize/public/randomThings/misc/condor/"
condorsub = condordir + "condorSubmit.sh"
condorpar = condordir + "condorParam.txt" if "desy" in input_base else condordir + "condorParam_lxpCombine.txt"
condorrun = condordir + "condorRun.sh" if "desy" in input_base else condordir + "condorRun_lxpCombine.sh"

kfactor_file_name = {
    171: input_base + "ahtt_kfactor_sushi/ulkfactor_final_mt171p5_230329.root",
    172: input_base + "ahtt_kfactor_sushi/ulkfactor_final_220129.root",
    173: input_base + "ahtt_kfactor_sushi/ulkfactor_final_mt173p5_230329.root"
}

def input_bkg(background, channels):
    if background != "":
        return background

    backgrounds = []
    if any(cc in channels for cc in ["ee", "em", "mm"]):
        backgrounds.append(input_base + "templates_ULFR2/dyscales_fix_231020/ll/bkg_ll_3D-33_rate_mtuX_pca_ewk.root")
    if any(cc in channels for cc in ["e3j", "e4pj", "m3j", "m4pj"]):
        backgrounds.append(input_base + "templates_ULFR2/dyscales_fix_231020/lj/templates_lj_bkg_rate_mtuX_pca_ewk.root")

    return ','.join(backgrounds)

def input_sig(signal, points, injects, channels, years):
    if signal != "":
        return signal

    signals = []
    masses = ["m" + str(im) for im in [365, 380] + range(400, 1001, 25)]
    for im in masses:
        if im in points or im in injects:
            if any(cc in channels for cc in ["ee", "em", "mm"]):
                signals.append(input_base + "templates_ULFR2/breaktype3_mtAH_230314/ll/sig_ll_3D-33_" + im + ".root")
            if any(cc in channels for cc in ["e3j", "e4pj", "m3j", "m4pj"]):
                signals.append(input_base + "templates_ULFR2/breaktype3_mtAH_230314/lj/templates_lj_sig_" + im + ".root")

    return ','.join(signals)

def remove_mjf():
    # in lxplus the file return output also gives an unneeded dir
    if "desy" not in input_base:
        syscall("rm -r mjf-{user}".format(user = os.environ.get('USER')), False, True)
