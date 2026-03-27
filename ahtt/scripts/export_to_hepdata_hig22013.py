import hepdata_lib as hepd
import json
from glob import glob
import os
from tqdm import tqdm
import numpy as np
import uproot

submission = hepd.Submission()

common_keywords = {
    "reactions": ["P P --> TOP TOPBAR"],
    "cmenergies": [13000.],
    "phrases": ["CMS", "physics", "top", "higgs", "BSM", "pseudoscalar", "scalar"]
}

kfactor_file = '/data/dust/group/cms/exotica-desy/HeavyHiggs/ahtt_kfactor_sushi/ulkfactor_sushi_mt172p5_241023.root'

with uproot.open(kfactor_file) as rf:
    for parity in ['A', 'H']:
        for part in ["res", "int"]:
            kfactorjson = rf[f"{parity}_{part}_sushi_nnlo_mg5_lo_kfactor_pdf_325500_nominal"].tojson()
            m_axis = kfactorjson['fX']
            w_axis = kfactorjson['fY']
            vals = kfactorjson['fZ']

            inds = [i for i in range(len(vals)) if m_axis[i] != 343. and w_axis[i] == 5.]
            m_axis = [m_axis[i] for i in inds]
            vals = [vals[i] for i in inds]

            table = hepd.Table(f"signal_kfactors_{parity}_{part}")
            if part == "res":
                table.description = f"LO-to-NNLO K-factors for the {parity} resonance signals, as a function of mass."
            else:
                table.description = f"LO-to-NNLO K-factors for the {parity}-SM interference signals, as a function of mass."
            table.location = "Supplemental material"
            table.keywords = {**common_keywords}

            var = hepd.Variable(f"$m_{parity}$", is_independent=True, is_binned=False, units="GeV")
            var.values = m_axis
            table.add_variable(var)

            #var = hepd.Variable(f"$\\Gamma_{parity}/m_{parity}$", is_independent=True, is_binned=False, units="%")
            #var.values = w_axis
            #table.add_variable(var)

            var = hepd.Variable(f"K-factor", is_independent=False, is_binned=False, units=None)
            var.values = vals
            table.add_variable(var)

            submission.add_table(table)

limitfiles = glob("/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/hig22013_limittxts/*.json")
files_withetat = [f for f in limitfiles if "withetat" in f]
files_noetat = [f for f in limitfiles if "withetat" not in f]

widths_paper = [1.0, 2.0, 5.0, 10.0, 18.0, 25.0]

for files, tag in [(files_noetat, "noetat"), (files_withetat, "withetat")]:
    for parity in ["A", "H"]:
        for width in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 10.0, 13.0, 15.0, 18.0, 21.0, 25.0]:
            wstr = f"w{width:.1f}".replace(".", "p")
            file = [f for f in files if f.split("/")[-1].startswith(parity) and wstr in f][0]
            print(tag, parity, wstr, file)
            with open(file) as jf:
                j = json.load(jf)
            
            table = hepd.Table(f"limit_{parity}_{wstr}_{tag}")
            coupling = f"g_{{{parity} t \\bar t}}"
            desc = f"Exclusion limits on the coupling modifier ${coupling}$ at 95% CL for the {parity} boson with {width:.1f}% width, as a function of the {parity} boson mass."
            if tag == "noetat":
                desc += " No contribution from $t \\bar{t}$ bound states is included in the background."
            else:
                desc += " An $\\eta_t$ contribution is added to the background."
            table.description = desc
            if width in widths_paper:
                if tag == "noetat":
                    fignumber = 10 if parity == "A" else 11
                else:
                    fignumber = 12 if parity == "A" else 13
                table.location = f"Figure {fignumber} in the paper"
            else:
                table.location = "Supplemental material"
            table.keywords = {**common_keywords}

            var = hepd.Variable(f"$m_{parity}$", is_independent=True, is_binned=False, units="GeV")
            var.values = j["mass"]
            table.add_variable(var)

            obs = hepd.Variable(f"Observed upper limit on ${coupling}$ at 95% CL", is_independent=False, is_binned=False, units=None)
            obs.values = j["obs"]
            obs.add_qualifier("Limit", "Observed")
            obs.add_qualifier("SQRT(S)", 13, "TeV")
            obs.add_qualifier("LUMINOSITY", 138, "fb$^{-1}$")
            table.add_variable(obs)

            exp = hepd.Variable(f"Expected upper limit on ${coupling}$ at 95% CL", is_independent=False, is_binned=False, units=None)
            exp.values = j["exp0"]
            exp.add_qualifier("Limit", "Expected")
            exp.add_qualifier("SQRT(S)", 13, "TeV")
            exp.add_qualifier("LUMINOSITY", 138, "fb$^{-1}$")

            # +/- 1 sigma
            unc_1s = hepd.Uncertainty("1 s.d.", is_symmetric=False)
            unc_1s.set_values_from_intervals(zip(j["exp-1"], j["exp+1"]), nominal=exp.values)
            exp.add_uncertainty(unc_1s)

            # +/- 2 sigma
            unc_2s = hepd.Uncertainty("2 s.d.", is_symmetric=False)
            unc_2s.set_values_from_intervals(zip(j["exp-2"], j["exp+2"]), nominal=exp.values)
            exp.add_uncertainty(unc_2s)

            table.add_variable(exp)

            submission.add_table(table)


#print("writing output")
#submission.create_files("hig22013_hepdata",remove_old=True)

#breakpoint()

folders = [
    "/data/dust/user/lauridsj/ah/hepdata_nllscan/CMSSW_10_2_13/src/CombineHarvester/ahtt/plots",
    '/data/dust/user/lauridsj/ah/hepdata_nllscan/CMSSW_10_2_13/src/CombineHarvester/ahtt/plots_sam',
    #"/data/dust/user/bachjoer/fc-scan/afiqs_ch/CMSSW_10_2_13/src/CombineHarvester/ahtt/scripts/jsons_nll",
    "/data/dust/user/baxtersa/CMSSW_10_2_13/src/CombineHarvester/ahtt/contours_nll",
    "/data/dust/user/baxtersa/CMSSW_10_2_13/src/CombineHarvester/ahtt/contours_nll/w2p0",
    "/data/dust/user/baxtersa/CMSSW_10_2_13/src/CombineHarvester/ahtt/contours_nll/w5p0",
    "/data/dust/user/baxtersa/CMSSW_10_2_13/src/CombineHarvester/ahtt/contours_nll/w21p0",
    "/data/dust/user/bachjoer/fc-scan/afiqs_ch/CMSSW_10_2_13/src/CombineHarvester/ahtt/scripts/plots_2dnll_mixed",
    "/data/dust/user/bachjoer/fc-scan/afiqs_ch/CMSSW_10_2_13/src/CombineHarvester/ahtt/scripts/plots_2dnll"
]





print("globbing json files...")

alljsons = []
basenames = []
for folder in folders:
    jinf = glob(folder + "/*obs.json")
    print(folder, len(jinf))
    for f in jinf:
        basename = os.path.basename(f)
        if basename in basenames:
            print(f"WARNING: {f} already collected, skipping...")
            continue
        alljsons.append(f)
        basenames.append(basename)

alljsons.sort(key=lambda path: os.path.basename(path))

for jsonpath in tqdm(alljsons):
    ahpnt = os.path.basename(jsonpath).split("_nll")[0]
    ahpntsplit = ahpnt.split("__")
    mA = int(ahpntsplit[0].split("_")[1][1:])
    mH = int(ahpntsplit[1].split("_")[1][1:])
    wA = float(ahpntsplit[0].split("_")[2][1:].replace('p','.'))
    wH = float(ahpntsplit[1].split("_")[2][1:].replace('p','.'))
    #print(f"Processing A, {mA} GeV, {wA}%; H, {mH} GeV, {wH}%")

    with open(jsonpath) as f:
        ahjson = json.load(f)

    #gA = []
    #gH = []
    gs = []
    twodnll = []
    for key, val in ahjson["2dNLLs"].items():
        key = key.split(",")

        gA = round(float(key[0].strip()), 3)
        gH = round(float(key[1].strip()), 3)
        if (gA,gH) in gs:
            dnllother = twodnll[gs.index((gA,gH))]
            if abs(val - dnllother) > 0.01:
                print(f"WARNING: {jsonpath}: Point {gA}, {gH} is present twice with different dnll!")
                print(f"2dNLLs are: {twodnll[gs.index((gA,gH))]}, {val}")
            continue
        gs.append((gA,gH))
        twodnll.append(val)

    gA = [gg[0] for gg in gs]
    gH = [gg[1] for gg in gs] 

    #ind00 = [i for i in range(len(twodnll)) if gA[i] == 0. and gH[i] == 0.]
    dist = [gA[i] + gH[i] for i in range(len(twodnll))]
    ind00 = np.argmin(dist)
    #if len(ind00) != 1:
    #    raise ValueError(f"ERROR: found {len(ind00)} 0 points for file {jsonpath}")
    if dist[ind00] > 0.:
        print(f"WARNING: {jsonpath} has lowest point at gA={gA[ind00]} and gH={gH[ind00]}")
    twodnll_00 = twodnll[ind00]
    twodnll = [v - twodnll_00 for v in twodnll]

    table = hepd.Table(ahpnt + "_2dnll")
    table.description = "Observed values of twice the negative log-likelihood with respect to the SM (corresponding to $g_{A t \\bar t} = g_{H t \\bar t} = 0$) " + f"for the simultaneous presence of A, $m_A = {mA:.0f}$ GeV, $\\Gamma_A/m_A = {wA:.1f}$% and H, $m_H = {mH:.0f}$ GeV, $\\Gamma_H/m_H = {wH:.1f}$% " + "as a function of the coupling modifiers $g_{A t \\bar t}$ and $g_{H t \\bar t}$."
    table.location = "Supplemental material"
    table.keywords = {**common_keywords}

    var = hepd.Variable("$g_{A t \\bar t}$", is_independent=True, is_binned=False, units=None)
    var.values = gA
    table.add_variable(var)

    var = hepd.Variable("$g_{H t \\bar t}$", is_independent=True, is_binned=False, units=None)
    var.values = gH
    table.add_variable(var)

    var = hepd.Variable("-2dNLL", is_independent=False, is_binned=False, units=None)
    var.values = twodnll
    table.add_variable(var)

    submission.add_table(table)

print("writing output")
submission.create_files("hig22013_hepdata",remove_old=True)
