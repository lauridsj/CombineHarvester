#!/usr/bin/env python
# makes a datacard off the requested signal point
# assumes that both background and signal root files are of a certain structure
# for nuisances that are smoothed, a histogram containing fit quality information is expected
# which is used to determine if they are to be dropped/assigned as pseudo-lnN sources instead

from argparse import ArgumentParser
import os
import sys
import math

import glob
from datetime import datetime
from collections import OrderedDict

from ROOT import TFile, gDirectory, TH1, TH1D
TH1.AddDirectory(False)
TH1.SetDefaultSumw2(True)

from numpy import random as rng
import CombineHarvester.CombineTools.ch as ch

from utilities import kfactor_file_name, syscall, get_point, flat_reldev_wrt_nominal, scale, zero_out, project, add_scaled_nuisance
from desalinator import prepend_if_not_empty, tokenize_to_list, remove_spaces_quotes
from argumentative import common_point, common_common, make_datacard_pure
from hilfemir import combine_help_messages

scale_choices = ["nominal", "uF_up", "uF_down", "uR_up", "uR_down"]
def get_kfactor(sigpnt):
    kfile = TFile.Open(kfactor_file_name, "read")
    khist = [
        (kfile.Get(sigpnt[0] + "_res_sushi_nnlo_mg5_lo_kfactor_pdf_325500_" + syst),
         kfile.Get(sigpnt[0] + "_int_sushi_nnlo_mg5_lo_kfactor_pdf_325500_" + syst))
        for syst in scale_choices
    ]
    kvals = tuple([(syst[0].Interpolate(sigpnt[1], sigpnt[2]), syst[1].Interpolate(sigpnt[1], sigpnt[2])) for syst in khist])
    kfile.Close()

    return kvals

def get_lo_ratio(sigpnt, channel):
    kfile = TFile.Open(kfactor_file_name, "read")
    xhist = [
        (kfile.Get(sigpnt[0] + "_res_mg5_pdf_325500_scale_dyn_0p5mtt_" + syst + "_xsec_" + channel),
         kfile.Get(sigpnt[0] + "_int_mg5_pdf_325500_scale_dyn_0p5mtt_" + syst + "_xabs_" + channel),
         kfile.Get(sigpnt[0] + "_int_mg5_pdf_325500_scale_dyn_0p5mtt_" + syst + "_positive_event_fraction_" + channel))
        for syst in scale_choices
    ]

    rvals = [[syst[0].Interpolate(sigpnt[1], sigpnt[2]),
              syst[1].Interpolate(sigpnt[1], sigpnt[2]) * syst[2].Interpolate(sigpnt[1], sigpnt[2]),
              syst[1].Interpolate(sigpnt[1], sigpnt[2]) * (1. - syst[2].Interpolate(sigpnt[1], sigpnt[2]))] for syst in xhist]

    rvals = tuple([[r[0] / rvals[0][0], r[1] / rvals[0][1], r[2] / rvals[0][2]] for r in rvals])
    kfile.Close()

    # ratio of [res, pos, neg] xsecs, syst / nominal, in the syst ordering above
    return rvals

# assumption is some specific list is fully correlated, others fully uncorrelated
original_nominal = {}
def read_category_process_nuisance(ofile, inames, channel, year, cpn, pseudodata, replaces, drops, keeps, alwaysshape, threshold, lnNsmall,
                                   projection_rule = "", sigpnt = None, kfactor = False):
    # because afiq hates seeing jets spelled outside of text
    if not hasattr(read_category_process_nuisance, "aliases"):
        read_category_process_nuisance.aliases = OrderedDict([
            ("e3j" , "e3jets"),
            ("e4pj", "e4pjets"),
            ("m3j" , "mu3jets"),
            ("m4pj", "mu4pjets"),
        ])

    # to note nuisances that need special handling
    # 'regular' nuisances are those that are uncorrelated between years with a scaling of 1
    if not hasattr(read_category_process_nuisance, "specials"):
        read_category_process_nuisance.specials = OrderedDict([
            ("QCDscale_MEFac_AH",                  (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("QCDscale_MERen_AH",                  (("2016pre", "2016post", "2017", "2018"), 1.)),

            ("QCDscale_ISR_AH",                    (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("QCDscale_FSR_AH",                    (("2016pre", "2016post", "2017", "2018"), 1.)),

            ("QCDscale_MEFac_TT",                  (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("QCDscale_MERen_TT",                  (("2016pre", "2016post", "2017", "2018"), 1.)),

            ("EWK_yukawa",                         (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("EWK_scheme",                         (("2016pre", "2016post", "2017", "2018"), 1.)),

            ("CMS_PDF_alphaS",                     (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_PDF_hessian",                    (("2016pre", "2016post", "2017", "2018"), 1.)),

            ("QCDscale_ISR_TT",                    (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("QCDscale_FSR_TT",                    (("2016pre", "2016post", "2017", "2018"), 1.)),

            ("hdamp_TT",                           (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("tmass_3GeV_TT",                      (("2016pre", "2016post", "2017", "2018"), 1. / 3.)), # gaussian 1 GeV
            #("tmass_3GeV_TT",                      (("2016pre", "2016post", "2017", "2018"), ("shapeU", 1.))), # flat 3 GeV

            ("CMS_UEtune_13TeV",                   (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CR_ERD_TT",                          (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CR_QCD_TT",                          (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CR_Gluon_TT",                        (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_ColorRec_13TeV",                 (("2016pre", "2016post", "2017", "2018"), 1.)),

            ("CMS_pileup",                         (("2016pre", "2016post", "2017", "2018"), 1.)),

            ("CMS_eff_e_reco",                     (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_e_id",                       (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_trigger_e",                  (("2016pre", "2016post", "2017", "2018"), 1.)),

            ("CMS_eff_m_reco",                     (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_m_id_syst",                  (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_m_iso_syst",                 (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_trigger_m_syst",             (("2016pre", "2016post", "2017", "2018"), 1.)),

            ("CMS_fake_b_13TeV",                   (("2016pre", "2016post", "2017", "2018"), 1.)),

            #("CMS_eff_b_13TeV_JEC",                      (("2016pre", "2016post", "2017", "2018"), 1.)),
            #("CMS_eff_b_13TeV_Pileup",                   (("2016pre", "2016post", "2017", "2018"), 1.)),
            #("CMS_eff_b_13TeV_Type3",                    (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_b_13TeV_GluonSplitting",           (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_b_13TeV_BottomFragmentation",      (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_b_13TeV_BottomTemplateCorrection", (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_b_13TeV_CharmFragmentation",       (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_b_13TeV_CharmTemplateCorrection",  (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_b_13TeV_CharmToMuonBR",            (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_b_13TeV_LightCharmRatio",          (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_b_13TeV_VzeroParticles",           (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_b_13TeV_MuonRelativePt",           (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_b_13TeV_MuonPt",                   (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_b_13TeV_MuonDeltaR",               (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_b_13TeV_AwayJetTag",               (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_b_13TeV_JPCorrection",             (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_eff_b_13TeV_LifetimeOthers",           (("2016pre", "2016post", "2017", "2018"), 1.)),
            #("CMS_eff_b_13TeV",                          (("2016pre", "2016post", "2017", "2018"), 1.)),

            ("CMS_JEC_13TeV_Absolute",             (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_JEC_13TeV_BBEC1",                (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_JEC_13TeV_FlavorQCD",            (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_JEC_13TeV_FlavorPureBottom",     (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_JEC_13TeV_FlavorQCDOnNonBottom", (("2016pre", "2016post", "2017", "2018"), 1.)),
            ("CMS_JEC_13TeV_RelativeBal",          (("2016pre", "2016post", "2017", "2018"), 1.)),
        ])

        for ipdf in range(100):
            read_category_process_nuisance.specials["CMS_PDF_PCA_" + str(ipdf)] = (("2016pre", "2016post", "2017", "2018"), 1.)

        for c1, c2 in read_category_process_nuisance.aliases.items():
            read_category_process_nuisance.specials[c2 + '_shape_EWQCD'] = (("2016pre", "2016post", "2017", "2018"), 1.)

    processes = []
    nuisances = []

    odir = channel + '_' + year
    idir = odir.replace(channel, read_category_process_nuisance.aliases[channel]) if channel in read_category_process_nuisance.aliases else odir
    if not bool(ofile.GetDirectory(odir)):
        ofile.mkdir(odir)

    ifile = None
    kfactors = None
    loratios = None

    if sigpnt == None:
        fname = ""
        for iname in inames:
            ifile = TFile.Open(iname, "read")
            dirs = [key.GetName() for key in ifile.GetListOfKeys()]
            if idir in dirs:
                fname = iname
                break

            ifile.Close()
            ifile = None

        ifile.cd(idir)
        keys = gDirectory.GetListOfKeys()

        for key in keys:
            kname = key.GetName()

            if not kname.endswith("Up") and not kname.endswith("Down") and not kname.endswith("_chi2"):
                if kname != "data_obs" or not pseudodata:
                    processes.append((kname, fname))
                    ofile.cd(odir)

                    hn = key.ReadObj()
                    if projection_rule != "":
                        hn = project(hn, projection_rule)

                    zero_out(hn)
                    hn.Write()

                    if replaces is not None:
                        if odir not in original_nominal:
                            original_nominal[odir] = {}
                        if kname not in original_nominal[odir]:
                            original_nominal[odir][kname] = hn.Clone(kname + '_original_no_bootleg_frfr')
                            original_nominal[odir][kname].SetDirectory(0)

        ifile.Close()
        ifile = None

    else:
        for sig in sigpnt:
            if kfactor:
                kfactors = get_kfactor(get_point(sig))

            fname = ""
            for iname in inames:
                ifile = TFile.Open(iname, "read")
                dirs = [key.GetName() for key in ifile.GetListOfKeys()]

                if idir in dirs:
                    ifile.cd(idir)
                    if any([key.GetName() == sig + "_res" for key in gDirectory.GetListOfKeys()]):
                        fname = iname
                        break

                ifile.Close()
                ifile = None

            for ss in ["_res", "_pos", "_neg"]:
                pnt = sig + ss
                processes.append((pnt, fname))

                hn = ifile.Get(idir + '/' + pnt)
                if projection_rule != "":
                    hn = project(hn, projection_rule)

                if kfactor and kfactors is not None:
                    if ss == "_res":
                        scale(hn, kfactors[0][0])
                    else:
                        scale(hn, kfactors[0][1])

                if "_neg" in ss:
                    scale(hn, -1.)
                else:
                    zero_out(hn)

                ofile.cd(odir)
                hn.Write()

                if replaces is not None:
                    if odir not in original_nominal:
                        original_nominal[odir] = {}
                    if pnt not in original_nominal[odir]:
                        original_nominal[odir][pnt] = hn.Clone(pnt + '_original_no_bootleg_frfr')
                        original_nominal[odir][pnt].SetDirectory(0)

            ifile.Close()
            ifile = None
            kfactors = None

    for pp, ff in processes:
        if not sigpnt == None:
            sig = "_".join(pp.split("_")[:-1])
            loratios = get_lo_ratio(get_point(sig), "ll") if channel in ["ee", "em", "mm", "ll"] else get_lo_ratio(get_point(sig), "lj")
            if kfactor:
                kfactors = get_kfactor(get_point(sig))

        ifile = TFile.Open(ff, "read")
        ifile.cd(idir)

        nuisance = []

        keys = gDirectory.GetListOfKeys()
        for key in keys:
            kname = key.GetName()

            if pp + '_' in kname and kname.endswith("Up"):
                nn1 = "".join(kname.rsplit("Up", 1)).replace(pp + '_', "", 1)
                if nn1 in read_category_process_nuisance.specials:
                    nn2 = nn1 if (year in read_category_process_nuisance.specials[nn1][0] or year in nn1) else nn1 + '_' + year

                    if "QCDscale_ME" in nn2 and "_AH" in nn2:
                        nn2 = nn2 + "_res" if "_res" in pp else nn2 + "_int"
                else:
                    nn2 = nn1 if year in nn1 else nn1 + '_' + year

                if channel in read_category_process_nuisance.aliases:
                    for c1, c2 in read_category_process_nuisance.aliases.items():
                        nn2 = nn2.replace(c2, c1)

                nuisance.append((nn2, read_category_process_nuisance.specials[nn1][1]) if nn1 in read_category_process_nuisance.specials else (nn2, 1.))

                hu = key.ReadObj()
                hd = ifile.Get(idir + '/' + "Down".join(kname.rsplit("Up", 1)))

                if projection_rule != "":
                    hu = project(hu, projection_rule)
                    hd = project(hd, projection_rule)

                hc = ifile.Get(idir + '/' + "_chi2".join(kname.rsplit("Up", 1))) if keys.Contains("_chi2".join(kname.rsplit("Up", 1))) else None

                drop_nuisance = False
                if keeps is not None:
                    drop_nuisance = not any([dn in nn2 for dn in keeps])
                if drops is not None:
                    drop_nuisance = drop_nuisance or drops == ['*'] or any([dn in nn2 for dn in drops])

                if not alwaysshape and hc is not None:
                    # the values are smooth chi2 up, down, flat chi2 up, down and flat values up, down
                    chi2s = [hc.GetBinContent(ii) for ii in range(1, 7)]

                    scaleu = 1.
                    scaled = 1.

                    hn = ifile.Get(idir + '/' + pp)
                    up_norm_rdev = (hu.Integral() - hn.Integral()) / hn.Integral()
                    down_norm_rdev = (hd.Integral() - hn.Integral()) / hn.Integral()
                    two_sided_smooth = up_norm_rdev / down_norm_rdev < 0.

                    # use smooth if it is two-sided and its chi2 is better than flat in both up and down
                    if not two_sided_smooth or chi2s[2] < chi2s[0] or chi2s[3] < chi2s[1]:
                        above_threshold = lnNsmall or (abs(chi2s[4]) > threshold and abs(chi2s[5]) > threshold)
                        threshold_is_sensible = (1. > abs(chi2s[4]) > 1e-5 and 1. > abs(chi2s[5]) > 1e-5)
                        two_sided_flat = chi2s[4] / chi2s[5] < 0.
                        keep_value = above_threshold and threshold_is_sensible and two_sided_flat

                        scaleu = chi2s[4] if keep_value else 0.
                        scaled = chi2s[5] if keep_value else 0.

                        flat_reldev_wrt_nominal(hu, ifile.Get(idir + '/' + pp), scaleu)
                        flat_reldev_wrt_nominal(hd, ifile.Get(idir + '/' + pp), scaled)

                        if scaleu == 0. and scaled == 0.:
                            drop_nuisance = True

                        if not drop_nuisance:
                            print("make_datacard :: " + str((pp, year, channel)) + " nuisance " + nn2 + " flattened with (up, down) scales of " + str((scaleu, scaled)))

                if not sigpnt == None:
                    # rescale varied LO to nominal
                    idxu = 1 if "_MEFac_" in nn2 else 3 if "_MERen_" in nn2 else 0
                    idxd = 2 if "_MEFac_" in nn2 else 4 if "_MERen_" in nn2 else 0
                    idxp = 0 if "_res" in pp else 1 if "_pos" in pp else 2
                    scale(hu, 1. / loratios[idxu][idxp])
                    scale(hd, 1. / loratios[idxd][idxp])

                    if kfactor and kfactors is not None:
                        # and then to varied NNLO
                        idxp = 0 if "_res" in pp else 1
                        scale(hu, kfactors[idxu][idxp])
                        scale(hd, kfactors[idxd][idxp])

                        # FIXME if instead the varied LO is desired, comment the loratios scaling above
                        # FIXME and use 0 instead of idxu/idxp for kfactors

                hu.SetName(hu.GetName().replace(nn1, nn2))
                hd.SetName(hd.GetName().replace(nn1, nn2))

                if "_neg" in pp:
                    scale(hu, -1.)
                    scale(hd, -1.)
                else:
                    zero_out(hu)
                    zero_out(hd)

                if replaces is not None:
                    for rn in replaces:
                        nds = rn.split(':') # name direction scale

                        if len(nds) > 1 and nds[0] == nn2:
                            hn = ofile.Get(odir + '/' + pp)
                            ho = original_nominal[odir][pp]
                            hv = hu if nds[1].lower() == "up" else hd

                            hr = add_scaled_nuisance(hv, hn, ho, 1. if len(nds) < 3 else float(nds[2]))
                            hr.SetName(pp)
                            ofile.cd(odir)
                            hr.Write()

                if drop_nuisance:
                    nuisance.pop()
                    print("make_datacard :: " + str((pp, year, channel)) + " nuisance " + nn2 + " has been dropped")
                    continue

                ofile.cd(odir)
                hu.Write()
                hd.Write()

        nuisances.append(nuisance)
        ifile.Close()
        ifile = None
        kfactors = None
        loratios = None

    if odir not in cpn:
        cpn[odir] = OrderedDict()

    for pp, nn in zip(processes, nuisances):
        cpn[odir][pp[0]] = nn

def make_pseudodata(ofile, cpn, sigpnt = None, seed = None):
    if seed is None or seed >= 0:
        rng.seed(seed)

    for category, processes in cpn.items():
        dd = None
        for pp in processes.keys():
            issig = any([ss in pp for ss in ["_res", "_pos", "_neg"]])
            hh = ofile.Get(category + '/' + pp).Clone("hhtmphh")
            if "_neg" in pp:
                scale(hh, -1.)

            if not issig or (issig and sigpnt is not None and any([s in pp for s in sigpnt])):
                if dd is None:
                    dd = hh.Clone("data_obs")
                else:
                    dd.Add(hh)

        for ii in range(1, dd.GetNbinsX() + 1):
            content = rng.poisson(dd.GetBinContent(ii)) if seed is not None and seed >= 0 else round(dd.GetBinContent(ii))
            dd.SetBinContent(ii, content)
            dd.SetBinError(ii, math.sqrt(content))

        ofile.cd(category)
        dd.Write()

def write_datacard(oname, cpn, years, sigpnt, injsig, drops, keeps, mcstat, rateparam, tag):
    # to note nuisances that need special handling
    # 'regular' nuisances are those that are uncorrelated between years with a scaling of 1
    if not hasattr(write_datacard, "lnNs"):
        # FIXME check again for UL
        write_datacard.lnNs = OrderedDict([
            (("2018",), (
                ("lumi_13TeV", ("2018",), "all", 1.025),
            )),
            (("2017",), (
                ("lumi_13TeV", ("2017",), "all", 1.023),
            )),
            (("2016post",), (
                ("lumi_13TeV", ("2016post",), "all", 1.012),
            )),
            (("2016pre",), (
                ("lumi_13TeV", ("2017pre",), "all", 1.012),
            )),
            (("2016post", "2016pre"), (
                ("lumi_13TeV_16",      ("2016pre", "2016post"), "all", 1.01),
                ("lumi_13TeV_correlated", ("2016pre", "2016post"), "all", 1.006),
            )),
            (("2017", "2018"), (
                ("lumi_13TeV_17",         ("2017",), "all", 1.02),
                ("lumi_13TeV_18",         ("2018",), "all", 1.015),
                ("lumi_13TeV_1718",       ("2017",), "all", 1.006),
                ("lumi_13TeV_1718",       ("2018",), "all", 1.002),
                ("lumi_13TeV_correlated", ("2017",), "all", 1.009),
                ("lumi_13TeV_correlated", ("2018",), "all", 1.02),
            )),
            (("2016post", "2016pre", "2017", "2018"), (
                # single scheme
                ("lumi_13TeV",         ("2016pre", "2016post", "2017", "2018"), "all", 1.016),

                # minimal scheme
                #("lumi_13TeV_16",         ("2016pre", "2016post"), "all", 1.01),
                #("lumi_13TeV_17",         ("2017",), "all", 1.02),
                #("lumi_13TeV_18",         ("2018",), "all", 1.015),
                #("lumi_13TeV_1718",       ("2017",), "all", 1.006),
                #("lumi_13TeV_1718",       ("2018",), "all", 1.002),
                #("lumi_13TeV_correlated", ("2016pre", "2016post"), "all", 1.006),
                #("lumi_13TeV_correlated", ("2017",), "all", 1.009),
                #("lumi_13TeV_correlated", ("2018",), "all", 1.02),

                # full scheme
                #("lumi_13TeV_16",                     ("2016pre", "2016post"), "all", 1.01),
                #("lumi_13TeV_17",                     ("2017",), "all", 1.02),
                #("lumi_13TeV_18",                     ("2018",), "all", 1.015),
                #("lumi_13TeV_BeamCurrentCalibration", ("2016pre", "2016post"), "all", 1.002),
                #("lumi_13TeV_BeamCurrentCalibration", ("2017",), "all", 1.003),
                #("lumi_13TeV_BeamCurrentCalibration", ("2018",), "all", 1.002),
                #("lumi_13TeV_BeamBeamEffect",         ("2017",), "all", 1.006),
                #("lumi_13TeV_BeamBeamEffect",         ("2018",), "all", 1.002),
                #("lumi_13TeV_GhostSatellite",         ("2016pre", "2016post", "2017", "2018"), "all", 1.001),
                #("lumi_13TeV_LengthScale",            ("2016pre", "2016post", "2017"), "all", 1.003),
                #("lumi_13TeV_LengthScale",            ("2018",), "all", 1.002),
                #("lumi_13TeV_XYFactorization",        ("2016pre", "2016post"), "all", 1.005),
                #("lumi_13TeV_XYFactorization",        ("2017",), "all", 1.008),
                #("lumi_13TeV_XYFactorization",        ("2018",), "all", 1.02),
            )),
            ("ll" , (
                ("CMS_DY_norm_13TeV",  ("2016pre", "2016post", "2017", "2018"), "DY", 1.05),
                ("CMS_VV_norm_13TeV",  ("2016pre", "2016post", "2017", "2018"), "VV", 1.3),
                ("CMS_TTV_norm_13TeV", ("2016pre", "2016post", "2017", "2018"), "TTV", 1.3),
            )),
            ("e3j" , (
                ("CMS_EWQCD_e3j_norm_13TeV_2016pre",  ("2016pre",), "EWQCD", 1.5),
                ("CMS_EWQCD_e3j_norm_13TeV_2016post", ("2016post",), "EWQCD", 1.5),
                ("CMS_EWQCD_e3j_norm_13TeV_2017",     ("2017",), "EWQCD", 1.5),
                ("CMS_EWQCD_e3j_norm_13TeV_2018",     ("2018",), "EWQCD", 1.5),
            )),
            ("e4pj" , (
                ("CMS_EWQCD_e4pj_norm_13TeV_2016pre",  ("2016pre",), "EWQCD", 1.5),
                ("CMS_EWQCD_e4pj_norm_13TeV_2016post", ("2016post",), "EWQCD", 1.5),
                ("CMS_EWQCD_e4pj_norm_13TeV_2017",     ("2017",), "EWQCD", 1.5),
                ("CMS_EWQCD_e4pj_norm_13TeV_2018",     ("2018",), "EWQCD", 1.5),
            )),
            ("m3j" , (
                ("CMS_EWQCD_m3j_norm_13TeV_2016pre",  ("2016pre",), "EWQCD", 1.5),
                ("CMS_EWQCD_m3j_norm_13TeV_2016post", ("2016post",), "EWQCD", 1.5),
                ("CMS_EWQCD_m3j_norm_13TeV_2017",     ("2017",), "EWQCD", 1.5),
                ("CMS_EWQCD_m3j_norm_13TeV_2018",     ("2018",), "EWQCD", 1.5),
            )),
            ("m4pj" , (
                ("CMS_EWQCD_m4pj_norm_13TeV_2016pre",  ("2016pre",), "EWQCD", 1.5),
                ("CMS_EWQCD_m4pj_norm_13TeV_2016post", ("2016post",), "EWQCD", 1.5),
                ("CMS_EWQCD_m4pj_norm_13TeV_2017",     ("2017",), "EWQCD", 1.5),
                ("CMS_EWQCD_m4pj_norm_13TeV_2018",     ("2018",), "EWQCD", 1.5),
            )),
            ("common" , (
                ("CMS_TQ_norm_13TeV", ("2016pre", "2016post", "2017", "2018"), "TQ", 1.15),
                ("CMS_TW_norm_13TeV", ("2016pre", "2016post", "2017", "2018"), "TW", 1.15),
                ("CMS_TB_norm_13TeV", ("2016pre", "2016post", "2017", "2018"), "TB", 1.15),
                ("CMS_TT_norm_13TeV", ("2016pre", "2016post", "2017", "2018"), "TT", (0.948, 1.044)), # down, up
            ))
        ])
        write_datacard.lnNs["ee"] = write_datacard.lnNs["ll"]
        write_datacard.lnNs["em"] = write_datacard.lnNs["ll"]
        write_datacard.lnNs["mm"] = write_datacard.lnNs["ll"]

    cb = ch.CombineHarvester()
    categories = OrderedDict([(ii, cc) for ii, cc in enumerate(cpn.keys())])
    years = tuple(sorted(years))
    mstr = str( get_point(sigpnt[0])[1] )

    cb.AddObservations(['*'], ["ahtt"], ["13TeV"], [""], categories.items())
    for iicc in categories.items():
        ii = iicc[0]
        cc = iicc[1]

        sigs = [pp for pp in cpn[cc].keys() if any([ss in pp for ss in sigpnt])]
        bkgs = [pp for pp in cpn[cc].keys() if not any([ss in pp for ss in sigpnt]) and (injsig == None or not any([ss in pp for ss in injsig]))]
        cb.AddProcesses([''], ["ahtt"], ["13TeV"], [""], sigs, [iicc], True)
        cb.AddProcesses(['*'], ["ahtt"], ["13TeV"], [""], bkgs, [iicc], False)

        channel, year = cc.rsplit('_', 1)

        for process, nuisances in cpn[cc].items():
            for nuisance in nuisances:
                if isinstance(nuisance[1], float) or isinstance(nuisance[1], int):
                    cb.cp().process([process]).AddSyst(cb, nuisance[0], "shape", ch.SystMap("bin_id")([ii], nuisance[1]))
                elif isinstance(nuisance[1], tuple):
                    cb.cp().process([process]).AddSyst(cb, nuisance[0], nuisance[1][0], ch.SystMap("bin_id")([ii], nuisance[1][1]))
                else:
                    print("make_datacard :: unknown handling for nuisance " + nuisance[0] + ", skipping")

            for ll in [write_datacard.lnNs[years], write_datacard.lnNs[channel], write_datacard.lnNs["common"]]:
                for lnN in ll:
                    if keeps is not None:
                        if not any([dn in lnN[0] for dn in keeps]):
                            print("make_datacard :: nuisance " + lnN[0] + " has been dropped")
                            continue
                    if drops is not None:
                        if drops == ['*'] or any([dn in lnN[0] for dn in drops]):
                            print("make_datacard :: nuisance " + lnN[0] + " has been dropped")
                            continue

                    if year in lnN[1] and (lnN[2] == "all" or lnN[2] == process):
                        if rateparam is None or all(["CMS_{rp}_norm_13TeV".format(rp = rp) != lnN[0] for rp in rateparam]):
                            cb.cp().process([process]).AddSyst(cb, lnN[0], "lnN", ch.SystMap("bin_id")([ii], lnN[3]))

    cb.cp().backgrounds().ExtractShapes(oname, "$BIN/$PROCESS", "$BIN/$PROCESS_$SYSTEMATIC")
    cb.cp().signals().ExtractShapes(oname, "$BIN/$PROCESS", "$BIN/$PROCESS_$SYSTEMATIC")
    syscall("rm {ooo}".format(ooo = oname), False, True)

    writer = ch.CardWriter("$TAG/$ANALYSIS_$BIN.txt", "$TAG/$ANALYSIS_input.root")
    sstr = "__".join(sorted(sigpnt))
    writer.WriteCards(sstr + tag, cb)

    if mcstat:
        txts = glob.glob(sstr + tag + "/ahtt_*.txt")
        for tt in txts:
            with open(tt, 'a') as txt:
                txt.write("\n* autoMCStats 0.\n")

    if rateparam is not None:
        for tt in txts:
            with open(tt, 'a') as txt:
                for rp in rateparam:
                    rpp = rp.split(':')
                    txt.write("\nCMS_{rpp}_norm_13TeV rateParam * {rpp} 1 {rpr}\n".format(rpp = rpp[0], rpr = '[' + rpp[1] + ']' if len(rpp) > 1 else "[0,2]"))

    if len(categories) > 1:
        os.chdir(sstr + tag)
        syscall("combineCards.py {cards} > {comb}".format(
            cards = " ".join([cats + "=ahtt_" + cats + ".txt" for cats in cpn.keys()]),
            comb = "ahtt_combined.txt"
        ))
        os.chdir("..")

if __name__ == '__main__':
    parser = ArgumentParser()
    common_point(parser)
    common_common(parser)
    make_datacard_pure(parser)

    parser.add_argument("--signal", help = combine_help_messages["--signal"], default = "", required = True, type = lambda s: tokenize_to_list( remove_spaces_quotes(s) ))
    parser.add_argument("--background", help = combine_help_messages["--background"], default = "", required = True, type = lambda s: tokenize_to_list( remove_spaces_quotes(s) ))

    parser.add_argument("--channel", help = combine_help_messages["--channel"], default = "ee,em,mm,e3j,e4pj,m3j,m4pj", required = False,
                        type = lambda s: tokenize_to_list( remove_spaces_quotes(s) ))
    parser.add_argument("--year", help = combine_help_messages["--year"], default = "2016pre,2016post,2017,2018", required = False,
                        type = lambda s: tokenize_to_list( remove_spaces_quotes(s) ))

    parser.add_argument("--drop", help = combine_help_messages["--drop"], default = "", required = False,
                        type = lambda s: None if s == "" else sorted(tokenize_to_list( remove_spaces_quotes(s) )))
    parser.add_argument("--keep", help = combine_help_messages["--keep"], default = "", required = False,
                        type = lambda s: None if s == "" else sorted(tokenize_to_list( remove_spaces_quotes(s) )))

    parser.add_argument("--threshold", help = combine_help_messages["--threshold"], default = 0.005, required = False, type = lambda s: float(remove_spaces_quotes(s)))
    parser.add_argument("--float-rate", help = combine_help_messages["--float-rate"], dest = "rateparam", default = "", required = False,
                        type = lambda s: None if s == "" else tokenize_to_list( remove_spaces_quotes(s), ';' ))

    parser.add_argument("--inject-signal", help = combine_help_messages["--inject-signal"], dest = "inject", default = "", required = False,
                        type = lambda s: None if s == "" else sorted(tokenize_to_list( remove_spaces_quotes(s) )))
    parser.add_argument("--replace-nominal", help = combine_help_messages["--replace-nominal"], dest = "replace", default = "", required = False,
                        type = lambda s: None if s == "" else sorted(tokenize_to_list( remove_spaces_quotes(s), ':' )))

    parser.add_argument("--projection", help = combine_help_messages["--projection"], default = "", required = False,
                        type = lambda s: [] if s == "" else tokenize_to_list( remove_spaces_quotes(s) ))

    parser.add_argument("--add-pseudodata", help = combine_help_messages["--add-pseudodata"], dest = "pseudodata", action = "store_true", required = False)
    parser.add_argument("--seed", help = combine_help_messages["--seed"], default = -1, required = False, type = lambda s: int(remove_spaces_quotes(s)))
    args = parser.parse_args()

    allyears = ["2016pre", "2016post", "2017", "2018"]
    if not all([yy in allyears for yy in args.year]):
        print "supported years:", allyears
        raise RuntimeError("unexpected year is given. aborting.")

    allchannels = ["ee", "em", "mm", "ll", "ej", "e3j", "e4pj", "mj", "m3j", "m4pj", "lj"]
    if not all([cc in allchannels for cc in args.channel]):
        print "supported channels:", allchannels
        raise RuntimeError("unexpected channel is given. aborting.")

    if not all([len(pp.split(';')) == 3 for pp in args.projection]):
        raise RuntimeError("unexpected projection instruction syntax. aborting")
    prjchan = [pp.split(';')[0].split(',') for pp in args.projection]
    prjrule = [pp.split(';')[1:] for pp in args.projection]

    if args.inject is not None:
        args.pseudodata = True

    oname = "{cwd}/tmp.root".format(cwd = os.getcwd())
    output = TFile(oname, "recreate")
    cpn = OrderedDict()
    for yy in args.year:
        for cc in args.channel:
            prule = ""
            for pchan in prjchan:
                if cc in pchan:
                    prule = prjrule[prjchan.index(pchan)]

            read_category_process_nuisance(output, args.signal, cc, yy, cpn, args.pseudodata,
                                           args.replace, args.drop, args.keep, args.alwaysshape, args.threshold, args.lnNsmall, prule,
                                           args.point, args.kfactor)
            if args.inject is not None and args.point != args.inject:
                remaining = list(set(args.inject).difference(args.point))
                if len(remaining) > 0:
                    read_category_process_nuisance(output, args.signal, cc, yy, cpn, args.pseudodata,
                                                   args.replace, args.drop, args.keep, args.alwaysshape, args.threshold, args.lnNsmall, prule,
                                                   remaining, args.kfactor)

            read_category_process_nuisance(output, args.background, cc, yy, cpn, args.pseudodata,
                                           args.replace, args.drop, args.keep, args.alwaysshape, args.threshold, args.lnNsmall, prule)

    if args.pseudodata:
        print "using ", args.seed, "as seed for pseudodata generation"
        make_pseudodata(output, cpn, args.inject, args.seed if args.seed != 0 else None)
    output.Close()

    write_datacard(oname, cpn, args.year, args.point, args.inject, args.drop, args.keep, args.mcstat, args.rateparam, args.tag)
