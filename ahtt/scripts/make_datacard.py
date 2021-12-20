#!/usr/bin/env python
# makes a datacard off the requested signal point
# assumes that both background and signal root files are produced by plot_diagnostic.cc
# the files contain the search templates for all considered processes in the analysis, and all shape nuisances
# for nuisances that are smoothed, additionally a histogram containing fit quality information is expected
# which is used to determine if they are to be dropped/assigned as lnN sources instead
# the code assumes that smoothed nuisances are attached to one background process
# always lnN nuisances which aren't considered by plot_diagnostic.cc is for the moment hardcoded
# FIXME currently only 1 signal point is supported, 2 or more (e.g. for hMSSM) to come 

from argparse import ArgumentParser
import os
import sys
import math

import glob
from collections import OrderedDict

from ROOT import TFile, gDirectory, TH1
TH1.AddDirectory(False)
TH1.SetDefaultSumw2(True)

from numpy import random as rng
import CombineHarvester.CombineTools.ch as ch

def syscall(cmd, verbose = True):
    if verbose:
        print ("Executing: %s" % cmd)
    retval = os.system(cmd)
    if retval != 0:
        raise RuntimeError("Command failed!")

def flat_reldev_wrt_nominal(varied, nominal, offset):
    for ii in range(1, nominal.GetNbinsX() + 1):
        nn = nominal.GetBinContent(ii)
        varied.SetBinContent(ii, nn * (1. + offset))

def scale(histogram, factor):
    for ii in range(1, histogram.GetNbinsX() + 1):
        histogram.SetBinContent(ii, histogram.GetBinContent(ii) * factor)
        histogram.SetBinError(ii, histogram.GetBinError(ii) * abs(factor))

def get_point(sigpnt):
    pnt = sigpnt.split('_')
    return (pnt[0][0], float(pnt[0][1:]), float(pnt[1].replace("relw", "").replace('p', '.')))

def get_kfactor(sigpnt):
    kfile = TFile.Open("/nfs/dust/cms/group/exotica-desy/HeavyHiggs/ahtt_kfactor_sushi/ulkfactor_final_21xxxx.root")
    khist = [
        (kfile.Get(sigpnt[0] + "_res_sushi_nnlo_mg5_lo_kfactor_pdf_325500_" + syst),
         kfile.Get(sigpnt[0] + "_int_sushi_nnlo_mg5_lo_kfactor_pdf_325500_" + syst))
        for syst in ["nominal", "uF_up", "uF_down", "uR_up", "uR_down"]
    ]
    idx = khist[0][0].FindBin(sigpnt[1], sigpnt[2])
    kvals = tuple([(syst[0].GetBinContent(idx), syst[1].GetBinContent(idx)) for syst in khist])
    kfile.Close()
    return kvals

# FIXME partial shape correlations not yet implemented, meaning not clear yet
# current assumption is some specific list is fully correlated, others fully uncorrelated
def read_category_process_nuisance(ofile, ifile, channel, year, cpn, pseudodata, alwaysshape, threshold, lnNsmall, sigpnt = "", kfactor = False):
    # to note nuisances that need special handling
    # 'regular' nuisances are those that are uncorrelated between years with a scaling of 1
    if not hasattr(read_category_process_nuisance, "specials"):
        read_category_process_nuisance.specials = OrderedDict([
            ("TT_uR"     , (("2016", "2016apv", "2017", "2018"), 1.)),
            ("TT_uF"     , (("2016", "2016apv", "2017", "2018"), 1.)),
            ("hdamp"     , (("2016", "2016apv", "2017", "2018"), 1.)),
            ("colorrec"  , (("2016", "2016apv", "2017", "2018"), 1.)),
            ("yukawa"    , (("2016", "2016apv", "2017", "2018"), 1.)),
            ("PS_ISR"    , (("2016", "2016apv", "2017", "2018"), 1.)),
            ("PS_FSR"    , (("2016", "2016apv", "2017", "2018"), 1.)),
            ("UE_tune"   , (("2016", "2016apv", "2017", "2018"), 1.)),
            ("pileup"    , (("2016", "2016apv", "2017", "2018"), 1.)),
            ("mtop_3GeV" , (("2016", "2016apv", "2017", "2018"), 1. / 6.)),
            ("AH_uR"     , (("2016", "2016apv", "2017", "2018"), 1.)),
            ("AH_uF"     , (("2016", "2016apv", "2017", "2018"), 1.))
        ])

    # because afiq hates seeing jets spelled outside of text
    if not hasattr(read_category_process_nuisance, "aliases"):
        read_category_process_nuisance.aliases = OrderedDict([
            ("e3j" , "e3jets"),
            ("e4pj", "e4pjets"),
            ("m3j" , "mu3jets"),
            ("m4pj", "mu4pjets"),
        ])

    processes = []
    nuisances = []

    odir = channel + '_' + year
    chan = read_category_process_nuisance.aliases[channel] if channel in read_category_process_nuisance.aliases else channel
    if not bool(ofile.GetDirectory(odir)):
        ofile.mkdir(odir)

    kfactors = None
    if sigpnt == "":
        ifile.cd(chan)
        keys = gDirectory.GetListOfKeys()

        for key in keys:
            kname = key.GetName()

            if not kname.endswith("Up") and not kname.endswith("Down") and not kname.endswith("_chi2"):
                if kname != "data_obs" or not pseudodata:
                    processes.append(kname)
                    ofile.cd(odir)
                    key.ReadObj().Write()
    else:
        if kfactor:
            kfactors = get_kfactor(get_point(sigpnt))

        for ss in ["_res", "_int_pos", "_int_neg"]:
            pnt = sigpnt + ss
            processes.append(pnt)

            hn = ifile.Get(chan + '/' + pnt)

            if kfactor and kfactors is not None:
                if ss == "_res":
                    scale(hn, kfactors[0][0])
                else:
                    scale(hn, kfactors[0][1])

            if ss == "_int_neg":
                scale(hn, -1.)

            ofile.cd(odir)
            hn.Write()

    for pp in processes:
        nuisance = []
        ifile.cd(chan)
        keys = gDirectory.GetListOfKeys()

        for key in keys:
            kname = key.GetName()

            if pp + '_' in kname and kname.endswith("Up"):
                nn1 = "".join(kname.rsplit("Up", 1)).replace(pp + '_', "", 1)
                if nn1 in read_category_process_nuisance.specials:
                    if year in read_category_process_nuisance.specials[nn1][0]:
                        nn2 = nn1
                    else:
                        nn2 = nn1 + '_' + year

                    if "AH_u" in nn2:
                        nn2 = nn2 + "_res" if pp == sigpnt + "_res" else nn2 + "_int"

                    nuisance.append((nn2, read_category_process_nuisance.specials[nn1][1]))
                else:
                    nn2 = nn1 + '_' + year
                    nuisance.append((nn2, 1.))

                hu = key.ReadObj()
                hd = ifile.Get(channel + '/' + "Down".join(kname.rsplit("Up", 1)))
                hc = ifile.Get(channel + '/' + "_chi2".join(kname.rsplit("Up", 1))) if keys.Contains("_chi2".join(kname.rsplit("Up", 1))) else None

                drop_nuisance = False
                if not alwaysshape and hc is not None:
                    # the values are smooth chi2 up, down, flat chi2 up, down and flat values up, down
                    chi2s = [hc.GetBinContent(ii) for ii in range(1, 7)]

                    scaleu = 1.
                    scaled = 1.

                    if chi2s[2] < chi2s[0] and chi2s[3] < chi2s[1]:
                        keepvalue = abs(chi2s[4]) > threshold or abs(chi2s[5]) > threshold or lnNsmall

                        scaleu = chi2s[4] if keepvalue else 0.
                        scaled = chi2s[5] if keepvalue else 0.

                        flat_reldev_wrt_nominal(hd, ifile.Get(channel + '/' + pp), scaled)
                        flat_reldev_wrt_nominal(hu, ifile.Get(channel + '/' + pp), scaleu)

                    if scaleu == 0. and scaled == 0.:
                        drop_nuisance = True

                if drop_nuisance:
                    nuisance.pop()
                    continue

                if kfactor and kfactors is not None:
                    idxp = 0 if pp == sigpnt + "_res" else 1
                    idxu = 1 if "_uF_" in nn2 else 3 if "_uR_" in nn2 else 0
                    idxd = 2 if "_uF_" in nn2 else 4 if "_uR_" in nn2 else 0

                    scale(hu, kfactors[idxu][idxp])
                    scale(hd, kfactors[idxd][idxp])

                ofile.cd(odir)
                hu.SetName(hu.GetName().replace(nn1, nn2))
                hd.SetName(hd.GetName().replace(nn1, nn2))

                if "int_neg" in pp:
                    scale(hu, -1.)
                    scale(hd, -1.)

                hu.Write()
                hd.Write()

        nuisances.append(nuisance)

    if odir not in cpn:
        cpn[odir] = OrderedDict()

    for pp, nn in zip(processes, nuisances):
        cpn[odir][pp] = nn

def make_pseudodata(ofile, cpn, sigpnt = "", seed = None):
    if seed is None or seed >= 0:
        rng.seed(seed)

    for category, processes in cpn.items():
        dd = None
        for pp in processes.keys():
            issig = any([ss in pp for ss in ["_res", "_int_pos", "_int_neg"]])
            hh = ofile.Get(category + '/' + pp).Clone("hhtmphh")
            if "_int_neg" in pp:
                scale(hh, -1.)

            if not issig or (issig and sigpnt != "" and sigpnt in pp):
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

def write_datacard(oname, cpn, years, sigpnt, injsig, mcstat, tag):
    # to note nuisances that need special handling
    # 'regular' nuisances are those that are uncorrelated between years with a scaling of 1
    if not hasattr(write_datacard, "lnNs"):
        # FIXME check again for UL
        write_datacard.lnNs = OrderedDict([
            (("2018",), (
                ("lumi", ("2018",), "all", 1.025),
            )),
            (("2017",), (
                ("lumi", ("2017",), "all", 1.023),
            )),
            # FIXME 2016, year pairs, all years
            ("ll" , (
                ("DY_norm",  ("2016", "2016apv", "2017", "2018"), "DY", 1.3),
                ("TTV_norm", ("2016", "2016apv", "2017", "2018"), "TTV", 1.15),
            )),
            # FIXME lj is just a candidate impl for later dev
            ("lj" , (
                ("EWQCD_norm", ("2016", "2016apv", "2017", "2018"), "EWQCD", (2.0, 1.5)), # down/up, where down = scale by 1/x and up = scale by x
            )),
            ("common" , (
                ("TT_norm", ("2016", "2016apv", "2017", "2018"), "TT", (1.065, 1.056)), # as above, with down 0.939 * nominal
                ("TQ_norm", ("2016", "2016apv", "2017", "2018"), "TQ", 1.15),
                ("TW_norm", ("2016", "2016apv", "2017", "2018"), "TW", 1.15),
                ("TB_norm", ("2016", "2016apv", "2017", "2018"), "TB", 1.15),
            ))
        ])
        write_datacard.lnNs["ee"]   = write_datacard.lnNs["ll"]
        write_datacard.lnNs["em"]  = write_datacard.lnNs["ll"]
        write_datacard.lnNs["mm"] = write_datacard.lnNs["ll"]

        write_datacard.lnNs["m3j"]  = write_datacard.lnNs["lj"]
        write_datacard.lnNs["m4pj"] = write_datacard.lnNs["lj"]
        write_datacard.lnNs["e3j"]   = write_datacard.lnNs["lj"]
        write_datacard.lnNs["e4pj"]  = write_datacard.lnNs["lj"]

    cb = ch.CombineHarvester()
    categories = OrderedDict([(ii, cc) for ii, cc in enumerate(cpn.keys())])
    years = tuple(sorted(years))
    point = get_point(sigpnt)
    mstr = str(point[1]).replace(".0", "")

    cb.AddObservations(['*'], ["ahtt"], ["13TeV"], [""], categories.items())
    for iicc in categories.items():
        ii = iicc[0]
        cc = iicc[1]

        sigs = [pp for pp in cpn[cc].keys() if sigpnt in pp]
        bkgs = [pp for pp in cpn[cc].keys() if sigpnt not in pp and (injsig == "" or injsig not in pp)]
        cb.AddProcesses([mstr], ["ahtt"], ["13TeV"], [""], sigs, [iicc], True)
        cb.AddProcesses(['*'], ["ahtt"], ["13TeV"], [""], bkgs, [iicc], False)

        channel, year = cc.rsplit('_', 1)

        for process, nuisances in cpn[cc].items():
            for nuisance in nuisances:
                cb.cp().process([process]).AddSyst(cb, nuisance[0], "shape", ch.SystMap("bin_id")([ii], nuisance[1]))

            for ll in [write_datacard.lnNs[years], write_datacard.lnNs[channel], write_datacard.lnNs["common"]]:
                for lnN in ll:
                    if year in lnN[1] and (lnN[2] == "all" or lnN[2] == process):
                        cb.cp().process([process]).AddSyst(cb, lnN[0], "lnN", ch.SystMap("bin_id")([ii], lnN[3]))

    cb.cp().backgrounds().ExtractShapes(oname, "$BIN/$PROCESS", "$BIN/$PROCESS_$SYSTEMATIC")
    cb.cp().signals().ExtractShapes(oname, "$BIN/$PROCESS", "$BIN/$PROCESS_$SYSTEMATIC")
    os.remove(oname)

    writer = ch.CardWriter("$TAG/$ANALYSIS_$BIN.txt", "$TAG/$ANALYSIS_input.root")
    writer.WriteCards(sigpnt + tag, cb)

    if mcstat:
        txts = glob.glob(sigpnt + tag + "/ahtt_*.txt")
        for tt in txts:
            with open(tt, 'a') as txt:
                txt.write("\n* autoMCStats 0.\n")

    if len(categories) > 1:
        syscall("combineCards.py {cards} > {comb}".format(
            cards = " ".join([cats + "=" + sigpnt + tag + "/ahtt_" + cats + ".txt" for cats in cpn.keys()]),
            comb = sigpnt + tag + "/ahtt_combined.txt"
        ))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--signal", help = "signal filename. comma separated", default = "", required = True)
    parser.add_argument("--background", help = "data/background filename. comma separated", default = "", required = True)
    parser.add_argument("--point", help = "signal point to make datacard of", default = "", required = True)

    parser.add_argument("--channel", help = "final state channels considered in the analysis. comma separated", default = "ll", required = False)
    parser.add_argument("--year", help = "analysis year determining the correlation model to assume. comma separated", default = "2018", required = False)
    parser.add_argument("--tag", help = "extra tag to be put on datacard names", default = "", required = False)
    parser.add_argument("--threshold", help = "threshold under which nuisances that are better fit by a flat line are dropped/assigned as lnN",
                        default = 0.005, required = False, type = float)
    parser.add_argument("--sushi-kfactor", help = "apply nnlo kfactors computing using sushi on A/H signals",
                        dest = "kfactor", action = "store_true", required = False)
    parser.add_argument("--lnN-under-threshold", help = "assign as lnN nuisances as considered by --threshold",
                        dest = "lnNsmall", action = "store_true", required = False)
    parser.add_argument("--use-shape-always", help = "use lowess-smoothened shapes even if the flat fit chi2 is better",
                        dest = "alwaysshape", action = "store_true", required = False)
    parser.add_argument("--use-pseudodata", help = "don't read the data from file, instead construct pseudodata using poisson-varied sum of backgrounds",
                        dest = "pseudodata", action = "store_true", required = False)
    parser.add_argument("--inject-signal", help = "signal point to inject into the pseudodata", dest = "injectsignal", default = "", required = False)
    parser.add_argument("--no-mc-stats", help = "don't add nuisances due to limited mc stats (barlow-beeston lite)",
                        dest = "mcstat", action = "store_false", required = False)
    parser.add_argument("--seed",
                        help = "random seed to be used for pseudodata generation. give 0 to read from machine, and negative values to use no rng",
                        default = 423029859, required = False, type = int)
    args = parser.parse_args()
    if (args.tag != "" and not args.tag.startswith("_")):
        args.tag = "_" + args.tag

    sfiles = [TFile.Open(ff) for ff in args.signal.strip().split(',')]
    bfiles = [TFile.Open(ff) for ff in args.background.strip().split(',')]
    years = args.year.strip().split(',')
    channels = args.channel.strip().split(',')

    allyears = ["2016", "2016apv", "2017", "2018"]
    if not all([yy in allyears  for yy in years]):
        print "supported years:", allyears
        raise RuntimeError("unxpected year is given. aborting.")

    allchannels = ["ee", "em", "mm", "ll", "ej", "e3j", "e4pj", "mj", "m3j", "m4pj", "lj"]
    if not all([cc in allchannels for cc in channels]):
        print "supported channels:", allchannels
        raise RuntimeError("unxpected channel is given. aborting.")

    if len(years) != len(sfiles) or len(sfiles) != len(bfiles):
        raise RuntimeError("number of signal/background files don't match the number of years. aborting.")

    if args.injectsignal != "":
        args.pseudodata = True

    oname = "tmp.root"
    output = TFile(oname, "recreate")
    cpn = OrderedDict()
    for bb, ss, yy in zip(bfiles, sfiles, years):
        for cc in channels:
            read_category_process_nuisance(output, ss, cc, yy, cpn, args.pseudodata, args.alwaysshape, args.threshold, args.lnNsmall,
                                           args.point, args.kfactor)
            if args.injectsignal != "" and args.point != args.injectsignal:
                read_category_process_nuisance(output, ss, cc, yy, cpn, args.pseudodata, args.alwaysshape, args.threshold, args.lnNsmall,
                                               args.injectsignal, args.kfactor)
            read_category_process_nuisance(output, bb, cc, yy, cpn, args.pseudodata, args.alwaysshape, args.threshold, args.lnNsmall)

    if args.pseudodata:
        print "using", args.seed, "as seed for pseudodata generation"
        make_pseudodata(output, cpn, args.injectsignal, args.seed if args.seed != 0 else None)
    output.Close()

    write_datacard(oname, cpn, years, args.point, args.injectsignal, args.mcstat, args.tag)
