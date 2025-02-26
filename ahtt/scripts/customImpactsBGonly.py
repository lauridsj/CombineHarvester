#!/usr/bin/env python

# //--------------------------------------------
#-- Original script by Evan Ranken (in the context of TOP-19-008), adapted from the default combine script (https://github.com/cms-analysis/CombineHarvester/blob/master/CombineTools/scripts/plotImpacts.py)
#-- Slightly adapted by Nicolas Tonon for wider use within TOP PAG

#-- Script and instructions available in TOP PAG combine repository: https://gitlab.cern.ch/CMS-TOPPAG/Datacards
# //--------------------------------------------

# Environment: CMSSW, like standard plotImpacts.py
# Usage:
# ./customImpacts.py --input path/to/merged/data/impacts.json --asimov-input path/to/merged/asimov/impacts.json --output outputfile --translate path/to/nuisance_map.json --per-page 20

import ROOT
import math
import json
import argparse
import CombineHarvester.CombineTools.plotting as plot
import CombineHarvester.CombineTools.combine.rounding as rounding

ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(ROOT.kTRUE)
ROOT.TH1.AddDirectory(0)

#-- Arguments
# //--------------------------------------------
parser = argparse.ArgumentParser()
#MAIN ARGS
parser.add_argument('--input' ,'-i', help='Input json file for observed results', default=None)
parser.add_argument('--asimov-input','-ia', help='Input json file for asimov/expected results', default=None)
parser.add_argument('--output', '-o', help='Name of output file to create (without .pdf extension)')
parser.add_argument('--translate', '-t', help='json file for remapping of parameter names')
parser.add_argument('--POI', default=None, help='Specify a POI to draw') #E.g. 'r'
parser.add_argument('--blind', action='store_true', help='Do not print best fit signal strength')
#OTHERS
parser.add_argument("--onlyfirstpage", metavar="onlyfirstpage", default=True, help="Only display the first page", nargs='?', const=1)
parser.add_argument('--units', default=None, help='Add units to the best-fit parameter value')
parser.add_argument('--per-page','-n', type=int, default=10, help='Number of parameters to show per page')
parser.add_argument('--cms-label', default='', help='Label next to the CMS logo')
parser.add_argument("--pullDef",  default=None, help="Choose the definition of the pull, see HiggsAnalysis/CombinedLimit/python/calculate_pulls.py for options")
args = parser.parse_args()

# //--------------------------------------------
# //--------------------------------------------

has_data = args.input is not None
has_asimov = args.asimov_input is not None

if not has_data and not has_asimov:
    raise ValueError("Need to give at least one of data and asimov input")

externalPullDef = False
if args.pullDef is not None:
    externalPullDef = True
    import HiggsAnalysis.CombinedLimit.calculate_pulls as CP

#-- Returns translated name if key found in json dict, else returns arg
def Translate(name, ndict):
    nuisances = ndict["nuisances"]
    if name in nuisances:
        return nuisances[name]
    else:
        for chankey, channame in ndict["channels"].items():
            if chankey in name:
                name_temp = name.replace(chankey, "$CHAN")
                if name_temp in nuisances:
                    return nuisances[name_temp].replace("$CHAN", channame)

        for year in ndict["years"]:
            if year in name:
                name_temp = name.replace(year, "$YEAR")
                if name_temp in nuisances:
                    return nuisances[name_temp].replace("$YEAR", year)
                else:
                    for chankey, channame in ndict["channels"].items():
                        if chankey in name:
                            name_temp = name_temp.replace(chankey, "$CHAN")
                            if name_temp in nuisances:
                                return nuisances[name_temp].replace("$YEAR", year).replace("$CHAN", channame)

        for prockey, procname in ndict["processes"].items():
            if prockey in name:
                name_temp = name.replace(prockey, "$PROC")
                if name_temp in nuisances:
                    return nuisances[name_temp].replace("$PROC", procname)
    return name

#-- Rounding nominal, uperror, downerror
def GetRounded(nom, e_hi, e_lo):
    if e_hi < 0.0:
        e_hi = 0.0
    if e_lo < 0.0:
        e_lo = 0.0
    rounded = rounding.PDGRoundAsym(nom,
            e_hi if e_hi != 0.0 else 1.0,
            e_lo if e_lo != 0.0 else 1.0)
    s_nom = rounding.downgradePrec(rounded[0],rounded[2])
    s_hi = rounding.downgradePrec(rounded[1][0][0],rounded[2]) if e_hi != 0.0 else '0'
    s_lo = rounding.downgradePrec(rounded[1][0][1],rounded[2]) if e_lo != 0.0 else '0'
    return (s_nom, s_hi, s_lo)

#Load json output of combineTool.py -M Impacts (observed)
if has_data:
    data = {}
    with open(args.input) as jsonfile:
        data = json.load(jsonfile)
    if not has_asimov:
        asidata = data

#Load json output of combineTool.py -M Impacts (expected/asimov)
if has_asimov:
    asidata = {}
    with open(args.asimov_input) as asifile:
        asidata = json.load(asifile)
    if not has_data:
        data = asidata

#Set global plotting style
plot.ModTDRStyle(l=0.4, b=0.10, width=700)

#Assume the first POI is the one to plot
POIs = [ele['name'] for ele in data['POIs']]
POI = POIs[0]
if args.POI:
    POI = args.POI
for ele in data['POIs']:
    if ele['name'] == POI:
        POI_info = ele
        break
POI_fit = POI_info['fit']

xs_base = 6.43

#json dictionary to translate parameter names
translate = {}
if args.translate is not None:
    with open(args.translate) as jsonfile:
        translate = json.load(jsonfile)
if POI == "CMS_EtaT_norm_13TeV":
    poi_translated = "\\hat{\\sigma}(\\eta_{t})"
elif POI == "CMS_ChiT_norm_13TeV":
    poi_translated = "\\hat{\\sigma}(\\chi_{t})"
else:
    poi_translated = Translate(POI,translate) #Get translated POI name (if available)

#Sort parameters by largest absolute impact on this POI
data['params'].sort(key=lambda x: abs(x['impact_%s' % POI]), reverse=True)

#Set the number of parameters per page (show) and the number of pages ((n)
show = args.per_page
n = int(math.ceil(float(len(data['params'])) / float(show)))
if args.onlyfirstpage == True: n = 1

colors = {
    'Gaussian': 1,
    'Poisson': 1,
    'AsymmetricGaussian': 1,
    'Unconstrained': 1,
    'Unrecognised': 2
}
color_hists = {}
color_group_hists = {}
for name, col in colors.iteritems():
    color_hists[name] = ROOT.TH1F()
    plot.Set(color_hists[name], FillColor=col, Title=name)

seen_types = set()


# //--------------------------------------------
# //--------------------------------------------

for page in xrange(n):
    canv = ROOT.TCanvas(args.output, args.output)
    n_params = len(data['params'][show * page:show * (page + 1)])
    pdata = data['params'][show * page:show * (page + 1)]
    asipdata = asidata["params"]
    print '>> Doing page %i, have %i parameters' % (page, n_params)

    #-- Set top/bottom margins
    ROOT.gStyle.SetPadBottomMargin(0.1)
    ROOT.gStyle.SetPadTopMargin(0.10 if args.blind else 0.15) #If not displaying best-fit poi, reduce top margin

    #-- Set bkg color of every other line to gray
    boxes = []
    for i in xrange(n_params):
        y1 = ROOT.gStyle.GetPadBottomMargin()
        y2 = 1. - ROOT.gStyle.GetPadTopMargin()
        h = (y2 - y1) / float(n_params)
        y1 = y1 + float(i) * h
        y2 = y1 + h
        x1 = 0 #Start from canvas' left border
        x2 = 0.965 #Fine-tuned such that gray bkg (for every other param) stops at vertical black line #or 1 to reach right-end of canvas
        box = ROOT.TPaveText(x1, y1, x2, y2, 'NDC')
        plot.Set(box, TextSize=0.02, BorderSize=0, FillColor=0, TextAlign=12, Margin=0.005)
        if i % 2 == 0: box.SetFillColor(18) #Gray
        box.Draw()
        boxes.append(box)

    #-- Pad style
    pads = plot.MultiRatioSplitColumns([0.7], [0.], [0.])
    pads[0].SetGrid(1, 0)
    pads[0].SetTickx(1)
    pads[1].SetGrid(1, 0)
    pads[1].SetTickx(1)

    #-- Define main objects
    h_pulls = ROOT.TH2F("pulls", "pulls", 6, -2.9, 2.9, n_params, 0, n_params) #TH2F defining the frame, bkg style, etc.
    
    if has_data:
        g_pulls = ROOT.TGraphAsymmErrors(n_params) #Pulls obs  
        g_impacts_hi = ROOT.TGraphAsymmErrors(n_params) #Obs impact
        g_impacts_lo = ROOT.TGraphAsymmErrors(n_params) #Obs impact

    if has_asimov:
        g_pullsA = ROOT.TGraphAsymmErrors(n_params) #Pulls exp
    #    g_impactsA_hi = ROOT.TGraphAsymmErrors(n_params) #Exp impact
    #    g_impactsA_lo = ROOT.TGraphAsymmErrors(n_params) #Exp impact
    g_check = ROOT.TGraphAsymmErrors()

    #-- Read values, fill pull histogram
    max_impact = 0.
    text_entries = []
    redo_boxes = []
    for p in xrange(n_params):
        i = n_params - (p + 1)
        thisname = pdata[p]['name']
        asipnum = -1
        asipnumfound = False
        while not asipnumfound:
            asipnum +=1
            if asipnum >= len(asipdata)-1:
                print "too many vars in asimov data ", thisname,asipnum
                break
                # raise Exception('ERROR: the parameter with name {}'.format(thisname)
                #        +' has no counterpart in the expected results!')
            if asipdata[asipnum]["name"]==thisname:
     			asipnumfound = True
        pre = pdata[p]['prefit']
        fit = pdata[p]['fit']
        preA = asipdata[asipnum]['prefit']
        fitA = asipdata[asipnum]['fit']
        tp = pdata[p]['type']
        seen_types.add(tp)

        if pdata[p]['type'] != 'Unconstrained':
            pre_err_hi = (pre[2] - pre[1])
            pre_err_lo = (pre[1] - pre[0])

            if externalPullDef:
                fit_err_hi = (fit[2] - fit[1])
                fit_err_lo = (fit[1] - fit[0])
                pull, pull_hi, pull_lo = CP.returnPullAsym(args.pullDef,fit[1],pre[1],fit_err_hi,pre_err_hi,fit_err_lo,pre_err_lo)
            else:
                pull = fit[1] - pre[1]
                pull = (pull/pre_err_hi) if pull >= 0 else (pull/pre_err_lo)
                pull_hi = fit[2] - pre[1]
                pull_hi = (pull_hi/pre_err_hi) if pull_hi >= 0 else (pull_hi/pre_err_lo)
                pull_hi = pull_hi - pull
                pull_lo = fit[0] - pre[1]
                pull_lo = (pull_lo/pre_err_hi) if pull_lo >= 0 else (pull_lo/pre_err_lo)
                pull_lo =  pull - pull_lo


            if has_data:
                g_pulls.SetPoint(i, pull, float(i) + 0.65)
                g_pulls.SetPointError(i, pull_lo, pull_hi, 0., 0.)

            pre_err_hi = (preA[2] - preA[1])
            pre_err_lo = (preA[1] - preA[0])

            if externalPullDef:
                fit_err_hi = (fit[2] - fit[1])
                fit_err_lo = (fit[1] - fit[0])
                pull, pull_hi, Apull_lo = CP.returnPullAsym(args.pullDef,fit[1],pre[1],fit_err_hi,pre_err_hi,fit_err_lo,pre_err_lo)
            else:
                pull = fitA[1] - preA[1]
                pull = (pull/pre_err_hi) if pull >= 0 else (pull/pre_err_lo)
                pull_hi = fitA[2] - preA[1]
                pull_hi = (pull_hi/pre_err_hi) if pull_hi >= 0 else (pull_hi/pre_err_lo)
                pull_hi = pull_hi - pull
                pull_lo = fitA[0] - preA[1]
                pull_lo = (pull_lo/pre_err_hi) if pull_lo >= 0 else (pull_lo/pre_err_lo)
                pull_lo =  pull - pull_lo

            if has_asimov:
                g_pullsA.SetPoint(i, pull, float(i) + 0.35)
                g_pullsA.SetPointError(
                    i, pull_lo, pull_hi, 0., 0.)
        else: #Unconstrained, hide point
            if has_data:
                g_pulls.SetPoint(i, 0., 9999.)
            y1 = ROOT.gStyle.GetPadBottomMargin()
            y2 = 1. - ROOT.gStyle.GetPadTopMargin()
            x1 = ROOT.gStyle.GetPadLeftMargin()
            h = (y2 - y1) / float(n_params)
            y1 = y1 + ((float(i)+0.5) * h)
            x1 = x1 + (1 - pads[0].GetRightMargin() -x1)/2.
            # s_nom, s_hi, s_lo = GetRounded(fit[1], fit[2] - fit[1], fit[1] - fit[0])
            # this part stores text/shapeU vars and sets the digits to show
            s_nom, s_hi, s_lo = "{:.3f}".format(fit[1]), "{:.3f}".format( fit[2] - fit[1]), "{:.3f}".format( fit[1] - fit[0])
            text_entries.append((x1, y1, '%s^{#plus%s}_{#minus%s}' % (s_nom, s_hi, s_lo)))
            redo_boxes.append(i)

        if has_data:
            g_impacts_hi.SetPoint(i, 0, float(i) + 0.5)
            g_impacts_lo.SetPoint(i, 0, float(i) + 0.5)
        #if has_asimov:
        #    g_impactsA_hi.SetPoint(i, 0, float(i) + 0.5)
        #    g_impactsA_lo.SetPoint(i, 0, float(i) + 0.5)
        imp = pdata[p][POI]
        imp = [ii*xs_base for ii in imp]
        #impA = asipdata[asipnum][POI]
        if has_data:
            if imp[2]-imp[1]>0:
                g_impacts_hi.SetPointError(i, 0, imp[2] - imp[1], 0.0, 0.0)
                g_impacts_lo.SetPointError(i, imp[1] - imp[0], 0, 0.0, 0.0)
            else:
                g_impacts_hi.SetPointError(i,  imp[1] - imp[2],0, 0.0, 0.0)
                g_impacts_lo.SetPointError(i, 0, imp[0] - imp[1],  0.0, 0.0)
        max_impact = max(max_impact, abs(imp[1] - imp[0]), abs(imp[2] - imp[1]))
        #max_impact = max(max_impact, abs(impA[1] - impA[0]), abs(impA[2] - impA[1]))
        col = colors.get(tp, 2)
        #if has_asimov:
        #    if impA[2]-impA[1]>0:
        #        g_impactsA_hi.SetPointError(i, 0, impA[2] - impA[1], 0.49, 0.49)
        #        g_impactsA_lo.SetPointError(i, impA[1] - impA[0], 0, 0.49, 0.49)
        #    else:
        #        g_impactsA_hi.SetPointError(i,  impA[1] - impA[2],0, 0.49, 0.49)
        #        g_impactsA_lo.SetPointError(i, 0, impA[0] - impA[1],  0.49, 0.49)

        thisname = Translate(thisname,translate)
        h_pulls.GetYaxis().SetBinLabel(i + 1, ('#color[%i]{%s}'% (col, thisname)))

    #-- Style and draw the pulls histo
    if externalPullDef:
        plot.Set(h_pulls.GetXaxis(), TitleSize=0.04, LabelSize=0.03, Title=CP.returnTitle(args.pullDef))
    else:
        plot.Set(h_pulls.GetXaxis(), TitleSize=0.04, LabelSize=0.03, Title='(#hat{#theta}-#theta_{0})/#Delta#theta')
    plot.Set(h_pulls.GetYaxis(), LabelSize=0.04, TickLength=0.0)
    h_pulls.GetYaxis().SetLabelFont(42)
    h_pulls.GetYaxis().LabelsOption('v')
    h_pulls.Draw()

    for i in redo_boxes:
        newbox = boxes[i].Clone()
        newbox.Clear()
        newbox.SetY1(newbox.GetY1()+0.005)
        newbox.SetY2(newbox.GetY2()-0.005)
        newbox.SetX1(ROOT.gStyle.GetPadLeftMargin()+0.002)
        newbox.SetX2(0.7-0.001)
        newbox.Draw()
        boxes.append(newbox)
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    # this part draws the shapeU/text constraints and you can change the size here
    latex.SetTextSize(0.024)
    latex.SetTextAlign(22)
    for entry in text_entries:
        latex.DrawLatex(*entry)

    #-- Draw impacts histogram
    pads[1].cd()
    if max_impact == 0.: max_impact = 1E-6  # otherwise the plotting gets screwed up
    if max_impact > .9: max_impact = .5
    h_impacts = ROOT.TH2F(
        "impacts", "impacts", 6, -max_impact * 1.06, max_impact * 1.06, n_params, 0, n_params)
    plot.Set(h_impacts.GetXaxis(), LabelSize=0.03, TitleSize=0.04, Ndivisions=505, Title=
     # '#Delta#mu')
     '#Delta%s [pb]' % poi_translated)
    plot.Set(h_impacts.GetYaxis(), LabelSize=0, TickLength=0.0)
    h_impacts.Draw()

    #-- Draw arXiv entry (if needed)
    draw_arxiv = False
    if draw_arxiv == True:
        pads[0].cd()
        xpos = ROOT.gPad.GetLeftMargin()
        xposR = 1-ROOT.gPad.GetRightMargin()
        ypos = 1.-ROOT.gPad.GetTopMargin()+0.01
        lx = ROOT.TLatex(0., 0., 'Z')
        lx.SetNDC(True)
        lx.SetTextAlign(13)
        lx.SetTextFont(42)
        lx.SetTextSize(0.04)
        lx.SetTextColor(16)
        lx.DrawLatex(0.014, .99, 'arXiv:xxx')

    #-- Draw the pulls graph
    pads[0].cd()
    alpha = 0.7
    if has_data:
        plot.Set(g_pulls, MarkerSize=0.8, LineWidth=2)
    if has_asimov:
        #g_pullsA.SetLineWidth(0)
        #g_pullsA.SetFillColor(plot.CreateTransparentColor(15, alpha))
        plot.Set(g_pullsA, MarkerSize=0.8, LineWidth=2)
        g_pullsA.SetLineColor(12)
        g_pullsA.SetMarkerColor(13)
        g_pullsA.SetMarkerStyle(24)
        g_pullsA.Draw('PSAME')
    if has_data:
        g_pulls.Draw('PSAME')

    #-- Draw impacts graphs
    pads[1].cd()
    alpha = 0.7

    #if has_asimov:
    #    g_impactsA_hi.SetFillColor(plot.CreateTransparentColor(46, alpha))
    #    g_impactsA_hi.SetLineWidth(0)
    #    g_impactsA_lo.SetLineWidth(0)
    #    g_impactsA_lo.SetFillColor(plot.CreateTransparentColor(38, alpha))
    #    g_impactsA_hi.SetMarkerSize(0)
    #    g_impactsA_lo.SetLineStyle(1)
    #    g_impactsA_lo.SetLineWidth(0)
    #    g_impactsA_hi.Draw('2 SAME')
    #    g_impactsA_lo.Draw('2 SAME')

    if has_data:
        g_impacts_hi.SetLineWidth(2)
        g_impacts_lo.SetLineWidth(2)
        g_impacts_hi.SetLineStyle(1)
        g_impacts_hi.SetLineColor(ROOT.kRed+1)
        g_impacts_lo.SetLineColor(ROOT.kAzure+4)
        g_impacts_lo.SetMarkerSize(0)
        g_impacts_hi.SetMarkerSize(0)
        g_impacts_hi.Draw('E SAME')
        g_impacts_lo.Draw('E SAME')
    pads[1].RedrawAxis()
    pads[1].RedrawAxis()

    #-- Legend, CMS logo, best-fit value
    # legendbox = [0.001, 0.995, 0.77, 0.915] #x1, y1, x2, y2
    legendbox = [0.23, 0.995, 0.995, 0.915] #x1, y1, x2, y2
    legendtextsize = 0.03
    legend = ROOT.TLegend(legendbox[0], legendbox[1], legendbox[2], legendbox[3], '', 'NBNDC')
    legend.SetFillStyle(0)
    legend.SetTextSize(legendtextsize)
    legend.SetNColumns(2)
    if has_data:
        legend.AddEntry(g_pulls, 'Fit constraint (obs., BG + #eta_{t})', 'LP')
        
    if has_asimov:
        legend.AddEntry(g_pullsA, 'Fit constraint (obs., BG only)', 'LP')
    #    legend.AddEntry(g_impactsA_hi, '+1#sigma impact (exp.)', 'F')
    #    legend.AddEntry(g_impactsA_lo, '-1#sigma impact (exp.)', 'F')
    if has_data:
        legend.AddEntry(g_impacts_hi, '+1#sigma impact (obs., BG + #eta_{t})', 'l')
        legend.AddEntry(g_impacts_lo, '-1#sigma impact (obs., BG + #eta_{t})', 'l')
    legend.Draw()

    #-- Draw CMS logo
    # docmslogo = True
    # cmslogopad = pads[0]
    # left = 0.35; top = 0.87 #Position of 'CMS' label
    # if args.blind:
    #     left = 0.87; top = 0.92
    #     if args.cms_label != "": left = 0.78
    docmslogo = True
    cmslogopad = pads[0]
    left = 0.01; top = .99 #Position of 'CMS' label
    if args.blind:
        left = 0.87; top = 0.92
        if args.cms_label != "": left = 0.78

    if docmslogo:

        #-- Use built-in CombineHarvester function
        # plot.DrawCMSLogo(pad=cmslogopad, cmsText='CMS', extraText=args.cms_label, iPosX=0,
        #        relPosX=cmslogo_extrax, relPosY=0, relExtraDY=0, extraText2='', cmsTextSize=0.3)

        #-- Draw labels manually (more flexible)
        CMS_text = ROOT.TLatex(left, top, "CMS")
        CMS_text.SetNDC()
        CMS_text.SetTextColor(ROOT.kBlack)
        CMS_text.SetTextFont(61)
        CMS_text.SetTextAlign(13)
        CMS_text.SetTextSize(0.055)

        extraText = ROOT.TLatex(left, top-0.06, args.cms_label)
        extraText.SetNDC()
        extraText.SetTextFont(52)
        extraText.SetTextAlign(13)
        extraText.SetTextSize(0.04)

        CMS_text.Draw('same') #Draw 'CMS'
        extraText.Draw('same') #Draw extra label, if any

    #-- Draw best-fit value and uncertainties
    # s_nom, s_hi, s_lo = GetRounded(POI_fit[1], POI_fit[2] - POI_fit[1], POI_fit[1] - POI_fit[0])
    # THIS PART DRAWS THE POI and sets how many digits to display
    s_nom, s_hi, s_lo = "{:.1f}".format(POI_fit[1]*xs_base), "{:.1f}".format( (POI_fit[2] - POI_fit[1])*xs_base), "{:.1f}".format( (POI_fit[1] - POI_fit[0])*xs_base)
    if s_hi == s_lo:
        unctext = ' \\pm %s' % s_hi
    else:
        unctext = '^{#plus%s}_{#minus%s}' % (s_hi, s_lo)
    if not args.blind:
        plot.DrawTitle(pad=pads[1], text='%s' % (
                '%s' % poi_translated
                )
            +' = %s%s%s pb' % (
                s_nom, unctext,
                '' if args.units is None else ' '+args.units
            ), align=3, textOffset=0.14, textSize=0.25)
    # pads[0].RedrawAxis()
    pads[1].RedrawAxis()

    #-- Write to .pdf file
    extra = ''
    if page == 0:
        extra = '('
    if page == n - 1:
        extra = ')'
    canv.Print('.pdf%s' % extra)
# //--------------------------------------------
# //--------------------------------------------
