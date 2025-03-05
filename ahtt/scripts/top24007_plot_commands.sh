#!/bin/bash

outdir=.
fmt="pdf,svg,png"

maindir=/data/dust/user/afiqaize/cms/ahtt_run2ul_stat_200803/combine/CMSSW_10_2_13/src/CombineHarvester/ahtt/for_top-24-007_250109/A_m365_w2p0__H_m365_w2p0_ll
etatfile=${maindir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
etatbatch=${maindir}/A_m365_w2p0__H_m365_w2p0_fixfrz_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

bgonlyfile=${maindir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_b.root
bgonlybatch=${maindir}/A_m365_w2p0__H_m365_w2p0_fixfrz_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_b.root

mbblldir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_mbbllspin_w2p8_fix/A_m365_w2p0__H_m365_w2p0_ll_nonps
mbbllfile=${mbblldir}/A_m365_w2p0__H_m365_w2p0_ll_nonps_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
mbbllbatch=${mbblldir}/A_m365_w2p0__H_m365_w2p0_ll_nonps_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

for panel in both lower; do
   
    # s+b mtt x chel x chan, combined
    python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${etatfile} --plot-formats "${fmt}" --log --skip-each --batch ${etatbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --panel ${panel} --xsec --ignore ChiT --best-fit-from default

    # b mtt x chel x chan, combined
    python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${bgonlyfile} --plot-formats "${fmt}" --log --skip-each --batch ${bgonlybatch} --skip-prefit --as-signal '' --skip-ah --panel ${panel} --ignore 'EtaT,ChiT' 

    # s+b mtt x chel x chan, split
    python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${etatfile} --plot-formats "${fmt}" --log --skip-each --batch ${etatbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --panel ${panel} --xsec --ignore ChiT --best-fit-from default --split-bins --panel-labels 

    # b mtt x chel x chan, split
    python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${bgonlyfile} --plot-formats "${fmt}" --log --skip-each --batch ${bgonlybatch} --skip-prefit --as-signal '' --skip-ah --panel ${panel} --ignore 'EtaT,ChiT' --split-bins --panel-labels

    # s+b mtt x chel x chan, mtt inclusive
    python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${etatfile} --plot-formats "${fmt}" --log --skip-each --batch ${etatbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --panel ${panel} --xsec --ignore ChiT --best-fit-from default --panel-labels --project-to mtt

    # b mtt x chel x chan, mtt inclusive
    python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${bgonlyfile} --plot-formats "${fmt}" --log --skip-each --batch ${bgonlybatch} --skip-prefit --as-signal '' --skip-ah --panel ${panel} --ignore 'EtaT,ChiT' --panel-labels --project-to mtt

    for angle in chel chan; do
        # s+b mtt x chel x chan, angles, mtt slices
        python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${etatfile} --plot-formats "${fmt}" --skip-each --batch ${etatbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --panel ${panel} --xsec --ignore ChiT --best-fit-from default --panel-labels --project-to ${angle}

        # b mtt x chel x chan, angles, mtt slices
        python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${bgonlyfile} --plot-formats "${fmt}" --skip-each --batch ${bgonlybatch} --skip-prefit --as-signal '' --skip-ah --panel ${panel} --ignore 'EtaT,ChiT' --panel-labels --project-to ${angle}

        # s+b mtt x chel x chan, angles inclusive
        python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${etatfile} --plot-formats "${fmt}" --skip-each --batch ${etatbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --panel ${panel} --xsec --ignore ChiT --best-fit-from default --panel-labels --project-to ${angle} --mass-cut 320,1460

        # b mtt x chel x chan, angles inclusive
        python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${bgonlyfile} --plot-formats "${fmt}" --skip-each --batch ${bgonlybatch} --skip-prefit --as-signal '' --skip-ah --panel ${panel} --ignore 'EtaT,ChiT' --panel-labels --project-to ${angle} --mass-cut 320,1460
    done

    # s+b, mbbll x chel x chan, combined
    python3 ../scripts/plot_prepost_split.py --plot-tag mbbllspin --odir ${outdir} --ifile ${mbbllfile} --plot-formats "${fmt}" --log --skip-each --batch ${mbbllbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --panel ${panel} --xsec --ignore ChiT

    # s+b, mbbll x chel x chan, split
    python3 ../scripts/plot_prepost_split.py --plot-tag mbbllspin --odir ${outdir} --ifile ${mbbllfile} --plot-formats "${fmt}" --log --skip-each --batch ${mbbllbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --panel ${panel} --xsec --ignore ChiT --split-bins --panel-labels

    # s+b, mbbll x chel x chan, mbbll inclusive
    python3 ../scripts/plot_prepost_split.py --plot-tag mbbllspin --odir ${outdir} --ifile ${mbbllfile} --plot-formats "${fmt}" --log --skip-each --batch ${mbbllbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --panel ${panel} --xsec --ignore ChiT --panel-labels --project-to mbbll --logx

done
