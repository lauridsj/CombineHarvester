#!/bin/bash

outdir=.
fmt="pdf,png,svg"

maindir=/data/dust/user/afiqaize/cms/ahtt_run2ul_stat_200803/combine/CMSSW_10_2_13/src/CombineHarvester/ahtt/for_top-24-007_250109/A_m365_w2p0__H_m365_w2p0_ll
etatfile=${maindir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
etatbatch=${maindir}/A_m365_w2p0__H_m365_w2p0_fixfrz_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

chitfile=${maindir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_ChiT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
chitbatch=${maindir}/A_m365_w2p0__H_m365_w2p0_fixfrz_psfromws_ll_all_CMS_ChiT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

etatchitfile=${maindir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_ChiT_norm_13TeV__CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
etatchitbatch=${maindir}/A_m365_w2p0__H_m365_w2p0_fixfrz_psfromws_ll_all_CMS_ChiT_norm_13TeV__CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

bgonlyfile=${maindir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_b.root
bgonlybatch=${maindir}/A_m365_w2p0__H_m365_w2p0_fixfrz_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_b.root

mbblldir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_mbbllspin_w2p8_fix/A_m365_w2p0__H_m365_w2p0_ll_nonps
mbbllfile=${mbblldir}/A_m365_w2p0__H_m365_w2p0_ll_nonps_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
mbbllbatch=${mbblldir}/A_m365_w2p0__H_m365_w2p0_ll_nonps_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

scenarios=("--ifile ${etatfile} --batch ${etatbatch} --as-signal EtaT --ignore ChiT --best-fit-from default" "--ifile ${chitfile} --batch ${chitbatch} --as-signal ChiT --ignore EtaT --best-fit-from default" "--ifile ${etatchitfile} --batch ${etatchitbatch} --as-signal EtaT,ChiT --best-fit-from default" "--ifile ${bgonlyfile} --batch ${bgonlybatch} --as-signal '' --ignore EtaT,ChiT")


for panel in both lower; do
   
    for ((i = 0; i < ${#scenarios[@]}; i++)); do

        echo ${scenarios[$i]}

        python3 ../scripts/plot_prepost_split.py --odir ${outdir} ${scenarios[$i]} --plot-formats "${fmt}" --log --skip-each --skip-prefit --skip-ah --panel ${panel} --xsec

        python3 ../scripts/plot_prepost_split.py --odir ${outdir} ${scenarios[$i]} --plot-formats "${fmt}" --log --skip-each  --skip-prefit --skip-ah --panel ${panel} --xsec --panel-labels --split-bins

        python3 ../scripts/plot_prepost_split.py --odir ${outdir} ${scenarios[$i]} --plot-formats "${fmt}" --log --skip-each  --skip-prefit --skip-ah --panel ${panel} --xsec --panel-labels --project-to mtt

        for angle in chel chan; do
            python3 ../scripts/plot_prepost_split.py --odir ${outdir} ${scenarios[$i]} --plot-formats "${fmt}" --skip-each  --skip-prefit --skip-ah --panel ${panel} --xsec --panel-labels --project-to ${angle}

            python3 ../scripts/plot_prepost_split.py --odir ${outdir} ${scenarios[$i]} --plot-formats "${fmt}" --skip-each  --skip-prefit --skip-ah --panel ${panel} --xsec --panel-labels --project-to ${angle} --mass-cut 320,1460
        done
    done

    # s+b, mbbll x chel x chan, combined
    python3 ../scripts/plot_prepost_split.py --plot-tag mbbllspin --odir ${outdir} --ifile ${mbbllfile} --plot-formats "${fmt}" --log --skip-each --batch ${mbbllbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --panel ${panel} --xsec --ignore ChiT

    # s+b, mbbll x chel x chan, split
    python3 ../scripts/plot_prepost_split.py --plot-tag mbbllspin --odir ${outdir} --ifile ${mbbllfile} --plot-formats "${fmt}" --log --skip-each --batch ${mbbllbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --panel ${panel} --xsec --ignore ChiT --split-bins --panel-labels

    # s+b, mbbll x chel x chan, mbbll inclusive
    python3 ../scripts/plot_prepost_split.py --plot-tag mbbllspin --odir ${outdir} --ifile ${mbbllfile} --plot-formats "${fmt}" --log --skip-each --batch ${mbbllbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --panel ${panel} --xsec --ignore ChiT --panel-labels --project-to mbbll --logx

done
