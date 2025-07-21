#!/bin/bash

outdir=.
fmt="pdf,svg,png"
panel="both"

maindir=/data/dust/user/afiqaize/cms/ahtt_run2ul_stat_200803/combine/CMSSW_10_2_13/src/CombineHarvester/ahtt/for_top-24-007_250109/A_m365_w2p0__H_m365_w2p0_ll
etatfile=${maindir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
etatbatch=${maindir}/A_m365_w2p0__H_m365_w2p0_fixfrz_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

chitfile=${maindir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_ChiT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
chitbatch=${maindir}/A_m365_w2p0__H_m365_w2p0_fixfrz_psfromws_ll_all_CMS_ChiT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

etatchitfile=${maindir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_ChiT_norm_13TeV__CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
etatchitbatch=${maindir}/A_m365_w2p0__H_m365_w2p0_fixfrz_psfromws_ll_all_CMS_ChiT_norm_13TeV__CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

bgonlyfile=${maindir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_b.root
bgonlybatch=${maindir}/A_m365_w2p0__H_m365_w2p0_fixfrz_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_b.root

#mbblldir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_mbbllspin_w2p8_fix/A_m365_w2p0__H_m365_w2p0_ll_nonps
#mbbllfile=${mbblldir}/A_m365_w2p0__H_m365_w2p0_ll_nonps_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
#mbbllbatch=${mbblldir}/A_m365_w2p0__H_m365_w2p0_ll_nonps_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

mbblldir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_mbbllspin_fine/A_m365_w2p0__H_m365_w2p0_ll_etatfit_oldsysts_sumbins2
mbbllfile=${mbblldir}/A_m365_w2p0__H_m365_w2p0_ll_etatfit_oldsysts_sumbins2_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
mbbllbatch=${mbblldir}/A_m365_w2p0__H_m365_w2p0_ll_etatfit_oldsysts_sumbins2_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
mbbllbestfit=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_mbbllspin_w2p8_fix/A_m365_w2p0__H_m365_w2p0_ll_nonps/A_m365_w2p0__H_m365_w2p0_ll_nonps_single_obs_g1_0p0_g2_0p0_fixed_CMS_EtaT_norm_13TeV.root

python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${etatfile} --batch ${etatbatch} --as-signal EtaT --ignore ChiT --best-fit-from default --skip-prefit --plot-formats "${fmt}" --log --skip-each --skip-ah --panel ${panel} --xsec --panel-labels --split-bins

python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${etatfile} --batch ${etatbatch} --as-signal EtaT --ignore ChiT --best-fit-from default --skip-prefit --plot-formats "${fmt}" --skip-each --skip-ah --panel ${panel} --xsec --panel-labels --project-to chel --mass-cut='-1,360'

python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${mbbllfile} --batch ${mbbllbatch} --as-signal EtaT --ignore ChiT --best-fit-from ${mbbllbestfit} --skip-prefit --plot-formats "${fmt}" --log --skip-each --skip-ah --panel ${panel} --xsec --panel-labels --project-to mbbll --plot-tag mbbllspin

python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${bgonlyfile} --batch ${bgonlybatch} --as-signal '' --ignore EtaT,ChiT --skip-prefit --plot-formats "${fmt}" --log --skip-each --skip-ah --panel ${panel} --xsec --panel-labels --split-bins

python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${bgonlyfile} --batch ${bgonlybatch} --as-signal '' --ignore EtaT,ChiT --skip-prefit --plot-formats "${fmt}" --skip-each --skip-ah --panel ${panel} --xsec --panel-labels --project-to chel --mass-cut='-1,360'

# the EtaT vs ChiT scan is missing, Afiq made that one, forgot where the input files are aa