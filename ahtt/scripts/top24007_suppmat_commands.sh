#!/bin/bash

outdir=.
fmt="pdf,svg,png"
panel="both"

maindir=/data/dust/user/afiqaize/cms/ahtt_run2ul_stat_200803/combine/CMSSW_10_2_13/src/CombineHarvester/ahtt/for_top-24-007_250109/A_m365_w2p0__H_m365_w2p0_ll
etatfile=${maindir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
etatbatch=${maindir}/A_m365_w2p0__H_m365_w2p0_ll_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

etatprefitbatch=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/suppmat_plots/A_m365_w2p0__H_m365_w2p0_ll/A_m365_w2p0__H_m365_w2p0_ll_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

impacts=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_short_paper/A_m400_w5p0_ll_bb4l_herwig_gauss_nonorm/A_m400_w5p0_ll_bb4l_herwig_gauss_nonorm_impacts_CMS_EtaT_norm_13TeV_g_0p0_fixed_all.json
impacts_asimov=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_short_paper/A_m400_w5p0_ll_bb4l_herwig_gauss_nonorm_asimov_try2/A_m400_w5p0_ll_asimov_try2_impacts_CMS_EtaT_norm_13TeV_g_0p0_fixed_all.json

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

bb4ldir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_w2p8_altbgs/A_m400_w5p0__H_m400_w5p0_ll_bb4l
bb4lfile=${bb4ldir}/A_m400_w5p0__H_m400_w5p0_ll_bb4l_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
bb4lbatch=${bb4ldir}/A_m400_w5p0__H_m400_w5p0_ll_bb4l_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

hvqdir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_w2p8_altbgs/A_m400_w5p0__H_m400_w5p0_ll_nonps
hvqfile=${hvqdir}/A_m400_w5p0__H_m400_w5p0_ll_nonps_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
hvqbatch=${hvqdir}/A_m400_w5p0__H_m400_w5p0_ll_nonps_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

herwigdir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_w2p8_altbgs/A_m400_w5p0__H_m400_w5p0_ll_herwig
herwigfile=${herwigdir}/A_m400_w5p0__H_m400_w5p0_ll_herwig_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
herwigbatch=${herwigdir}/A_m400_w5p0__H_m400_w5p0_ll_herwig_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

amcatnlodir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_w2p8_altbgs/A_m400_w5p0__H_m400_w5p0_ll_amcatnlo
amcatnlofile=${amcatnlodir}/A_m400_w5p0__H_m400_w5p0_ll_amcatnlo_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
amcatnlobatch=${amcatnlodir}/A_m400_w5p0__H_m400_w5p0_ll_amcatnlo_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

scenarios=("--ifile ${etatfile} --batch ${etatbatch} --as-signal EtaT --ignore ChiT --best-fit-from default --skip-prefit --cmslabel Supplementary" "--ifile ${etatfile} --batch ${etatprefitbatch} --as-signal EtaT --ignore ChiT --skip-postfit --prefit-signal-from default --cmslabel Supplementary --normalize" "--ifile ${etatfile} --batch ${etatprefitbatch} --as-signal EtaT --ignore ChiT --skip-postfit --prefit-signal-from default --cmslabel Supplementary" "--ifile ${chitfile} --batch ${chitbatch} --as-signal ChiT --ignore EtaT --best-fit-from default --skip-prefit --cmslabel Supplementary" "--ifile ${etatchitfile} --batch ${etatchitbatch} --as-signal EtaT,ChiT --best-fit-from default --skip-prefit --cmslabel Supplementary" "--ifile ${bgonlyfile} --batch ${bgonlybatch} --as-signal '' --ignore EtaT,ChiT --skip-prefit --cmslabel Supplementary")

scenarios_altbgs=("--ifile ${hvqfile} --batch ${bb4lbatch} --as-signal EtaT --ignore ChiT --best-fit-from default --plot-tag hvq --generator-label hvq --cmslabel Supplementary" "--ifile ${bb4lfile} --batch ${bb4lbatch} --as-signal EtaT --ignore ChiT --best-fit-from default --plot-tag bb4l --generator-label bb4l --cmslabel Supplementary" "--ifile ${herwigfile} --batch ${herwigbatch} --as-signal EtaT --ignore ChiT --best-fit-from default --plot-tag herwig --generator-label herwig --cmslabel Supplementary" "--ifile ${amcatnlofile} --batch ${amcatnlobatch} --as-signal EtaT --ignore ChiT --best-fit-from default --plot-tag amcatnlo --generator-label amcatnlo --cmslabel Supplementary")

for ((i = 0; i < ${#scenarios[@]}; i++)); do

    echo ${scenarios[$i]}

    python3 ../scripts/plot_prepost_split.py --odir ${outdir} ${scenarios[$i]} --plot-formats "${fmt}" --log --skip-each --skip-ah --panel ${panel} --xsec

    python3 ../scripts/plot_prepost_split.py --odir ${outdir} ${scenarios[$i]} --plot-formats "${fmt}" --log --skip-each --skip-ah --panel ${panel} --xsec --panel-labels --split-bins

    python3 ../scripts/plot_prepost_split.py --odir ${outdir} ${scenarios[$i]} --plot-formats "${fmt}" --log --skip-each --skip-ah --panel ${panel} --xsec --panel-labels --project-to mtt

    for angle in chel chan; do
        python3 ../scripts/plot_prepost_split.py --odir ${outdir} ${scenarios[$i]} --plot-formats "${fmt}" --skip-each --skip-ah --panel ${panel} --xsec --panel-labels --project-to ${angle} --mass-cut 320,1460

        python3 ../scripts/plot_prepost_split.py --odir ${outdir} ${scenarios[$i]} --plot-formats "${fmt}" --skip-each --skip-ah --panel ${panel} --xsec --panel-labels --project-to ${angle} --mass-cut='-1,360'

        python3 ../scripts/plot_prepost_split.py --odir ${outdir} ${scenarios[$i]} --plot-formats "${fmt}" --skip-each --skip-ah --panel ${panel} --xsec --panel-labels --project-to ${angle} --mass-cut='800,1050'
    done
done

python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${mbbllfile} --batch ${mbbllbatch} --as-signal EtaT --ignore ChiT --best-fit-from ${mbbllbestfit} --skip-prefit --plot-formats "${fmt}" --log --skip-each --skip-ah --panel ${panel} --xsec --panel-labels --split-bins --plot-tag mbbllspin

python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${mbbllfile} --batch ${mbbllbatch} --as-signal EtaT --ignore ChiT --best-fit-from ${mbbllbestfit} --skip-prefit --plot-formats "${fmt}" --log --skip-each --skip-ah --panel ${panel} --xsec --plot-tag mbbllspin

for ((i = 0; i < ${#scenarios_altbgs[@]}; i++)); do

    echo ${scenarios_altbgs[$i]}

    python3 ../scripts/plot_prepost_split.py --odir ${outdir} ${scenarios_altbgs[$i]} --plot-formats "${fmt}" --log --skip-each --skip-prefit --skip-ah --panel ${panel} --xsec

done

python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${etatfile} --as-signal EtaT --ignore ChiT --prefit-signal-from default --plot-formats "${fmt}" --log --skip-postfit --skip-ah --panel ${panel} --xsec --cmslabel Supplementary

python3 ../scripts/plot_prepost_split.py --odir ${outdir} --ifile ${etatfile} --as-signal EtaT --ignore ChiT --best-fit-from default --plot-formats "${fmt}" --log --skip-prefit --skip-ah --panel ${panel} --xsec --cmslabel Supplementary

python3 ../scripts/plot_correlation.py --infile ${etatfile} --outfile correlation_etat.pdf --signal EtaT --only 21 --nuisance_map ../scripts/nuisance_map.json --impacts ${impacts} --cmslabel Supplementary

cmsswdir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13

cmssw-el7 --cleanenv --contain --bind /afs:/afs --bind /cvmfs:/cvmfs --bind /pnfs:/pnfs --bind /nfs:/nfs --bind /tmp:/host/tmp --env "CMSSW_DIR=${cmsswdir}" --pwd "$(realpath .)" --command-to-run "../scripts/condorRun.sh python ../scripts/customImpacts.py -i ${impacts} -ia ${impacts_asimov} -o impacts_etat_obs_exp -t ../scripts/nuisance_map.json --per-page 20 --cms-label Supplementary"

cmssw-el7 --cleanenv --contain --bind /afs:/afs --bind /cvmfs:/cvmfs --bind /pnfs:/pnfs --bind /nfs:/nfs --bind /tmp:/host/tmp --env "CMSSW_DIR=${cmsswdir}" --pwd "$(realpath .)" --command-to-run "../scripts/condorRun.sh python ../scripts/customImpacts.py -i ${impacts} -o impacts_etat_obs -t ../scripts/nuisance_map.json --per-page 20 --cms-label Supplementary"



