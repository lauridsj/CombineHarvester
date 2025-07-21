#!/bin/bash

outdir=`realpath .`
fmt="pdf"

cmsswdir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13

impacts=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_w2p8/A_m400_w5p0_ll_nocut/A_m400_w5p0_ll_nocut_impacts_CMS_EtaT_norm_13TeV_g_0p0_fixed_all.json
impacts_asimov=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_w2p8/A_m400_w5p0_ll_nocut_asimov/A_m400_w5p0_ll_nocut_asimov_impacts_CMS_EtaT_norm_13TeV_g_0p0_fixed_all.json

cmssw-el7 --cleanenv --contain --bind /afs:/afs --bind /cvmfs:/cvmfs --bind /pnfs:/pnfs --bind /nfs:/nfs --bind /tmp:/host/tmp --env "CMSSW_DIR=${cmsswdir}" --pwd "$(realpath .)" --command-to-run "../scripts/condorRun.sh python ../scripts/customImpacts.py -i ${impacts} -ia ${impacts_asimov} -o impacts_nonps -t ../scripts/nuisance_map.json --per-page 20 --cms-label='Private work'"

cd /data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_thesis_old

python ../scripts/plot_2D_nll.py --point A_m365_w2p0,H_m365_w2p0 --tag 'll:obs' --tag-label Observed --draw-best-fit --formal --plot-format pdf --odir . --parameters CMS_EtaT_norm_13TeV,CMS_ChiT_norm_13TeV --intervals='-5,30;-15,20' --parameter-scales '6.43,6.43' --max-sigma 5 --cms-append='Private work' --drops='10,11;0,1 : 4,5;6,7 : 2,3;8,9'

cd /data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_thesis_fix

python ../scripts/plot_2D_nll.py --point A_m365_w2p0,H_m925_w3p0 --tag 'll_noetat:exp-b;ll_noetat:obs' --tag-label 'Expected;Observed' --plot-tag ll_noetat --draw-best-fit --formal --A343-background 0 --plot-format "${fmt}" --odir ${outdir} --parameters g1,g2 --intervals='0,1.6;0,2.5' --drops='-0.3,0.3;-0.5,0.5 : 0.2,0.25;1.3,1.4 -- '

cd ${outdir}

maindir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_w2p8_altbgs/A_m400_w5p0__H_m400_w5p0_ll_nonps
etatfile=${maindir}/A_m400_w5p0__H_m400_w5p0_ll_nonps_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
etatbatch=${maindir}/A_m400_w5p0__H_m400_w5p0_ll_nonps_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

#chitfile=${maindir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_ChiT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
#chitbatch=${maindir}/A_m365_w2p0__H_m365_w2p0_fixfrz_psfromws_ll_all_CMS_ChiT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

#etatchitdir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_thesis/A_m365_w2p0__H_m365_w2p0_ll
#etatchitfile=${etatchitdir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_ChiT_norm_13TeV__CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
#etatchitbatch=${etatchitdir}/A_m365_w2p0__H_m365_w2p0_ll_psfromws_ll_all_CMS_ChiT_norm_13TeV__CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

#bgonlyfile=${maindir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_b.root
#bgonlybatch=${maindir}/A_m365_w2p0__H_m365_w2p0_fixfrz_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_b.root

mbblldir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_mbbllspin_fine/A_m365_w2p0__H_m365_w2p0_ll_etatfit_oldsysts_sumbins2
mbbllfile=${mbblldir}/A_m365_w2p0__H_m365_w2p0_ll_etatfit_oldsysts_sumbins2_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
mbbllbatch=${mbblldir}/A_m365_w2p0__H_m365_w2p0_ll_etatfit_oldsysts_sumbins2_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
mbbllbestfit=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_mbbllspin_w2p8_fix/A_m365_w2p0__H_m365_w2p0_ll_nonps/A_m365_w2p0__H_m365_w2p0_ll_nonps_single_obs_g1_0p0_g2_0p0_fixed_CMS_EtaT_norm_13TeV.root

prefitdir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_thesis_old/A_m365_w2p0__H_m365_w2p0_ll
prefitfile=${prefitdir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

prefitdir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/suppmat_plots/A_m365_w2p0__H_m365_w2p0_ll
prefitfile=${prefitdir}/A_m365_w2p0__H_m365_w2p0_ll_fitdiagnostics_result_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root
prefitbatch=${prefitdir}/A_m365_w2p0__H_m365_w2p0_ll_psfromws_ll_all_CMS_EtaT_norm_13TeV_g1_0p0_g2_0p0_fixed_s.root

ahdir=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_thesis_fix/A_m365_w2p0__H_m925_w3p0_ll_noetat
ahfile=${ahdir}/A_m365_w2p0__H_m925_w3p0_ll_noetat_fitdiagnostics_result_s.root
ahbatch=${ahdir}/A_m365_w2p0__H_m925_w3p0_ll_noetat_psfromws_ll_all_s.root

common_opts="--odir ${outdir} --plot-formats ${fmt} --skip-each --xsec --panel both"

python3 ../scripts/plot_prepost_split.py ${common_opts} --ifile ${prefitfile} --batch ${prefitbatch} --skip-postfit --as-signal 'A,H,EtaT' --no-total --ignore 'ChiT' --prefit-signal-from default --log --cmslabel='Private work'

python3 ../scripts/plot_prepost_split.py ${common_opts} --ifile ${etatfile} --batch ${etatbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --ignore 'ChiT' --best-fit-from default --log --cmslabel='Private work'

python3 ../scripts/plot_prepost_split.py ${common_opts} --ifile ${etatfile} --batch ${etatbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --ignore 'ChiT' --best-fit-from default --log --panel-labels --project-to mtt --cmslabel='Private work'

python3 ../scripts/plot_prepost_split.py ${common_opts} --ifile ${etatfile} --batch ${etatbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --ignore 'ChiT' --best-fit-from default --panel-labels --project-to chel --mass-cut='-1,400' --cmslabel='Private work'

python3 ../scripts/plot_prepost_split.py ${common_opts} --ifile ${etatfile} --batch ${etatbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --ignore 'ChiT' --best-fit-from default --panel-labels --project-to chel --mass-cut='400,-1' --cmslabel='Private work'

python3 ../scripts/plot_prepost_split.py ${common_opts} --ifile ${mbbllfile} --batch ${mbbllbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --ignore 'ChiT' --best-fit-from ${mbbllbestfit} --log --plot-tag 'mbbllspin' --cmslabel='Private work'

python3 ../scripts/plot_prepost_split.py ${common_opts} --ifile ${mbbllfile} --batch ${mbbllbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --ignore 'ChiT' --best-fit-from ${mbbllbestfit} --log --panel-labels --project-to mbbll --plot-tag 'mbbllspin' --cmslabel='Private work'

python3 ../scripts/plot_prepost_split.py ${common_opts} --ifile ${mbbllfile} --batch ${mbbllbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --ignore 'ChiT' --best-fit-from ${mbbllbestfit} --panel-labels --project-to chel --mass-cut='-1,300' --plot-tag 'mbbllspin' --cmslabel='Private work'

python3 ../scripts/plot_prepost_split.py ${common_opts} --ifile ${mbbllfile} --batch ${mbbllbatch} --skip-prefit --as-signal 'EtaT' --skip-ah --ignore 'ChiT' --best-fit-from ${mbbllbestfit} --panel-labels --project-to chel --mass-cut='300,-1' --plot-tag 'mbbllspin' --cmslabel='Private work'

python3 ../scripts/plot_prepost_split.py ${common_opts} --ifile ${ahfile} --batch ${ahbatch} --skip-prefit --best-fit-from cross --log --cmslabel='Private work'

