#!/bin/bash

fmt='pdf,svg,png'
outdir=.

outdir=`realpath ${outdir}`

scriptdir=`dirname ${BASH_SOURCE[0]}`

#PRE/POST

indir_withetat=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_long_paper/A_m365_w2p0__H_m425_w3p0_lx_withetat
tag_withetat=lx_withetat
indir_noetat=/data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/workdir_long_paper/A_m365_w2p0__H_m425_w3p0_lx_noetat
tag_noetat=lx_noetat

for channel in ll l4pj l3j; do
    python3 ${scriptdir}/plot_prepost_combined_etat.py --odir $outdir --ifile ${indir_withetat}/A_*__H_*_${tag_withetat}_fitdiagnostics_result_s.root --ifileah ${indir_noetat}/A_*__H_*_${tag_noetat}_fitdiagnostics_result_s.root --plot-formats "${fmt}" --log --batch ${indir_withetat}/A_*__H_*_${tag_withetat}_psfromws_${channel}_all_s.root --batchah ${indir_noetat}/A_*__H_*_${tag_noetat}_psfromws_${channel}_all_s.root --best-fit-from cross
done

#CONTOURS

cd /data/dust/user/lauridsj/ah/CMSSW_10_2_13/src/CombineHarvester/ahtt/fc_plots_newnew

masses=('m365' 'm500' 'm750' 'm1000')
width='w2p0'

for ((i = 0; i < ${#masses[@]}; i++)); do
for ((j = 0; j < ${#masses[@]}; j++)); do

    pair="A_${masses[$i]}_${width},H_${masses[$j]}_${width}"
    for tag in lx lx_etat; do
        indir="A_${masses[$i]}_${width}__H_${masses[$j]}_${width}_${tag}"
        if [ -e ${indir} ]; then
            python3 ${scriptdir}/plot_contour.py --point "${pair}" --contour "${tag}/exp-b,obs" --odir ${outdir} --label 'Expected;Observed' --formal --A343-background 1 --draw-best-fit --plot-formats "${fmt}" --autoscale
        fi
    done

done
done

pair='A_m365_w2p0,H_m425_w3p0'

tag='lx_no_etat'
python3 ${scriptdir}/plot_contour.py --point "${pair}" --contour "${tag}/exp-b,obs" --odir ${outdir} --label 'Expected;Observed' --formal --A343-background 0 --draw-best-fit --plot-formats "${fmt}" --intervals '0,1.6;0,1.6' --proper-sigma --max-sigma 3 --plot-tag 'lx_smtt'

tag='lx_eww_etat'
python3 ${scriptdir}/plot_contour.py --point "${pair}" --contour "${tag}/exp-b,obs" --odir ${outdir} --label 'Expected;Observed' --formal --A343-background 1 --draw-best-fit --plot-formats "${fmt}" --intervals '0,1.6;0,1.6' --proper-sigma --max-sigma 3 --plot-tag 'lx_etat'

cd -

# 1D LIMITS

mkdir lim1D && cd lim1D
for idir in smtt etat; do for ichan in lx ll lj; do mkdir -p ${idir}/${ichan}; done; done
fmt="pdf,png,svg"

# limits w/o etat
basedir="/data/dust/user/afiqaize/cms/ahtt_run2ul_stat_200803/combine/CMSSW_10_2_13/src/CombineHarvester/ahtt/cleanup_1D_240205"
python3 ../../scripts/plot_limit.py --tag "lx_smtt" --odir smtt/lx --read-from "${basedir}" --observed --formal --A343-background 0 --plot-formats "${fmt}"
python3 ../../scripts/plot_limit.py --tag "ll_smtt" --odir smtt/ll --read-from "${basedir}" --observed --formal --cms-append "Supplementary" --A343-background 0 --plot-formats "${fmt}"
basedir="/data/dust/user/afiqaize/cms/ahtt_run2ul_stat_200803/combine/CMSSW_10_2_13/src/CombineHarvester/ahtt/unblind_stage3_1D_231205"
python3 ../../scripts/plot_limit.py --tag "lj_smtt" --odir smtt/lj --read-from "${basedir}" --observed --formal --cms-append "Supplementary" --A343-background 0 --plot-formats "${fmt}"

# limits with etat
basedir="/data/dust/user/afiqaize/cms/ahtt_run2ul_stat_200803/combine/CMSSW_10_2_13/src/CombineHarvester/ahtt/ah_etatfloat_240318"
python3 ../../scripts/plot_limit.py --tag "lx" --odir etat/lx --read-from "${basedir}" --observed --formal --A343-background 1 --plot-formats "${fmt}"
python3 ../../scripts/plot_limit.py --tag "ll" --odir etat/ll --read-from "${basedir}" --observed --formal --cms-append "Supplementary" --A343-background 1 --plot-formats "${fmt}"
python3 ../../scripts/plot_limit.py --tag "lj" --odir etat/lj --read-from "${basedir}" --observed --formal --cms-append "Supplementary" --A343-background 1 --plot-formats "${fmt}"