# How To Run Fits

This is a non-exhaustive list of the different types of fits that can be run with this wrapper.
All commands should be executed in a fresh shell, with no container and nothing sourced save the usual CMS defaults.

## Setup

Copypaste the commands [here](setup).

Some environment variables are colleced in `env.sh`:
```bash
source env.sh
```

## Making workspaces

Datacard & workspace creation can be done locally or on condor. For a single A/H fit, a workspace is created for each mass/width point, which is a lot.

The command looks like this:
```bash
./../scripts/submit_point.py --mode 'datacard,validate' --point "${points}" --tag "${tag}" --year "${years}" --channel "${channels}" --keep "${keeps}" --drop "${drops}" --sushi-kfactor --lnN-under-threshold  --unblind --exclude-process 'EtaT,ChiT'
```

The arguments mean the following:

- `--points`: The A/H mass/width points to run. This uses matching, so giving e.g. `w5p0` will run all 5% width points.
- `--tag`: A unique tag that will be appended to the created files for organization.
- `--year`: The years to include. Use `${years_run2}` as defined in `env.sh` for full Run 2.
- `--channel`: The channels to include. Use `${channels_ll}` or `${channels_lx}` as defined in `env.sh` for dilepton or ll+lj combination.
- `--keep`: Systematics to keep. This needs to be included. The setup for A/H is stored as `${keeps_AH}` in `env.sh`. 
- `--drop`: Systematics to drop. This needs to be included. The setup for A/H is stored as `${drops_AH}` in `env.sh`. 
- `--sushi-kfactor`: Use a k-factor for the A/H signals.
- `--lnN-under-threshold`: For shape systematics which have little shape effect, use a normalization instead.
- `--unblind`: Fit on real data instead of pseudodata.
- ` --exclude-process`: Dont consider the given processes. Here used to not include EtaT and ChiT, which are otherwise in the files.

So for example, to produce workspaces for A/H, all masses, 5% width, for the dilepton channels, with no EtaT and ChiT background:

```bash
mkdir workdir
cd workdir
./../scripts/submit_point.py --mode 'datacard,validate' --sushi-kfactor --lnN-under-threshold --year "${years_run2}" --channel "${channels_ll}" --tag ll --keep "${keeps_AH}" --drop "${drops_AH}" --unblind --point "w5p0" --exclude-process 'EtaT,ChiT'
```

There are actually two workspaces produced for a single A/H fit (in the same `.tar.gz` file): One contains only $g$ as POI, while the other contains both $g$ and a signal strength $r$. The latter is used for the limit setting using the raster scan method, while the former should be used for all other purposes.

To produce workspaces for fitting EtaT, one still needs to give an arbitrary A/H point. This is Really Hacky (TM) and should probably not be used in the future. The set of systematics is slightly different (it includes the bb4l and Herwig NPs). For example:

```bash
./../scripts/submit_point.py --mode 'datacard,validate' --sushi-kfactor --lnN-under-threshold --year "${years_run2}" --channel "${channels_ll}" --tag ll --keep "${keeps_etat}" --drop "${drops_etat}" --unblind --point "A_m400_w5p0" --exclude-process 'ChiT' --poi-set 'CMS_EtaT_norm_13TeV' --g-value 0 --fix-poi --one-poi
```

Finally, one can produce 2D workspaces with one A and one H signal. This is also sometimes required for EtaT since some methods (most importantly, postfit plots) are only implemented for 2D workspaces. For A/H:

```bash
./../scripts/submit_twin.py --mode 'datacard,validate' --sushi-kfactor --lnN-under-threshold --year "${years_run2}" --channel "${channels_ll}" --tag ll --keep "${keeps_AH}" --drop "${drops_AH}" --unblind --point 'A_m400_w5p0,H_m400_w5p0'
```

For EtaT:

```bash
./../scripts/submit_twin.py --mode 'datacard,validate' --sushi-kfactor --lnN-under-threshold --year "${years_run2}" --channel "${channels_ll}" --tag ll --keep "${keeps_etat}" --drop "${drops_etat}" --unblind --point 'A_m400_w5p0,H_m400_w5p0' --poi-set 'CMS_EtaT_norm_13TeV' --g-values '0,0' --fix-poi --exclude-process 'ChiT'
```

For EtaT and ChiT, used for the parity scan:

```bash
./../scripts/submit_twin.py --mode 'datacard,validate' --sushi-kfactor --lnN-under-threshold --year "${years_run2}" --channel "${channels_ll}" --tag ll --keep "${keeps_etat}" --drop "${drops_etat}" --unblind --point 'A_m400_w5p0,H_m400_w5p0' --poi-set 'CMS_EtaT_norm_13TeV,CMS_ChiT_norm_13TeV' --g-values '0,0' --fix-poi --exclude-process ''
```

Further important arguments for all commands (datacards and fitting) are:
- `--local`: Run locally without submitting to condor.
- `--runtime`: Runtime in seconds for the condor submission. Default is 10800 (3h).
- `--memory`: Memory requirement in MB for the condor submission. Default is 2048. Sometimes very large combination fits exceed this and crash, in this case increase to 4096.
- `--output-tag`: Use this if you want to run the same mode multiple times with different settings within the same workspace (i.e. you set `--tag` to the original tag of the workspace, and `--output-tag` to something else).

## Best fits + uncertainties

After the workspaces are created, once can do a best fit + uncertainty estimation for a single parameter.
For single A/H, estimating $g$ (note the `--one-poi` to use the $g$-only workspace instead of the raster scan one):

```bash
./../scripts/submit_point.py --point "${points}" --mode single --tag ll --one-poi --unblind
```

For dual A/H, estimating $g_A$ and $g_H$ simultaneously:

```bash
./../scripts/submit_twin.py --point "${point}" --mode cross --tag ll --unblind
```

For EtaT, estimating $\mu(\eta_t)$:

```bash
./../scripts/submit_point.py --point 'A_m400_w5p0' --mode single --tag ll --unblind --poi-set 'CMS_EtaT_norm_13TeV' --g-value 0 --one-poi --fix-poi
```

For EtaT and ChiT:

```bash
./../scripts/submit_twin.py --point 'A_m400_w5p0,H_m400_w5p0' --mode cross --tag ll --unblind --poi-set 'CMS_EtaT_norm_13TeV,CMS_ChiT_norm_13TeV' --g-values '0,0' --fix-poi
```

In all cases, the results can be extracted from the created ROOT file by accessing the `limit` branch, as created by combine. E.g. for simultaneous A/H:

```bash
root -b -l ./A_m400_w5p0__H_m400_w5p0_ll/A_m400_w5p0__H_m400_w5p0_ll_cross_obs.root
limit->Scan("g1:g2:quantileExpected")
```

For the single fits, the result is also graciously printed in the console and/or logfile.

In some cases, it might be required to run the fits with a higher precision (applies also to other modes). This can be achieved by adding the followint arguments to the commands: `--fit-strategy 2 --use-hesse`. Beware: this is MUCH slower.

## A/H limits

This needs to be done with the raster scan method, i.e. with two POIs in the workspace:

```bash
./../scripts/submit_point.py --point "${points}" --mode limit --tag ll --unblind
```

This submits 6 jobs per A/H point in the usual settings, so beware of many jobs.

## Impacts

Impacts for A/H (only implemented for single A/H):

```bash
./../scripts/submit_point.py --point "${points}" --mode impact --one-poi --tag ll --unblind
```

Impacts for EtaT:

```bash
./../scripts/submit_point.py --point 'A_m400_w5p0' --mode impact --tag ll --unblind --poi-set CMS_EtaT_norm_13TeV --g-value 0 --one-poi --fix-poi
```

If you want Asimov impacts for EtaT with a non-zero expected EtaT signal strength, you need to cheat a little:

```bash
./../scripts/submit_point.py --point 'A_m400_w5p0' --mode impact --tag ll --poi-set CMS_EtaT_norm_13TeV --g-value 0 --one-poi --fix-poi --extra-option='--setParameters CMS_EtaT_norm_13TeV=1'
```

And if you want to have background-only pulls, you have to cheat even more (and waste computing power):

```bash
./../scripts/submit_point.py --point 'A_m400_w5p0' --mode impact --tag ll --poi-set EWK_const --g-value 0 --one-poi --fix-poi --freeze-zero CMS_EtaT_norm_13TeV
```

After the impacts have run through, you need to combine the files from the different jobs into one:

For A/H:
```bash
./../scripts/merge_pull.py --point "${point}" --tag ll
```

For EtaT:
```bash
./../scripts/merge_pull.py --point 'A_m400_w5p0' --tag ll --g-value 0 --fix-poi --poi-name CMS_EtaT_norm_13TeV
```

## Pre/postfit plots

Only implemented for twin workspaces. Also, extremely slow, always submit with ~24h runtime. Also also, can be unstable, so `--fit-strategy 2 --use-hesse` is recommended. If fits still fail or one wants to speed up things, one can also add `--freeze-post mcstat` to ignore MC statistical uncertainties for the error bands.

Two steps are needed. First, create the actual pre/postfit. E.g. for simultaneous A/H:

```bash
./../scripts/submit_twin.py --point "${point}" --mode prepost --tag ll --unblind --job-time 86400
```

Then, after step one has finished, one needs to merge the different years/channels into one plot, taking the correlations into account. This is one job per plot. For example, combining all ll channels and years:

```bash
./../scripts/submit_twin.py --point "${point}" --mode psfromws --prepost-merge ll_all --tag ll --unblind --job-time 86400
```

Similar two step procedures for e.g. A only by fixing H to zero (note it still uses `submit_twin`!):

```bash
./../scripts/submit_twin.py --point "${point}" --mode prepost --tag ll --unblind --g-values='-1,0' --fix-poi --job-time 86400
./../scripts/submit_twin.py --point "${point}" --mode psfromws --prepost-merge ll_all --tag ll --unblind --g-values='-1,0' --fix-poi --job-time 86400
```

For EtaT only:

```bash
./../scripts/submit_twin.py --point 'A_m400_w5p0,H_m400_w5p0' --mode prepost --tag ll --unblind --poi-set 'CMS_EtaT_norm_13TeV' --g-values '0,0' --fix-poi --job-time 86400
./../scripts/submit_twin.py --point 'A_m400_w5p0,H_m400_w5p0' --mode psfromws --prepost-merge ll_all --tag ll --unblind --poi-set 'CMS_EtaT_norm_13TeV' --g-values '0,0' --fix-poi --job-time 86400
```

For EtaT and ChiT:
```bash
./../scripts/submit_twin.py --point 'A_m400_w5p0,H_m400_w5p0' --mode prepost --tag ll --unblind --poi-set 'CMS_EtaT_norm_13TeV,CMS_ChiT_norm_13TeV' --g-values '0,0' --fix-poi --job-time 86400
./../scripts/submit_twin.py --point 'A_m400_w5p0,H_m400_w5p0' --mode psfromws --prepost-merge ll_all --tag ll --unblind --poi-set 'CMS_EtaT_norm_13TeV,CMS_ChiT_norm_13TeV' --g-values '0,0' --fix-poi --job-time 86400
```

## dNLL scans

dNLL scans were used in TOP-24-007 for the EtaT vs ChiT scan, and for HepData in HIG-22-013. They use `--algo fixed` in combine. It is implemented in `submit_twin`.

For an 1D scan of the EtaT signal strength using only one job:
```bash
./../scripts/submit_twin.py --point 'A_m400_w5p0,H_m400_w5p0' --mode nll --tag ll --unblind --nll-parameter CMS_EtaT_norm_13TeV --nll-interval='0,2' --nll-npoint='101' --nll-expect obs --g-values '0,0' --fix-poi --job-time 86400
```

The parameters mean:
- `--nll-parameter`: The parameters to scan.
- `--nll-interval`: The range for the parameters to scan.
- `--nll-npoint`: How many scan points per parameter. For some arcane reason this always needs to be one higher than actually wanted. So 101 actually means 100 points.
- `--nll-expect`: `obs` for data, `exp-b` for BG only Asimov. For others, need to cheat with `--extra-option`.

To scan a grid of multiple parameters, separate the values by commas resp. semicolons. E.g. to scan the EtaT and ChiT signal strengths in a 100x100 grid from -2 to 2 in both dimensions:

```bash
./../scripts/submit_twin.py --point 'A_m400_w5p0,H_m400_w5p0' --mode nll --tag ll --unblind --nll-parameter 'CMS_EtaT_norm_13TeV,CMS_ChiT_norm_13TeV' --nll-interval='-2,2;-2,2' --nll-npoint='101,101' --nll-expect obs --g-values '0,0' --fix-poi --job-time 86400
```

For large grids, one job will not be enough since it will run forever. The syntax to split the scan into multiple jobs is somewhat clunky. For example:

```bash
./../scripts/submit_twin.py --point 'A_m400_w5p0,H_m400_w5p0' --mode nll --tag ll --unblind --nll-parameter 'CMS_EtaT_norm_13TeV,CMS_ChiT_norm_13TeV' --nll-full-range='-2,2;-2,2' --nll-npoint='11,11' --nll-njob='11,11' --nll-expect obs --g-values '0,0' --fix-poi --job-time 86400
```

This means: a grid of 10x10 jobs (i.e. 100 jobs total) which each scan a grid of 10x10 points each - so a grid of 100x100 points total. Again the argument needs to be one higher than actually wanted. Annoying. Also note that `--nll-interval` is replaced by `--nll-full-range`.

## Feldman-Cousins scans

Described by Afiq in [`run_fc`](./run_fc). Only barely understood by myself. If you need this, I pray for your soul. 

# Plotting

The hopefully full commands for all plots in both papers are collected in [`top24007_plot_commands.sh`](./top24007_plot_commands.sh), [`hig22013_plot_commands.sh`](./hig22013_plot_commands.sh), and [`top24007_suppmat_commands.sh`](./top24007_suppmat_commands.sh), including links to all input files needed. Just hope that the files dont get deleted because the accounts run out lol. The last one contains lots of plots.

Almost all plots require a python3 environment with a reasonably up-to-date matplotlib and mplhep. The environment used for pepper usually works.
The exception is the impact plot, which sadly is in pyRoot and needs an EL7 container with CMSSW sourced. 

The script `plot_prepost_split.py` is a monster that can do almost everything regarding making pre/postfit plots including projections, normalization, unrolling etc. As a result, it is basically incomprehensible and unmaintainable. I am truly sorry.

Below I just collect some of the most important use cases:

### Pre/postfit plots:

The script is `plot_prepost_split.py`. Requires `mode prepost` to have been run, and `mode psfromws` as well for postfit plots.

Important options are:
- `--ifile`: The file produced by `mode prepost`, called something like `A_xxx__H_xxxxxx_fitdiagnostics_fit_s.root`.
- `--odir`: Output directory.
- `--plot-tag`: Tag to append to the file name.
- `--log`: Use log scale.
- `--cmslabel`: String to write next to the CMS logo.
- `--plot-formats`: Image output format. Can do e.g. `pdf,png,svg` to plot multiple formats at once.
- `--as-signal`: Add an additional signal, for example EtaT.
- `--skip-ah`: Do not plot A/H signals (e.g. when fitting only EtaT).
- `--no-total`: When plotting 2 or more signals, do not show their sum as well.
- `--ignore`: Ignore a given background process.
- `--xsec`: Use cross sections instead of signal strengths for the POI.
- `--best-fit-from`: Read the best fit POI and its uncertainties from a given file as made with `mode single` (1D) or `mode cross` (2D) instead of from FitDiagnostics (`mode prepost`). This is useful because these modes can give asymmetric uncertainties, while FitDiagnostics always symmetrizes.
- `--prefit-signal-from`: Needed for prefit plots because of combine stupidity; the path to the `ahtt_input.root` file in the workspace used to produce the fit. 
- `--batch`: Sum over years and channels instead of making one plot for each year and channel. This requires correctly projecting the covariance matrix for the error band, for which there are two options. Giving the path to the output file of `mode psfromws` reads it from there, which is recommended for postfit. Alternatively, giving `--batch project` projects directly from the covariance matrix as read in from the FitDiagnostics output. The latter is only tested for prefit.
- `--skip-each`: When using `--batch`, make only the year/channel summed plots and skip the others.
- `--normalize`: Plot normalized distributions, with correct propagation of the uncertainty. Only tested for prefit, might also work for postfit IDK.
- `--split-bins`: Split the 3D plot (e.g. $m_{t\bar t} \times c_{hel} \times c_{han}$) into 1D slices (for each chel/chan bin). Needs `--panel-labels` in addition to work properly.
- `--project-to`: Plot an 1D projection of the 3D plot (i.e. summing over other dimensions). Valid arguments are `mtt`, `mbbll`, `chel`, and `chan` as appropriate. Needs `--panel-labels` in addition to work properly.
- `--mass-cut`: When projecting with the above option, select only some bins in the mass variable (mtt or mbbll). Comma-seperated interval in GeV, with -1 standing in for infinity. E.g. use `--project-to chel --mass-cut='-1,360'` to plot $c_{hel}$ for $m_{t\bar t} < 360$ GeV.

See [`top24007_plot_commands.sh`](./top24007_plot_commands.sh) for concrete examples.


### Impact plots:

Get an EL7 container, source the CMSSW used for this repo (`cd XXX/CMSSW_10_2_13/src; cmsenv`), then run:

```bash
python2 ../scripts/customImpacts.py -i <path/to/data/impacts.json> -ia <path/to/asimov/impacts.json> -o <outfile_name> -t ../scripts/nuisance_map.json --per-page 20 --cms-label Supplementary
```

Both json files need to be merged using `merge_pull.py` before, as written above. You can give only one of `-i` or `-ia` to plot only data or only Asimov impacts.

### 1D / 2D limit plots:

See the examples in [`hig22013_plot_commands.sh`](./hig22013_plot_commands.sh).