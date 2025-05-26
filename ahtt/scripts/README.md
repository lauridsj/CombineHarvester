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

## A/H limits

## Impacts

## dNLL scans

## Feldman-Cousins scans