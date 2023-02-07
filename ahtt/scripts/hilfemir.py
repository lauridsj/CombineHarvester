#!/usr/bin/env python
# similar to utilities.py, but dedicated solely to a dump of help messages of various scripts
# life has enough tedium to be writing the same thing more than twice

# for options that occur on scripts that run combine directly
combine_help_messages = {
    "--tag": "extra tag on datacard directory names",
    "--point": "comma-separated list of signal points to run on",
    "--signal": "signal filenames. comma separated",
    "--background": "data/background filenames. comma separated",
    "--channel": "final state channels considered in the analysis. relevant only in make_datacard. comma separated",
    "--year": "analysis year determining the correlation model to assume. relevant only in make_datacard. comma separated",

    "--sushi-kfactor": "apply nnlo kfactors computing using sushi on A/H signals",
    "--add-pseudodata": "add pseudodata into the template files for combine, using the sum of backgrounds, instead of using real data",
    "--inject-signal": "comma-separated list of signal points to inject into the pseudodata",
    "--replace-nominal": "comma-separated names of nuisances where, if a process has them, its nominal template of is replaced by that of the nuisances.\n"
    "Up or Down must be specified after a colon e.g. tmass_3GeV_TT:Up. Accepts a third value after another colon, which must be a number that defaults to 1.\n"
    "tmass_3GeV_TT:Up:0.5 means assigning half the difference between the tmass_3GeV_TTUp and the original nominal template as nominal, binwise linearly interpolated."
    "tmass_3GeV_TT:Up:0.37,QCDScale_MERen_TT:Down:0.73,... means adding the interpolated differences of all these templates wrt original nominal to the latter, "
    "and assigning the resulting template as nominal.",

    "--drop": "comma separated list of nuisances to be dropped in datacard mode. 'XX, YY' means all sources containing XX or YY are dropped. '*' to drop all",
    "--keep": "comma separated list of nuisances to be kept in datacard mode. same syntax as --drop. implies everything else is dropped",
    "--threshold": "threshold under which nuisances that are better fit by a flat line are dropped/assigned as lnN",
    "--lnN-under-threshold": "assign as lnN nuisances as considered by --threshold",
    "--use-shape-always": "use lowess-smoothened shapes even if the flat fit chi2 is better",
    "--no-mc-stats": "don't add/use nuisances due to limited mc stats (barlow-beeston lite)",
    "--seed": "random seed to be used for pseudodata generation. give 0 to read from machine, and negative values to use no rng",

    "--float-rate":
    "semicolon separated list of processes to make the rate floating for, using combine's rateParam directive.\n"
    "syntax: proc1:min1,max1;proc2:min2,max2; ... procN:minN,maxN. min and max can be omitted, they default to 0,2.\n"
    "the implementation assumes a single rate parameter across all channels.\n"
    "it also automatically replaces the now-redundant CMS_[process]_norm_13TeV nuisances.\n"
    "relevant only in the datacard step.",

    "--projection":
    "instruction to project multidimensional histograms, assumed to be unrolled such that dimension d0 is presented "
    "in slices of d1, which is in turn in slices of d2 and so on. the instruction is in the following syntax:\n"
    "[instruction 0]:[instruction 1]:...:[instruction n] for n different types of templates.\n"
    "each instruction has the following syntax: c0,c1,...,cn;b0,b1,...,bn;t0,t1,tm with m < n, where:\n"
    "ci are the channels the instruction is applicable to, bi are the number of bins along each dimension, ti is the target projection index.\n"
    "e.g. a channel ll with 3D templates of 20 x 3 x 3 bins, to be projected into the first dimension: ll;20,3,3;0 "
    "or a projection into 2D templates along 2nd and 3rd dimension: ll;20,3,3;1,2\n"
    "indices are zero-based, and spaces are ignored. relevant only in datacard/workspace mode.",

    "--mode": "comma-separated list of combine modes to run",
    "--compress": "compress output into a tar file",
    "--base-directory": "in non-datacard modes, this is the location where datacard is searched for, and output written to. ignored otherwise",
    "--unblind": "use data when fitting",
    "--fix-poi": "fix pois in the fit, to the values set in --g-value(s) and/or --r-value",
    "--mask": "comma-separated list of channel_year combinations to be masked in statistical analysis modes",

    "--freeze-mc-stats-zero": "only in the pull/impact/prepost/corrmat/nll mode, freeze mc stats nuisances to zero",
    "--freeze-mc-stats-post": "only in the pull/impact/prepost/corrmat/nll mode, freeze mc stats nuisances to the postfit values. "
    "--freeze-mc-stats-zero takes priority over this option",
    "--freeze-nuisance-post": "only in the prepost/corrmat/nll mode, freeze all nuisances to the postfit values. "
    "--freeze-mc-stats-zero takes priority over this option",
    "--extra-option": "extra options to be passed to combine when running pull/impact/prepost/corrmat modes. irrelevant elsewhere. "
    "spaces should be used only to separate options, not arguments to the same option, as they present parsing difficulties.",
    "--output-tag": "a tag that is appended to the fit output files. equals --tag by default",

    "--one-poi": "use physics model with only g as poi",
    "--g-value": "g to use when evaluating impacts/fit diagnostics/nll. give negative values to leave floating",
    "--r-value": "r to use when evaluating impacts/fit diagnostics/nll, if --one-poi is not used. give negative values to leave floating",
    "--raster-n": "number of chunks to split the g raster limit scan into",
    "--raster-i": "which chunk to process, in doing the raster scan",
    "--impact-nuisances": "format: grp;n1,n2,...,nN where grp is the name of the group of nuisances, and n1,n2,...,nN are the nuisances belonging to that group",

    "--g-values": "the two values of g to e.g. do the FC grid scan for, comma separated. give g1 and/or g2 < 0 means to keep them floating",
    "--n-toy": "number of toys to throw per point when generating or performing FC scans",
    "--run-idx": "index to append to a given toy generation/FC scan run",
    "--toy-location": "directory to dump the toys in mode generate/contour/gof (with --save-toy), and file to read them from in mode contour/gof (without --save-toy)",
    "--save-toy": "in mode contour/gof, will generate toys and save them in --toy-location, instead of reading from it.\n"
    "if --toy-location is not given, it defaults to a randomly generated directory within --base-directory.",

    "--fc-expect": "expected scenarios to assume in the FC scan. comma separated.\n"
    "exp-b -> g1 = g2 = 0; exp-s -> g1 = g2 = 1; exp-01 -> g1 = 0, g2 = 1; exp-10 -> g1 = 1, g2 = 0",
    "--fc-nuisance-mode": "how to handle nuisance parameters in toy generation (see https://arxiv.org/abs/2207.14353)\n"
    "WARNING: profile mode implementation is incomplete!!",

    "--fc-skip-data": "skip running on data/asimov",
    "--delete-root": "delete root files after compiling",
    "--ignore-previous": "ignore previous grid when compiling",
}

# for options with the same name as above in submit scripts, that need a different help message
# or for options that occur only in submit scripts
submit_help_messages = {
    "--raster-i": "which chunks to process, in doing the raster scan.\n"
    "can be one or more comma separated non-negative integers, or something of the form A...B where A < B and A, B non-negative\n"
    "where the comma separated version is plainly the list of indices to be given to --raster-i, if --raster-n > 0\n"
    "and the A...B version builds a list of indices from [A, B). If A is omitted, it is assumed to be 0\n"
    "mixing of both syntaxes are not allowed.",

    "--impact-n": "maximum number of nuisances to run in a single impact job",
    "--skip-expth": "in pull/impact mode, skip running over the experimental and theory nuisances",
    "--run-mc-stats": "in pull/impact mode, run also over the BB nuisances individually. this option does not affect their treatment in any way (analytical minimization)",

    "--job-time": "time to assign to each job",
    "--local": "run jobs locally, do not submit to HTC",
    "--force": "force local jobs to run, even if a job log already exists",

    "--point": "desired pairs of signal points to run on, comma (between points) and semicolon (between pairs) separated\n"
    "another syntax is: m1,m2,...,mN;w1,w2,...,wN;m1,m2,...,mN;w1,w2,...,wN, where:\n"
    "the first mass and width strings refer to the A grid, and the second to the H grid.\n"
    "both mass and width strings must include their m and w prefix, and for width, their p0 suffix.\n"
    "e.g. m400,m450;w5p0;m600,m750;w10p0 expands to A_m400_w5p0,H_m600_w10p0;A_m450_w5p0,H_m600_w10p0;A_m400_w5p0,H_m750_w10p0;A_m450_w5p0,H_m750_w10p0",

    "--run-idxs": "can be one or more comma separated non-negative integers, or something of the form A...B where A < B and A, B non-negative\n"
    "where the comma separated version is plainly the list of indices to be given to --run-idx, if --n-toy > 0\n"
    "and the A...B version builds a list of indices from [A, B). If A is omitted, it is assumed to be 0\n"
    "mixing of both syntaxes is not allowed.",

    "--fc-mode": "what to do with the grid read from --fc-g-grid, can be 'add' for submitting more toys of the same points,\n"
    "or 'refine', for refining the contour that can be drawn using the grid",

    "--fc-single-point": "run FC scan only on a single point given by --g-values",
    "--fc-g-grid": "comma (between files) and semicolon (between pairs) separated json files generated by compile mode to read points from",
    "--fc-initial-distance": "initial distance between g grid points for FC scans",
    "--proper-sigma": "use proper 1 or 2 sigma CLs instead of 68% and 95% in FC scan alphas",
}
