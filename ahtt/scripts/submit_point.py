#!/usr/bin/env python
# submits single_point_ahtt jobs
# for imode in 'datacard,validate'; do ./../scripts/submit_point.py --sushi-kfactor --lnN-under-threshold --year "${years}" --channel "${channels}" --tag "${tag}" --keep "${keeps}" --drop "${drops}" --mode "${imode}"; done
# get the shell vars from run_fc

from argparse import ArgumentParser
import os
import sys
import glob
import copy
from collections import OrderedDict

from utilspy import syscall, chunks, index_list
from utilslab import input_base, input_sig, remove_mjf, parities, masses, widths
from utilsroot import get_nbin
from utilscombine import problematic_datacard_log
from utilshtc import submit_job, flush_jobs, common_job, make_singularity_command

from desalinator import prepend_if_not_empty, tokenize_to_list, remove_quotes, remove_spaces_quotes
from argumentative import common_point, common_common, common_fit_pure, common_fit_forwarded, make_datacard_pure, make_datacard_forwarded, common_1D, common_submit, parse_args
from hilfemir import combine_help_messages, submit_help_messages

if __name__ == '__main__':
    print ("submit_point :: called with the following arguments")
    print (sys.argv[1:])
    print ("\n")
    print (" ".join(sys.argv))
    print ("\n")
    sys.stdout.flush()

    parser = ArgumentParser()
    common_point(parser, False)
    common_common(parser)
    common_fit_pure(parser)
    common_fit_forwarded(parser)
    make_datacard_pure(parser)
    make_datacard_forwarded(parser)
    common_1D(parser)
    common_submit(parser)

    parser.add_argument("--raster-i", help = submit_help_messages["--raster-i"], dest = "ichunk", default = "..[--raster-n value]", required = False)
    parser.add_argument("--impact-n", help = submit_help_messages["--impact-n"], dest = "nnuisance", default = 10, required = False, type = lambda s: int(remove_spaces_quotes(s)))
    parser.add_argument("--impact-i", help = submit_help_messages["--impact-i"], dest = "inuisance", default = -1, required = False, type = lambda s: int(remove_spaces_quotes(s)))
    parser.add_argument("--skip-expth", help = submit_help_messages["--skip-expth"], dest = "runexpth", action = "store_false", required = False)
    parser.add_argument("--run-mc-stats", help = submit_help_messages["--run-mc-stats"], dest = "runbb", action = "store_true", required = False)

    args = parse_args(parser)

    remove_mjf()
    scriptdir = os.path.dirname(os.path.abspath(__file__))

    points = []
    keep_point = args.point

    for parity in parities:
        for mass in masses:
            for width in widths:
                if mass == "m343" and width != "w2p0":
                    continue

                pnt = "_".join([parity, mass, width])

                if keep_point == [] or any([kk in pnt for kk in keep_point]):
                    points.append(pnt)

    rundc = "datacard" in args.mode or "workspace" in args.mode
    runlimit = "limit" in args.mode
    runpull = "pull" in args.mode or "impact" in args.mode

    for pnt in points:
        flush_jobs()
        dorundc = rundc
        if not dorundc and not os.path.isdir(pnt + args.tag) and os.path.isfile(pnt + args.tag + ".tar.gz"):
            syscall("tar xf {ttt} && rm {ttt}".format(ttt = pnt + args.tag + ".tar.gz"))

        mode = ""
        hasworkspace = os.path.isfile(pnt + args.tag + "/workspace_g-scan.root") and os.path.isfile(pnt + args.tag + "/workspace_one-poi.root")
        if not dorundc and not hasworkspace:
            syscall("rm -r {ddd}".format(ddd = pnt + args.tag), True, True)
            dorundc = True
            mode = "datacard," + args.mode

        if os.path.isdir(pnt + args.tag):
            logs = glob.glob("single_point_" + pnt + args.tag + "_*.o*")
            for ll in logs:
                if 'validate' in ll and problematic_datacard_log(ll):
                    print("WARNING :: datacard of point {pnt} is tagged as problematic by problematic_datacard_log!!!\n\n\n".format(pnt = pnt + args.tag))
                syscall("mv {lll} {ddd}".format(lll = ll, ddd = pnt + args.tag))

        if dorundc and os.path.isdir(pnt + args.tag):
            mode = args.mode.replace("datacard,", "").replace("datacard", "").replace("workspace,", "").replace("workspace", "")

            if mode != "":
                dorundc = False
            else:
                continue

        if mode == "":
            mode = args.mode

        job_name = "single_point_" + pnt + args.otag + "_" + "_".join(tokenize_to_list( remove_spaces_quotes(mode) ))
        job_name += "{mod}{poi}{gvl}{rvl}{fix}".format(
            mod = "" if dorundc else "_one-poi" if args.onepoi else "_g-scan",
            poi = "_" + args.poiset.replace(",", "__") if args.poiset != "" else "",
            gvl = "_g_" + str(args.setg).replace('.', 'p') if args.setg >= 0. else "",
            rvl = "_r_" + str(args.setr).replace('.', 'p') if args.setr >= 0. and not args.onepoi else "",
            fix = "_fixed" if args.fixpoi and (args.setg >= 0. or args.setr >= 0.) else ""
        )

        job_arg = "--point {pnt} --mode {mmm} {sig} {one} {gvl} {rvl} {fix}".format(
            pnt = pnt,
            mmm = mode,
            sig = "--signal " + input_sig(args.signal, pnt, args.inject, args.channel, args.year) if dorundc else "",
            one = "--one-poi" if args.onepoi else "",
            gvl = "--g-value " + str(args.setg) if args.setg >= 0. else "",
            rvl = "--r-value " + str(args.setr) if args.setr >= 0. else "",
            fix = "--fix-poi" if args.fixpoi and (args.setg >= 0. or args.setr >= 0.) else ""
        )
        args.rundc = dorundc
        job_arg += common_job(args)

        if runlimit and not args.onepoi:
            if args.nchunk < 0:
                args.nchunk = 6

            if args.ichunk == "..[--raster-n value]":
                args.ichunk = ".." + str(args.nchunk)

            if args.nchunk > 1:
                idxs = index_list(args.ichunk)
            else:
                idxs = [0]

            for idx in idxs:
                jname = job_name.replace("g-scan", "g-scan_{nch}_{idx}".format(nch = "n" + str(args.nchunk), idx = "i" + str(idx)))
                logs = glob.glob(pnt + args.tag + "/" + jname + ".o*")

                if not (args.runlocal and args.forcelocal):
                    if len(logs) > 0:
                        continue

                jarg = job_arg
                jarg += " {nch} {idx}".format(
                    nch = "--raster-n " + str(args.nchunk),
                    idx = "--raster-i " + str(idx)
                )

                submit_job(jname, jarg, args.jobtime, 1, args.memory,
                           "." if dorundc else pnt + args.tag, scriptdir + "/single_point_ahtt.py", True, args.runlocal, args.writelog)
        elif runpull:
            if args.nnuisance < 0:
                args.nnuisance = 25

            nuisances = OrderedDict()
            if args.runexpth:
                pythoncmd = "python {cms}/src/HiggsAnalysis/CombinedLimit/test/systematicsAnalyzer.py --format brief --all {dcd}/ahtt_{ch}.txt > {dcd}/{sot}".format(
                    cms = r'${CMSSW_BASE}',
                    dcd = pnt + args.tag,
                    ch = "combined" if "," in args.channel or "," in args.year else args.channel + "_" + args.year,
                    sot = "ahtt_systanalyzer_output.txt"
                )
                syscall(make_singularity_command(pythoncmd))
                syscall("cat {dcd}/{sot} | "
                        "grep -v -e 'NUISANCE (TYPE)' | grep -v -e '--------------------------------------------------' | awk {awk} "
                        "> {dcd}/{nui} && grep {prm} {dcd}/ahtt_{ch}.txt | awk {awk} | sort -u >> {dcd}/{nui} && "
                        "grep rateParam {dcd}/ahtt_{ch}.txt | grep -v '@' | awk {awk} | sort -u >> {dcd}/{nui}".format(
                            dcd = pnt + args.tag,
                            ch = "combined" if "," in args.channel or "," in args.year else args.channel + "_" + args.year,
                            nui = "ahtt_nuisance.txt",
                            sot = "ahtt_systanalyzer_output.txt",
                            prm = r"'param '",
                            awk = r"'{print $1}'"
                        ))

                
                with open(pnt + args.tag + "/ahtt_nuisance.txt") as fexp:
                    nparts = fexp.readlines()
                    nparts = [et.rstrip() for et in nparts]
                    nparts = sorted(list(set(nparts)))
                    nsplit = (len(nparts) // args.nnuisance) + 1
                    nparts = chunks(nparts, nsplit)

                    for ip, ipart in enumerate(nparts):
                        if args.inuisance >= 0 and ip != args.inuisance:
                            continue
                        group = "expth_{ii}".format(ii = str(ip))
                        nuisances[group] = copy.deepcopy(ipart)
                syscall('rm {nui}'.format(nui = pnt + args.tag + "/ahtt_systanalyzer_output.txt"), False)
                syscall('rm {nui}'.format(nui = pnt + args.tag + "/ahtt_nuisance.txt"), False)

            if args.runbb:
                for cc in remove_spaces(args.channel).split(','):
                    for yy in remove_spaces(args.year).split(','):
                        if cc + "_" + yy in args.mask:
                            continue

                        nbin = get_nbin(pnt + args.tag + "/ahtt_input.root", cc, yy)
                        nsplit = (nbin // args.nnuisance) + 1 
                        nparts = chunks(range(nbin), nsplit)

                        for ip, ipart in enumerate(nparts):
                            if args.inuisance >= 0 and ip != args.inuisance:
                                continue
                            group = "mcstat_{cc}_{yy}_{ii}".format(cc = cc, yy = yy, ii = str(ip))
                            mcstats = ["prop_bin" + "{cc}_{yy}_bin".format(cc = cc, yy = yy) + str(ii) for ii in ipart]
                            nuisances[group] = copy.deepcopy(mcstats)

            for group, nuisance in nuisances.items():
                jname = job_name + "_" + group
                logs = glob.glob(pnt + args.tag + "/" + jname + ".o*")

                if not (args.runlocal and args.forcelocal):
                    if len(logs) > 0:
                        continue

                jarg = job_arg
                jarg += " --impact-nuisances '{grp};{nui}'".format(grp = group, nui = ",".join(nuisance))

                submit_job(jname, jarg, args.jobtime, 1, args.memory,
                           "." if dorundc else pnt + args.tag, scriptdir + "/single_point_ahtt.py", True, args.runlocal, args.writelog)
        else:
            logs = glob.glob(pnt + args.tag + "/" + job_name + ".o*")

            if not (args.runlocal and args.forcelocal):
                if len(logs) > 0:
                    continue

            submit_job(job_name, job_arg, args.jobtime, 1, args.memory,
                       "." if dorundc else pnt + args.tag, scriptdir + "/single_point_ahtt.py", True, args.runlocal, args.writelog)
    flush_jobs()
