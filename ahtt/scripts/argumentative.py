#!/usr/bin/env python
# similar to utils*.py, but dedicated to supplying stock argparse objects
# as scripts tend to have very similar implementation of these things
# we'll have different arg types - pure, forwarded and final
# pure args are just that - that needs no further conversion/cleanup
# forwarded args are those a script takes only to forward it to another script
# they should be kept very simple - only strings without any clean up, conversion, etc
# typically they correspond to arguments that are treated as final in some scripts, and only forwarded in others
# final args are those that arent forwarded further
# these can be cleaned up and/or transformed in ways as complex as needed for the usage
# note: at this stage this is a mere convention, and is not enforced at code level
# the main goal is to minimize duplication, rather than getting the categorization watertight

import re
import numpy as np
import itertools

from utilspy import rng_seeder
from utilscombine import update_mask
from desalinator import prepend_if_not_empty, append_if_not_empty, tokenize_to_list, remove_spaces_quotes
from hilfemir import combine_help_messages, submit_help_messages

#def script-name_arg-type(parser):
#    # add whatever args...
#    return parser

def common_common(parser):
    parser.add_argument("--no-mc-stats", help = combine_help_messages["--no-mc-stats"], dest = "mcstat", action = "store_false", required = False)
    parser.add_argument("--tag", help = combine_help_messages["--tag"], default = "", required = False, type = prepend_if_not_empty)
    parser.add_argument("--experimental", help = combine_help_messages["--experimental"], dest = "experimental", action = "store_true", required = False)
    parser.add_argument("--load-snapshot", help = combine_help_messages["--load-snapshot"], dest = "snapshot", default = "", required = False)
    parser.add_argument("--prepost-ws", help = combine_help_messages["--prepost-ws"], dest = "prepostws", action = "store_true", required = False)
    parser.add_argument("--seed", help = combine_help_messages["--seed"], default = -1, required = False, type = lambda s: int(remove_spaces_quotes(s)))
    return parser

def common_point(parser, required = True):
    parser.add_argument("--point", help = combine_help_messages["--point"], default = "", required = required, type = lambda s: sorted(tokenize_to_list( remove_spaces_quotes(s) )))
    return parser

def common_fit_pure(parser):
    parser.add_argument("--unblind", help = combine_help_messages["--unblind"], dest = "asimov", action = "store_false", required = False)
    parser.add_argument("--fix-poi", help = combine_help_messages["--fix-poi"], dest = "fixpoi", action = "store_true", required = False)
    parser.add_argument("--compress", help = combine_help_messages["--compress"], dest = "compress", action = "store_true", required = False)

    parser.add_argument("--fit-strategy", help = combine_help_messages["--fit-strategy"], dest = "fitstrat", default = -1, required = False,
                        choices = [-1, 0, 1, 2], type = lambda s: int(remove_spaces_quotes(s)))
    parser.add_argument("--use-hesse", help = combine_help_messages["--use-hesse"], dest = "usehesse", action = "store_true", required = False)
    parser.add_argument("--redo-best-fit", help = combine_help_messages["--redo-best-fit"], dest = "keepbest", action = "store_false", required = False)
    parser.add_argument("--default-workspace", help = combine_help_messages["--default-workspace"], dest = "defaultwsp", action = "store_true", required = False)

    parser.add_argument("--extra-option", help = combine_help_messages["--extra-option"], dest = "extopt", default = "", required = False)
    parser.add_argument("--output-tag", help = combine_help_messages["--output-tag"], dest = "otag", default = "", required = False, type = prepend_if_not_empty)

    parser.add_argument("--base-directory", help = combine_help_messages["--base-directory"], dest = "basedir", default = "", required = False, type = append_if_not_empty)
    return parser

def common_fit_forwarded(parser):
    parser.add_argument("--mode", help = combine_help_messages["--mode"], default = "datacard", required = False)
    parser.add_argument("--poi-set", help = combine_help_messages["--poi-set"], dest = "poiset", default = "", required = False)
    parser.add_argument("--mask", help = combine_help_messages["--mask"], dest = "mask", default = "", required = False)
    parser.add_argument("--freeze-zero", help = combine_help_messages["--freeze-zero"], dest = "frzzero", default = "", required = False)
    parser.add_argument("--freeze-nonzero", help = combine_help_messages["--freeze-nonzero"], dest = "frznzro", default = "", required = False)
    parser.add_argument("--freeze-post", help = combine_help_messages["--freeze-post"], dest = "frzpost", default = "", required = False)
    return parser

def common_fit(parser):
    parser.add_argument("--mode", help = combine_help_messages["--mode"], default = "datacard", required = False, type = lambda s: tokenize_to_list( remove_spaces_quotes(s) ))
    parser.add_argument("--poi-set", help = combine_help_messages["--poi-set"], dest = "poiset", default = "", required = False,
                        type = lambda s: [] if s == "" else tokenize_to_list(remove_spaces_quotes(s)))
    parser.add_argument("--mask", help = combine_help_messages["--mask"], dest = "mask", default = "", required = False,
                        type = lambda s: [] if s == "" else update_mask( tokenize_to_list( remove_spaces_quotes(s) ) ))
    parser.add_argument("--freeze-zero", help = combine_help_messages["--freeze-zero"], dest = "frzzero", default = "", required = False,
                        type = lambda s: set() if s == "" else set(tokenize_to_list( remove_spaces_quotes(s) )))
    parser.add_argument("--freeze-nonzero", help = combine_help_messages["--freeze-nonzero"], dest = "frznzro", default = "", required = False,
                        type = lambda s: set() if s == "" else set(tokenize_to_list( remove_spaces_quotes(s) )))
    parser.add_argument("--freeze-post", help = combine_help_messages["--freeze-post"], dest = "frzpost", default = "", required = False,
                        type = lambda s: set() if s == "" else set(tokenize_to_list( remove_spaces_quotes(s) )))
    parser.add_argument("--result-directory", help = combine_help_messages["--result-directory"], dest = "resdir", default = "", required = False,
                        type = append_if_not_empty)
    return parser

def common_1D(parser):
    parser.add_argument("--one-poi", help = combine_help_messages["--one-poi"], dest = "onepoi", action = "store_true", required = False)
    parser.add_argument("--g-value", help = combine_help_messages["--g-value"], dest = "setg", default = -1., required = False, type = lambda s: float(remove_spaces_quotes(s)))
    parser.add_argument("--r-value", help = combine_help_messages["--r-value"], dest = "setr", default = -1., required = False, type = lambda s: float(remove_spaces_quotes(s)))
    parser.add_argument("--raster-n", help = combine_help_messages["--raster-n"], dest = "nchunk", default = 6, required = False, type = lambda s: int(remove_spaces_quotes(s)))
    return parser

def common_2D(parser):
    parser.add_argument("--g-values", help = combine_help_messages["--g-values"], default = "-1., -1.", dest = "gvalues", required = False,
                        type = lambda s: [str(float(ss)) for ss in tokenize_to_list( remove_spaces_quotes(s) )])

    parser.add_argument("--n-toy", help = combine_help_messages["--n-toy"], default = 50, dest = "ntoy", required = False, type = lambda s: int(remove_spaces_quotes(s)))
    parser.add_argument("--toy-location", help = combine_help_messages["--toy-location"], dest = "toyloc", default = "", required = False,
                        type = lambda s: s if s == "" or s.endswith("/") or s.endswith(".root") else s + "/")
    parser.add_argument("--save-toy", help = combine_help_messages["--save-toy"], dest = "savetoy", action = "store_true", required = False)
    parser.add_argument("--gof-skip-data", help = combine_help_messages["--gof-skip-data"], dest = "gofrundat", action = "store_false", required = False)

    parser.add_argument("--fc-expect", "--nll-expect", help = combine_help_messages["--fc-expect"], default = "exp-b", dest = "fcexp", required = False,
                        type = lambda s: tokenize_to_list( remove_spaces_quotes(s), ';' if ';' in s or re.search(r',[^eo]', remove_spaces_quotes(s)) else ',' ))
    parser.add_argument("--fc-skip-data", help = combine_help_messages["--fc-skip-data"], dest = "fcrundat", action = "store_false", required = False)

    parser.add_argument("--cc-skip-data", help = combine_help_messages["--cc-skip-data"], dest = "ccrundat", action = "store_false", required = False)
    parser.add_argument("--cc-mask", help = combine_help_messages["--cc-mask"], dest = "ccmasks",
                        default = [], required = False,
                        type = lambda s: [] if s == "" else sorted(tokenize_to_list( remove_spaces_quotes(s) )))

    parser.add_argument("--delete-root", help = combine_help_messages["--delete-root"], dest = "rmroot", action = "store_true", required = False)
    parser.add_argument("--collect-toy", help = combine_help_messages["--collect-toy"], dest = "collecttoy", action = "store_true", required = False)
    parser.add_argument("--ignore-previous", help = combine_help_messages["--ignore-previous"], dest = "ignoreprev", action = "store_true", required = False)

    parser.add_argument("--prepost-fit", help = combine_help_messages["--prepost-fit"], dest = "prepostfit", default = "s", required = False, choices = ['s', 'b'], type = remove_spaces_quotes)
    parser.add_argument("--prepost-merge", help = combine_help_messages["--prepost-merge"], dest = "prepostmerge",
                        default = [], required = False,
                        type = lambda s: [] if s == "" else sorted(tokenize_to_list( remove_spaces_quotes(s) )))
    parser.add_argument("--prepost-result", help = combine_help_messages["--prepost-result"], dest = "prepostres",
                        default = "", required = False, type = remove_spaces_quotes)

    parser.add_argument("--nll-parameter", help = combine_help_messages["--nll-parameter"], dest = "nllparam", default = "", required = False,
                        type = lambda s: [] if s == "" else tokenize_to_list(remove_spaces_quotes(s)))
    parser.add_argument("--nll-npoint", help = combine_help_messages["--nll-npoint"], dest = "nllnpnt", default = "", required = False,
                        type = lambda s: [] if s == "" else [int(npnt) for npnt in tokenize_to_list(remove_spaces_quotes(s))])
    parser.add_argument("--nll-interval", help = combine_help_messages["--nll-interval"], dest = "nllwindow", default = "", required = False,
                        type = lambda s: [] if s == "" else tokenize_to_list(remove_spaces_quotes(s), ";"))
    parser.add_argument("--nll-unconstrained", help = combine_help_messages["--nll-unconstrained"], dest = "nllunconstrained", default = "", required = False,
                        type = lambda s: [] if s == "" else tokenize_to_list(remove_spaces_quotes(s)))
    return parser

def common_submit(parser):
    parser.add_argument("--job-time", help = submit_help_messages["--job-time"], default = "", dest = "jobtime", required = False,
                        type = lambda s: s if s != "" and int(s) > 10800 else "10800")
    parser.add_argument("--memory", help = submit_help_messages["--memory"], default="", required = False)
    parser.add_argument("--local", help = submit_help_messages["--local"], dest = "runlocal", action = "store_true", required = False)
    parser.add_argument("--force", help = submit_help_messages["--force"], dest = "forcelocal", action = "store_true", required = False)
    parser.add_argument("--no-log", help = submit_help_messages["--no-log"], dest = "writelog", action = "store_false", required = False)

def make_datacard_pure(parser):
    parser.add_argument("--sushi-kfactor", help = combine_help_messages["--sushi-kfactor"], dest = "kfactor", action = "store_true", required = False)
    parser.add_argument("--lnN-under-threshold", help = combine_help_messages["--lnN-under-threshold"], dest = "lnNsmall", action = "store_true", required = False)
    parser.add_argument("--use-shape-always", help = combine_help_messages["--use-shape-always"], dest = "alwaysshape", action = "store_true", required = False)
    return parser

def make_datacard_forwarded(parser):
    parser.add_argument("--signal", help = combine_help_messages["--signal"], default = "", required = False)
    parser.add_argument("--background", help = combine_help_messages["--background"], default = "", required = False)
    parser.add_argument("--channel", help = combine_help_messages["--channel"], default = "ee,em,mm,e3j,e4pj,m3j,m4pj", required = False)
    parser.add_argument("--year", help = combine_help_messages["--year"], default = "2016pre,2016post,2017,2018", required = False)
    parser.add_argument("--drop", help = combine_help_messages["--drop"], default = "", required = False)
    parser.add_argument("--keep", help = combine_help_messages["--keep"], default = "", required = False)
    parser.add_argument("--threshold", help = combine_help_messages["--threshold"], default = "", required = False)
    parser.add_argument("--float-rate", help = combine_help_messages["--float-rate"], dest = "rateparam", default = "", required = False)
    parser.add_argument("--inject-signal", help = combine_help_messages["--inject-signal"], dest = "inject", default = "", required = False)
    parser.add_argument("--as-signal", help = combine_help_messages["--as-signal"], dest = "assignal", default = "", required = False)
    parser.add_argument("--exclude-process", help = combine_help_messages["--exclude-process"], dest = "excludeproc", default = "ChiT", required = False)
    parser.add_argument("--ignore-bin", help = combine_help_messages["--ignore-bin"], dest = "ignorebin", default = "", required = False)
    parser.add_argument("--projection", help = combine_help_messages["--projection"], default = "", required = False)
    parser.add_argument("--chop-up", help = combine_help_messages["--chop-up"], dest = "chop", default = "", required = False)
    parser.add_argument("--replace-nominal", help = combine_help_messages["--replace-nominal"], dest = "repnom", default = "", required = False)
    parser.add_argument("--arbitrary-resonance-normalization", help = combine_help_messages["--arbitrary-resonance-normalization"],
                        dest = "arbnorm", default = "", required = False)
    parser.add_argument("--esu-scale-json", default="", dest="esu_scale_json")
    parser.add_argument("--esu-scale-to-lumi", default="", dest="esu_scale_to_lumi")
    parser.add_argument("--esu-scale-systs", action="store_true", dest="esu_scale_systs")
    return parser

def parse_args(parser):
    '''
    some stuff with the args is always done
    this introduces undesirable redundancy in the code downstream
    do those things here instead
    '''
    args = parser.parse_args()
    if args.otag == "":
        args.otag = args.tag
    if hasattr(args, 'nllfullrange') and hasattr(args, 'nllparam'):
        if args.nllfullrange != []:
            if not hasattr(args, 'nllnjob'):
                args.nllnjob = [10] * len(args.nllparam)
            result = []
            for minmax, njob, in zip(args.nllfullrange, args.nllnjob):
                mm = minmax.split(',')
                each = list(np.linspace(float(mm[0]), float(mm[1]), num = njob))
                each = ["{0},{1}".format(round(each[ii], 5), round(each[ii + 1], 5)) for ii in range(len(each) - 1)] if len(each) > 1 else ["{0},{0}".format(round(each[0], 5))] if len(each) == 1 else []
                result.append(each)
            args.nllfullrange = itertools.product(*result)
        else:
            args.nllfullrange = [args.nllwindow]
    rng_seeder(args.seed)
    return args
