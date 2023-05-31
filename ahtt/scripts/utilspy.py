#!/usr/bin/env python
# utilities containing functions used throughout - python/misc file

import os
import sys
import math
import fnmatch

from datetime import datetime
from desalinator import remove_spaces_quotes, tokenize_to_list, append_if_not_empty

max_nfile_per_dir = 4000

def syscall(cmd, verbose = True, nothrow = False):
    if verbose:
        print ("Executing: %s" % cmd)
        sys.stdout.flush()
    retval = os.system(cmd)
    if not nothrow and retval != 0:
        if not verbose:
            print ("Tried to execute: %s" % cmd)
        raise RuntimeError("Command failed with exit code {ret}!".format(ret = retval))

def get_point(sigpnt):
    pnt = sigpnt.split('_')
    return (pnt[0][0], float(pnt[1][1:]), float(pnt[2][1:].replace('p', '.')))

def stringify(gtuple):
    return str(gtuple)[1: -1]

def tuplize(gstring):
    return tuple([float(gg) for gg in tokenize_to_list(remove_spaces_quotes(gstring))])

def g_in_filename(gvalues):
    return "_".join(["g" + str(ii + 1) + "_" + str(gg) for ii, gg in enumerate(gvalues) if float(gg) >= 0.]).replace(".", "p")

def right_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

def index_n1(idxn, nbins):
    '''
    translate ND list of indices into an 'unrolled' 1D list of indices
    e.g. 2D -> 1D
    [
    0, 0 -> 0
    0, 1 -> 1
    1, 0 -> 2
    1, 1 -> 3
    ]
    '''
    idx1 = idxn[0]
    for ii in range(1, len(idxn)):
        multiplier = 1
        for jj in range(ii - 1, -1, -1):
            multiplier *= nbins[jj]

        idx1 += idxn[ii] * multiplier

    return idx1

def index_1d_1n(idx1, dim, nbins):
    '''
    inverse operation of index_n1, for a specific dimension
    '''
    multiplier = 1
    for dd in range(dim - 1, -1, -1):
        multiplier *= nbins[dd]

    return (idx1 // multiplier) % nbins[dim]

def index_1n(idx1, nbins):
    '''
    as above, but over all dimensions in a go
    '''
    idxn = [-1] * len(nbins)
    for iv in range(len(nbins)):
        idxn[iv] = index_1d_1n(idx1, iv, nbins)

    return idxn

def chunks(lst, npart):
    '''
    split a list of length nlst into npart chunks roughly of length nlst / npart
    '''
    neach = len(lst) // npart
    if neach < 1:
        print 'chunks called with a invalid npart. setting it to 1.'
        npart = 1
        neach = len(lst) // npart

    nreminder = len(lst) % npart
    nappend = 0
    result = []
    component = []

    for ii, ll in enumerate(lst):
        offset = 1 if nappend < nreminder else 0
        component.append(ll)
        if len(component) == neach + offset:
            result.append(component)
            component = []
            nappend += 1

    return result

def elementwise_add(list_of_lists):
    '''
    input: [[a, b, c], [1, 2, 3], [...]]
    output: [a + 1 + ..., b + 2 + ..., c + 3 + ...]
    '''
    if len(list_of_lists) < 1 or any(len(ll) < 1 or len(ll) != len(list_of_lists[0]) for ll in list_of_lists):
        raise RuntimeError("this method assumes that the argument is a list of lists of nonzero equal lengths!!!")

    result = list(list_of_lists[0])
    for ll in range(1, len(list_of_lists)):
        for rr in range(len(result)):
            result[rr] += list_of_lists[ll][rr]

    return result

def recursive_glob(base_directory, pattern):
    # https://stackoverflow.com/a/2186639
    results = []
    for base, dirs, files in os.walk(base_directory):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results

def index_list(index_string, baseline = 0):
    '''
    builds a list of indices from an index string, to be used in some options
    index_string can be a mixture of comma separated non-negative integers, or the form A..B where A < B and A, B non-negative
    where the comma separated integers are plainly the single indices and
    the A..B version builds a list of indices from [A, B). If A is omitted, it is assumed to be the baseline (0 by default)
    the returned list of indices is sorted, with duplicates removed
    returns empty list if syntax is not followed
    '''

    if not all([ii in "0123456789" for ii in index_string.replace("..", "").replace(",", "")]):
        return []

    index_string = tokenize_to_list(index_string)
    idxs = []

    for istr in index_string:
        if ".." in istr:
            ilst = tokenize_to_list(istr, '..' )
            idxs += range(int(ilst[0]), int(ilst[1])) if ilst[0] != "" else range(baseline, int(ilst[1]))
        else:
            idxs.append(int(istr))

    return list(set(idxs))

def directory_to_delete(location, flush = False):
    '''
    location can be a file or a directory
    the directory corresponding to location is tagged in the list
    and when flush is true, the empty tagged directories are deleted
    empty is taken to be containing no files as defined by recursive_glob
    '''
    if not hasattr(directory_to_delete, "tagged"):
        directory_to_delete.tagged = []

    if location is not None:
        if os.path.isfile(location):
            location = os.path.dirname(location)
            if os.path.isdir(location):
                directory_to_delete.tagged.append(os.path.abspath(location))
        elif os.path.isdir(location):
            directory_to_delete.tagged.append(os.path.abspath(location))

    if flush:
        for tag in set(directory_to_delete.tagged):
            if len(recursive_glob(tag, '*')) == 0:
                syscall("rm -r {tag}".format(tag = tag), False)
        directory_to_delete.tagged = []

def make_timestamp_dir(base, prefix = "tmp"):
    '''
    make a timestamp directory within some base, and return its absolute path
    '''
    timestamp_dir = os.path.join(base, append_if_not_empty(prefix, token = "_") + right_now())
    os.makedirs(timestamp_dir)
    dirname = os.path.abspath(timestamp_dir) + "/"
    directory_to_delete(location = dirname)
    return dirname
