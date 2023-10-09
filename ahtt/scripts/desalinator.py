#!/usr/bin/env python
# similar to utils*.py, but dedicated to functions that manipulate arguments given to argparse
# the name is just a quip on how different lab 'salts' the arguments differently

def remove_quotes(string):
    return string.replace('"',  '').replace("'",  "")

def remove_consecutive_quotes(string):
    result = string
    quotes = ['"', "'"]
    for q0 in quotes:
        for q1 in quotes:
            result = result.replace(q0 + q1, '')
    return result

def remove_spaces(string):
    return string.replace(" ", "")

def remove_spaces_quotes(string):
    return remove_quotes( remove_spaces(string) )

def tokenize_to_list(string, token = ',', astype = None):
    result = [] if string == "" else string.split(token)
    return result if astype is None else [astype(res) for res in result]

def prepend_if_not_empty(string, token = "_"):
    return token + string if string != "" and not string.startswith(token) else string

def append_if_not_empty(string, token = "/"):
    return string + token if string != "" and not string.endswith(token) else string

def clamp_with_quote(string, prefix = "", suffix = ""):
    if string == "":
        return string
    clean = remove_consecutive_quotes(string)
    clean = clean.replace('"', "'")
    q = "'"
    opencloseq = clean.startswith(q) and clean.endswith(q)
    if not opencloseq:
        return prefix + q + clean + q + suffix
    else:
        return prefix + clean + suffix
