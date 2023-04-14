#!/usr/bin/env python
# similar to utilities.py, but dedicated to functions that manipulate arguments given to argparse
# the name is just a quip on how different lab 'salts' the arguments differently

def remove_quotes(string):
    return string.replace('"',  '').replace("'",  "")

def remove_spaces(string):
    return string.replace(" ", "")

def remove_spaces_quotes(string):
    return remove_quotes( remove_spaces(string) )

def tokenize_to_list(string, token = ','):
    return [] if string == "" else string.split(token)

def prepend_if_not_empty(string, token = "_"):
    return token + string if string != "" and not string.startswith(token) else string

def append_if_not_empty(string, token = "/"):
    return string + token if string != "" and not string.endswith(token) else string
