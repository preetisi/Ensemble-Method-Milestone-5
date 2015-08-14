#!/usr/bin/env python

import os
import shutil
import sys


def replace_end(some_str, orig, new):
    if some_str.endswith(orig):
        some_str = some_str[:-len(orig)] + new
    return some_str


new_end = sys.argv[1]
if len(sys.argv) > 2:
    orig_end = sys.argv[2]
    print("Replacing {0} in every file with {1}".format(orig_end, new_end))
else:
    orig_end = ""
    print("Appending {0} to every file".format(new_end))

for filename in os.listdir("."):
    dataset, filetype = filename.split(".")
    new_dataset = replace_end(dataset, orig_end, new_end)
    new_name = "{0}.{1}".format(new_dataset, filetype)
    print("moving {0} to {1}".format(filename, new_name))
    shutil.move(filename, new_name)
