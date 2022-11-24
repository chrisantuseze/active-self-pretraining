#!/usr/bin/env python3
import sys
# from zipfile import PyZipFile
# for zip_file in sys.argv[1:]:
#     pzf = PyZipFile(zip_file)
#     pzf.extractall()

import zipfile
with zipfile.ZipFile("tiny-imagenet-200.zip", 'r') as zip_ref:
    zip_ref.extractall("imagenet")