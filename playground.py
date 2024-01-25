#!/usr/bin/env python3
import zipfile

with zipfile.ZipFile("sketch.zip", 'r') as zip_ref:
    zip_ref.extractall("sketch")

with zipfile.ZipFile("painting.zip", 'r') as zip_ref:
    zip_ref.extractall("painting")

with zipfile.ZipFile("clipart.zip", 'r') as zip_ref:
    zip_ref.extractall("clipart")

with zipfile.ZipFile("real.zip", 'r') as zip_ref:
    zip_ref.extractall("real")

# with zipfile.ZipFile("office-31.zip", 'r') as zip_ref:
#     zip_ref.extractall("office-31")

# with zipfile.ZipFile("officehome.zip", 'r') as zip_ref:
#     zip_ref.extractall("officehome")
