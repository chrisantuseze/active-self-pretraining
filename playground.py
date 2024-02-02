#!/usr/bin/env python3
import zipfile

with zipfile.ZipFile("domainnet/sketch.zip", 'r') as zip_ref:
    zip_ref.extractall("sketch")

with zipfile.ZipFile("domainnet/painting.zip", 'r') as zip_ref:
    zip_ref.extractall("painting")

with zipfile.ZipFile("domainnet/clipart.zip", 'r') as zip_ref:
    zip_ref.extractall("clipart")

with zipfile.ZipFile("domainnet/real.zip", 'r') as zip_ref:
    zip_ref.extractall("real")

# with zipfile.ZipFile("office-31.zip", 'r') as zip_ref:
#     zip_ref.extractall("office-31")

# with zipfile.ZipFile("officehome.zip", 'r') as zip_ref:
#     zip_ref.extractall("officehome")
