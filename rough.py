#!/usr/bin/env python3
import sys

import zipfile
with zipfile.ZipFile("imagewang.zip", 'r') as zip_ref:
    zip_ref.extractall("imagenet")  