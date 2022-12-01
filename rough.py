#!/usr/bin/env python3
import sys

import zipfile
with zipfile.ZipFile("tiny-imagenet-200.zip", 'r') as zip_ref:
    zip_ref.extractall("imagenet")  