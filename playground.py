#!/usr/bin/env python3
import zipfile

with zipfile.ZipFile("officehome.zip", 'r') as zip_ref:
    zip_ref.extractall("officehome")
