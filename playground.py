#!/usr/bin/env python3
import zipfile

with zipfile.ZipFile("office-31.zip", 'r') as zip_ref:
    zip_ref.extractall("office-31")
