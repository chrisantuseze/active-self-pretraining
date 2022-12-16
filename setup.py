import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command


# Package meta-data.
NAME = "ssl-al-ssl"
DESCRIPTION = "To be provided"
URL = "https://github.com/Chrisantos/ssl-al-ssl"
EMAIL = "chrisantus.eze@okstate.edu"
AUTHOR = "Chrisantus Eze"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "1.0.0"

# What packages are required for this module to be executed?
REQUIRED = [
    "torch",
    "torchvision",
    "pyyaml",
]

# What packages are optional?
EXTRAS = {
    "fancy feature": ["tensorboard"],
}

# Where the magic happens:
setup(
    name=NAME,
    version="", #about["__version__"],
    description=DESCRIPTION,
    long_description="", #long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    # $ setup.py publish support.
    # cmdclass={
    #     "upload": UploadCommand,
    # },
)