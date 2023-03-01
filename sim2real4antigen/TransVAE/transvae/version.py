from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 6
_version_micro = 3  # use '' for first of series, number for 1 and above
# _version_extra = 'dev1'
_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

with open('README.md') as readme_file:
    README = readme_file.read()

NAME = "transvae"
MAINTAINER = "Orion Dollar"
MAINTAINER_EMAIL = "orion.dollar@gmail.com"
DESCRIPTION = "A package for training and analyzing attention-based VAEs for molecular design."
CONTENT_TYPE="text/markdown"
URL = "https://github.com/oriondollar/TransVAE"
DOWNLOAD_URL = ""
LICENSE = "MIT License"
AUTHOR = "Orion Dollar"
AUTHOR_EMAIL = "orion.dollar@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'TransVAE': [pjoin('data', '*')]}
REQUIRES = ['pandas', 'numpy', 'torch', 'seaborn', 'matplotlib', 'scipy', 'sklearn']
