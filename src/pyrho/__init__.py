# -*- coding: utf-8 -*-
__author__ = """Jimmy Shen"""
__email__ = "jmmshn@gmail.com"
__version__ = "0.1.0"

from pkg_resources import get_distribution, DistributionNotFound
import os

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass

test_files_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../test_files"
