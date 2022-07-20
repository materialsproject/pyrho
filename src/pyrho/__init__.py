"""mp-pyrho package."""

__author__ = """Jimmy Shen"""
__email__ = "jmmshn@gmail.com"

import os

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass
