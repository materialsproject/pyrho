"""mp-pyrho package."""

__author__ = """Jimmy Shen"""
__email__ = "jmmshn@gmail.com"

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
