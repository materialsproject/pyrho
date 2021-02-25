# -*- coding: utf-8 -*-
import sys

from pkg_resources import VersionConflict, require
from setuptools import setup, find_packages

try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setup(
        name="mp-pyrho",
        use_pyscaffold=True,
        version="0.0.1",
        packages=find_packages("src"),
        package_dir={"": "src"},
        package_data={"pyrho": ["py.typed"]},
        description="Tools for re-griding periodic volumetric data for machine-learning applications.",
        long_description=long_description,
        longdescription_content_type="text/markdown",
    )
