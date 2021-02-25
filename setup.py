#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from typing import List

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = []  # type: List[str]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3.7",
]

setup(
    author="Jimmy-Xuan Shen",
    author_email="jmmshn@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Tools for re-griding volumetric quantum chemistry data for machine-learning purposes.",
    install_requires=requirements,
    license="BSD license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    name="mp-pyrho",
    packages=find_packages("src"),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/materialsproject/pyRho",
    version="0.0.8",
    zip_safe=False,
)
