#!/usr/bin/env python

from setuptools import setup, find_packages
from pathlib import Path

module_dir = Path(__file__).resolve().parent

with open(module_dir / "README.md") as f:
    long_desc = f.read()

setup(
    name="pyrho",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="Tools for processing volumetric data for material science applications",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/jmmshn/pyrho",
    author="Jimmy Shen",
    author_email="jmmshn@gmail.com",
    license="modified BSD",
    packages=find_packages("src"),
    package_dir={"": "src"},
    zip_safe=False,
    install_requires=["setuptools"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Machine Learning",
        "Operating System :: OS Independent",
        "Topic :: Other/Nonlisted Topic",
        "Topic :: Database :: Back-Ends",
        "Topic :: Scientific/Engineering",
    ],
    tests_require=["pytest"],
    python_requires=">=3.7",
)
