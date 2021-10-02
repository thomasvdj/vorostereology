# 
# setup.py : vorostereology
#

from setuptools import setup, Extension
import sys


compile_args = ['-std=c++11', '-O3', '-fopenmp']
link_args = ['-fopenmp']
if "MSC" in sys.version:
    compile_args = ["/std:c11", "/O2", '/openmp']
    link_args = ['/openmp']

extensions = [
    Extension("vorostereology.voroplusplus",
              sources=["vorostereology/voroplusplus.cpp",
                       "vorostereology/vpp.cpp",
                       "src/voro++.cc"],
              extra_compile_args=compile_args,
              extra_link_args=link_args,
              include_dirs=["src"],
              language="c++",
              )
]

setup(
    name="vorostereology",
    version="1.0",
    description="A python interface to the voro++ library based on pyvoro, along with functions related to generating regularized Laguerre-Voronoi diagrams and computing cross sections.",
    author="Thomas van der Jagt",
    packages=["vorostereology", ],
    package_dir={"vorostereology": "vorostereology"},
    install_requires=[
    	'cython',
    	'numpy',
    	'scipy',
    	'matplotlib',
    ],
    ext_modules=extensions,
    keywords=["geometry", "mathematics", "Voronoi"],
    classifiers=[
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    test_suite="test",
)
