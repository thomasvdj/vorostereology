# vorostereology
A Python package for computing Laguerre tessellations (aka Laguerre-Voronoi diagrams) and planar sections (cross sections) of these tessellations. In particular, centroidal Laguerre tessellations with a chosen cell volume distribution can be computed.

While the package is fully functional I intend to re-write some parts still, both for increasing performance as for better organizing the code structure. Documentation is currently limited to docstrings in _init_.py and the demos in the "examples" folder. The package uses source code of the C++ package Voro++, and the Python package pyvoro. The License accompanied with these software packages is LICENSE.txt.

# Warning
The package may still change in functionality and implementation, backwards compatibility is not yet guaranteed.

# Installation
The package may be installed via pip.

Option 1: pip install git+https://github.com/thomasvdj/vorostereology.git 

Option 2: Download and extract source code, cd into the folder and install via: pip install .

# MacOS support
For performance several functions are paralellized via OpenMP. This may complicate installation on MacOS since clang does not have OpenMP support. We refer to https://scikit-learn.org/stable/developers/advanced_installation.html which is another package which depends on OpenMP. There are two options given for enabling OpenMP on MacOS. Installation on Windows using MSVC or Linux via gcc should work without issues.
