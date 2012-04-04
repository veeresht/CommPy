import os
from setuptools import setup, find_packages

# Taken from scikit-learn setup.py
DISTNAME = 'scikit-commpy'
DESCRIPTION = 'Digital Communication Algorithms with Python'
LONG_DESCRIPTION = open('README').read()
MAINTAINER = 'Veeresh Taranalli'
MAINTAINER_EMAIL = 'veeresht@gmail.com'
URL = 'http://veeresht.github.com/CommPy'
LICENSE = 'GPL'
# DOWNLOAD_URL = 'http://sourceforge.net/projects/scikit-learn/files/'
VERSION = '0.1.0'

#This is a list of files to install, and where
#(relative to the 'root' dir, where setup.py is)
#You could be more specific.
files = ["channelcoding/*"]

setup(
    name = DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    #Name the folder where your packages live:
    #(If you have other packages (dirs) or modules (py files) then
    #put them into the package directory - they will be found 
    #recursively.)
    packages = ['commpy'],
    #'package' package must contain files (see list above)
    #This dict maps the package name =to=> directories
    #It says, package *needs* these files.
    package_data = {'commpy' : files },
    #'runner' is in the root.
    scripts = ["runner"],
    long_description = """ Work in progress """, 
    classifiers = [
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Telecommunications Industry',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ]
) 