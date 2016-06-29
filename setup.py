
# Authors: Veeresh Taranalli <veeresht@gmail.com>
# License: BSD 3-Clause

import os, sys, shutil, numpy
from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension

#use_cython = False

#try:
#    from Cython.Distutils import build_ext
    #use_cython = True
#except ImportError:
    #from distutils.command import build_ext
#    use_cython = False

#cmdclass = { }
#ext_modules = [ ]

#if use_cython:
#    ext_modules += [
#        Extension("commpy.channelcoding.acstb", [ "commpy/channelcoding/acstb.pyx" ], include_dirs=[numpy.get_include()]),
#        Extension("commpy.channelcoding.map_c", [ "commpy/channelcoding/map_c.pyx" ], include_dirs=[numpy.get_include()])
#    ]
#    cmdclass.update({ 'build_ext': build_ext })
#    print "Using Cython"
#else:
#    ext_modules += [
#        Extension("commpy.channelcoding.acstb", [ "commpy/channelcoding/acstb.c" ], include_dirs=[numpy.get_include()]),
#        Extension("commpy.channelcoding.map_c", [ "commpy/channelcoding/map_c.c" ], include_dirs=[numpy.get_include()])
#    ]

# Taken from scikit-learn setup.py
DISTNAME = 'scikit-commpy'
DESCRIPTION = 'Digital Communication Algorithms with Python'
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Veeresh Taranalli'
MAINTAINER_EMAIL = 'veeresht@gmail.com'
URL = 'http://veeresht.github.com/CommPy'
LICENSE = 'BSD 3-Clause'
# DOWNLOAD_URL = 'http://sourceforge.net/projects/scikit-learn/files/'
VERSION = '0.3.0'

#This is a list of files to install, and where
#(relative to the 'root' dir, where setup.py is)
#You could be more specific.
files = ["channelcoding/*, channelcoding/tests/*"]

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
    packages = ['commpy', 'commpy.channelcoding', 'commpy.channelcoding.tests'],
    #package_dir={
    #    'commpy' : 'commpy',
    #},
    install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
    ],
       #'package' package must contain files (see list above)
    #This dict maps the package name =to=> directories
    #It says, package *needs* these files.
    package_data = {'commpy' : files },
    #'runner' is in the root.
    scripts = ["runner"],
    test_suite='nose.collector',
    tests_require=['nose'],

    long_description = LONG_DESCRIPTION,
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
