# Authors: CommPy contributors
# License: BSD 3-Clause

from setuptools import setup

# Taken from scikit-learn setup.py
DISTNAME = 'scikit-commpy'
DESCRIPTION = 'Digital Communication Algorithms with Python'
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Veeresh Taranalli & Bastien Trotobas'
MAINTAINER_EMAIL = 'veeresht@gmail.com'
URL = 'http://veeresht.github.com/CommPy'
LICENSE = 'BSD 3-Clause'
VERSION = '0.7.0'

#This is a list of files to install, and where
#(relative to the 'root' dir, where setup.py is)
#You could be more specific.
files = ["channelcoding/*", "channelcoding/tests/*", "tests/*", "channelcoding/designs/ldpc/gallager/*", "channelcoding/designs/ldpc/wimax/*"]

setup(
    name=DISTNAME,
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
    packages=['commpy', 'commpy.channelcoding', 'commpy.channelcoding.tests', 'commpy.tests'],
    install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'sympy'
    ],
    #'package' package must contain files (see list above)
    #This dict maps the package name =to=> directories
    #It says, package *needs* these files.
    package_data={'commpy': files},
    #'runner' is in the root.
    scripts=["runner"],
    test_suite='nose.collector',
    tests_require=['nose'],

    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Telecommunications Industry',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
    python_requires='>=3.2',
)
