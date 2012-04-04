from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("acstb", ["acstb.pyx"]), 
               Extension("map_c", ["map_c.pyx"])]

setup(
  name = 'Cython Modules',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)