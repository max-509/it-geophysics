from distutils.core import setup, Extension
import numpy

module = Extension("CoherentSumModule", sources=['module.cpp'],
					extra_compile_args=["-O3", "-mavx2", "-fopenmp", "-std=c++11"],
					include_dirs=["/usr/local/include", numpy.get_include()],
					libraries=["gomp"],
					extra_link_args=['coherent_summation.a'])

setup (name = 'Coherent summation',
       version = '1.0',
       description = 'Coherent summation package',
       author = 'Vershinin Maxim',
       author_email = 'm.vershinin@g.nsu.ru',
       ext_modules = [module])