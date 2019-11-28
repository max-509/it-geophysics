import sys
import os
import subprocess
from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import numpy

class CMakeExtension(Extension):
	def __init__(self, name, cmake_lists_dir='.', **kwa):
		Extension.__init__(self, name, sources=[], **kwa)
		self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)

class cmake_build_ext(build_ext):
	def build_extensions(self):
		try:
			out = subprocess.check_output(['cmake', '--version'])
		except OSError:
			raise RuntimeError('Cannot find cmake')

		for ext in self.extensions:
			extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
			cfg = 'Release'
			cmake_args = [
				'-DCMAKE_BUILD_TYPE=%s' % cfg,
				'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
				'-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), self.build_temp),
				'-DPYTHON_EXECUTABLE={}'.format(sys.executable)
			]

			if not os.path.exists(self.build_temp):
				os.makedirs(self.build_temp)
			subprocess.check_call(['cmake', ext.cmake_lists_dir]+cmake_args,
										cwd=self.build_temp)
			subprocess.check_call(['cmake', '--build', '.', '--config'],
										cwd=self.build_temp)

setup (name = 'Coherent summation',
       version = '1.0',
       description = 'Coherent summation package',
       author = 'Vershinin Maxim',
       author_email = 'm.vershinin@g.nsu.ru',
       ext_modules = [CMakeExtension("CoherentSumModule", '.')],
       cmdclass = {'build_ext': cmake_build_ext})