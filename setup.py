from setuptools import setup, Extension
import pybind11
import sys

cpp_args = []
if sys.platform == 'win32':
    cpp_args.extend(['/O2', '/openmp'])
else:
    cpp_args.extend(['-O3', '-fopenmp'])

ext_modules = [
    Extension(
        name='pyterp',
        sources=['src/interpolator.cpp'],
        include_dirs=[
            'src',
            pybind11.get_include(),
        ],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(ext_modules=ext_modules)