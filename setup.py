from setuptools import setup, Extension
import pybind11
import sys

cpp_args = []
link_args = []

if sys.platform == 'win32':
    cpp_args.extend(['/std:c++20', '/O2', '/openmp'])
    link_args.extend(['/openmp'])
else:
    cpp_args.extend(['-std=c++20', '-O3', '-fopenmp'])
    link_args.extend(['-fopenmp'])

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
        extra_link_args=link_args,
    ),
]

setup(ext_modules=ext_modules)