from setuptools import setup, Extension
import pybind11
import sys

extra_compile_args = ['-O3', '-march=native', '-fopenmp']
extra_link_args = ['-fopenmp']

if sys.platform == 'win32':
     extra_compile_args = ['/O2', '/openmp']
     extra_link_args = []


ext_modules = [
    Extension(
        'PyTerp',
        ['src/interpolator.cpp'],
        include_dirs=[
            pybind11.get_include(),
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='PyTerp',
    version='0.1.0',
    author='Jonathan Motta',
    description='A package for performing optimized KNN IDW interpolation using C++.',
    ext_modules=ext_modules,
)