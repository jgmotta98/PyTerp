from setuptools import setup, Extension
import pybind11
import sys
from glob import glob

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


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='PyTerp',
    version='0.1.0',
    author='Jonathan Motta',
    author_email='jonathangmotta98@gmail.com',
    description='A package for performing optimized k-NN IDW interpolation using C++.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
)