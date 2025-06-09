from setuptools import setup, Extension, find_packages
import sys
import pybind11
import os

_SRC_PATH = "src"


cpp_extension = Extension(
    name='pyterp',
    sources=[os.path.join(_SRC_PATH, 'interpolator.cpp')],
    include_dirs=[
        _SRC_PATH,
        pybind11.get_include(),
    ],
    language='c++',
    extra_compile_args=['/O2', '/openmp'] if sys.platform == 'win32' else ['-O3', '-fopenmp'],
)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='pyterp',
    version='0.1.0',
    author='Jonathan Motta',
    author_email='jonathangmotta98@gmail.com',
    description='A package for performing optimized k-NN IDW interpolation using C++.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    ext_modules=[cpp_extension],
    zip_safe=False,
)