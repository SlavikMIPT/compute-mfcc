import os
import sys

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ['-std=c++11']

sfc_module = Extension(
    'bearmfcc', sources=['module.cpp'],
    include_dirs=['pybind11/include'],
    language='c++',
    extra_compile_args=cpp_args,
)

setup(
    name='bearmfcc',
    version='1.0',
    description='Python package with bearmfcc C++ extension (PyBind11)',
    ext_modules=[sfc_module],
)
