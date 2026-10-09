import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

class get_pybind_include(object):
    def __str__(self):
        return pybind11.get_include()

ext_modules = [
    Extension(
        'pcb_engine',
        ['pcb_engine.cpp'],
        include_dirs=[get_pybind_include()],
        language='c++'
    ),
]

class BuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        opts = []
        if compiler_type == 'msvc':
            opts.append('/O2')
            opts.append('/openmp') # 开启 MSVC OpenMP
            opts.append('/std:c++17')
        else:
            opts.append('-O3')
            opts.append('-fopenmp') # 开启 GCC/Clang OpenMP
            opts.append('-std=c++17')
            self.compiler.linker_so.append('-fopenmp')

        for ext in self.extensions:
            ext.extra_compile_args = opts
        super().build_extensions()

setup(
    name='pcb_engine',
    version='1.0',
    description='C++ Native PCB GA Routing Engine',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)