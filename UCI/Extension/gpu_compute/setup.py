from distutils.core import setup, Extension

name = "Gpu_compute"
setup(name=name, ext_modules=[Extension(name, sources=["src/main.c"])])
