from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize([
        Extension(
            "dtw_jhbai",
            ["dtw_jhbai.pyx", "dtw_core.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3"]
        )
    ])
)
