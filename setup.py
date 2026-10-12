"""Build the portable C++ coverage extension."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "vector_server._coverage", ["native/coverage.cpp"], cxx_std=17
        ),
        Pybind11Extension("vector_server._regions", ["native/regions.cpp"], cxx_std=17),
    ],
    cmdclass={"build_ext": build_ext},
)
