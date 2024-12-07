from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "big_little_transformer",
        ["big_little_transformer.cpp"],
        include_dirs=["/usr/include/eigen3"],  # Ruta a Eigen
    ),
]

setup(
    name="big_little_transformer",
    version="1.0",
    author="Tu Nombre",
    description="Modelo Big-Little Transformer en C++ con Pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
