from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extension = Extension(
    name="backend",
    sources=["backend.pyx","backend_dir/backend.c"],
    libraries=["backend"],
    extra_compile_args= ['-fopenmp'],
    extra_link_args=['-fopenmp'],
    library_dirs=["backend_dir"],
    include_dirs=["backend_dir"]

)


setup(
    name="backend",
    ext_modules=cythonize([extension], compiler_directives={'language_level' : "3"})
)
