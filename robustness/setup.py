import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

quadprog_path = "quadprog-master/quadprog/"

extension = Extension(
    name="tltk-rob",
    sources=["backend_dir/tltk.pyx", "backend_dir/backend.c",
    quadprog_path+"aind.c",quadprog_path+"solve.QP.c",quadprog_path+"util.c",
    quadprog_path+"dpofa.c",quadprog_path+"daxpy.c",quadprog_path+"ddot.c",
    quadprog_path+"dscal.c",quadprog_path+"f2c_lite.c"
    ],
    #libraries=["backend"],
    extra_compile_args= ['-fopenmp'],
    extra_link_args=['-fopenmp'],
    library_dirs=["backend_dir"],
    include_dirs=["backend_dir","quadprog-master/quadprog"],
    language='c'

)


setup(
    name="tltk-rob",
    ext_modules=cythonize([extension], compiler_directives={'language_level' : "3"})
)
