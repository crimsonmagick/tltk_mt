import setuptools
import os
from distutils.extension import Extension
from Cython.Build import cythonize

if os.path.isfile("tltk_rob/tltk_rob.pyx"):
    use_cython = True
else:
    use_cython = False

quadprog_path = "quadprog-master/quadprog/"
if use_cython:
    extension = [Extension(
        name="tltk_rob",
        sources=["tltk_rob/tltk_rob.pyx", "tltk_rob/backend.c",
        quadprog_path+"aind.c",quadprog_path+"solve.QP.c",quadprog_path+"util.c",
        quadprog_path+"dpofa.c",quadprog_path+"daxpy.c",quadprog_path+"ddot.c",
        quadprog_path+"dscal.c",quadprog_path+"f2c_lite.c"
        ],
        #libraries=["backend"],
        extra_compile_args= ['-fopenmp'],
        extra_link_args=['-fopenmp'],
        library_dirs=["backend_dir"],
        include_dirs=["backend_dir","quadprog-master/quadprog"],
        language='c',
    )]   
else:
    extension = [Extension(
        name="tltk_rob",
        sources=["tltk_rob/tltk_rob.c", "tltk_rob/backend.c",
        quadprog_path+"aind.c",quadprog_path+"solve.QP.c",quadprog_path+"util.c",
        quadprog_path+"dpofa.c",quadprog_path+"daxpy.c",quadprog_path+"ddot.c",
        quadprog_path+"dscal.c",quadprog_path+"f2c_lite.c"
        ],
        #libraries=["backend"],
        extra_compile_args= ['-fopenmp'],
        extra_link_args=['-fopenmp'],
        library_dirs=["backend_dir"],
        include_dirs=["tltk_rob","quadprog-master/quadprog"],
        language='c',
    )]   

setuptools.setup(
    name="tltk_rob",
    ext_modules=cythonize(extension, compiler_directives={'language_level' : "3"}),
    version="0.0.6",
    author="Kole Cralley",
    author_email="jkolecr@gmail.com",
    description="A libary for effecient Metric temporal logic calculation",
    python_requires='>=3.6',
    classifiers=["Programming Language :: Python :: 3",
                  "Operating System :: POSIX :: Linux"],
    Platform="Linux",
)
