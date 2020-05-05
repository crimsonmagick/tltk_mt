import setuptools
import os
from distutils.extension import Extension
from Cython.Build import cythonize


with open("../mkdocs/docs/getting_started.md", "r") as fh:
    long_description = fh.read()


if os.path.isfile("tltk_mtl/tltk_mtl.pyx"):
    use_cython = True
else:
    use_cython = False

quadprog_path = "quadprog-master/quadprog/"
if use_cython:
    extension = [Extension(
        name="tltk_mtl",
        sources=["tltk_mtl/tltk_mtl.pyx", "tltk_mtl/backend.c",
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
        name="tltk_mtl",
        sources=["tltk_mtl/tltk_mtl.c", "tltk_mtl/backend.c",
        quadprog_path+"aind.c",quadprog_path+"solve.QP.c",quadprog_path+"util.c",
        quadprog_path+"dpofa.c",quadprog_path+"daxpy.c",quadprog_path+"ddot.c",
        quadprog_path+"dscal.c",quadprog_path+"f2c_lite.c"
        ],
        #libraries=["backend"],
        extra_compile_args= ['-fopenmp'],
        extra_link_args=['-fopenmp'],
        library_dirs=["backend_dir"],
        include_dirs=["tltk_mtl","quadprog-master/quadprog"],
        language='c',
    )]   

setuptools.setup(
    name="tltk_mtl",
    ext_modules=cythonize(extension, compiler_directives={'language_level' : "3"}),
    version="0.0.4",
    author="Kole Cralley",
    author_email="jkolecr@gmail.com",
    description="A libary for effecient Metric temporal logic calculation",
    python_requires='>=3.6',
    classifiers=["Programming Language :: Python :: 3",
                  "Operating System :: POSIX :: Linux"],
    Platform="Linux",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown"
)
