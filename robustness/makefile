LIB_DIR = backend_dir

default: backend

backend: setup.py backend.pyx $(LIB_DIR)/libbackend.a
	python3 setup.py build_ext --inplace && rm -f backend.c && rm -Rf build

$(LIB_DIR)/libbackend.a:
	make -C $(LIB_DIR) libbackend.a

clean:
	rm *.so
gpu: $(LIB_DIR)/libgpubackend.a
	python3 setup_gpu.py build_ext --inplace && rm -f gpubackend.cpp &&rm -Rf build

$(LIB_DIR)/libgpubackend.a:
	make $(LIB_DIR) gpu
