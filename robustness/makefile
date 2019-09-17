LIB_DIR = backend_dir

default: backend

backend: setup.py backend.pyx $(LIB_DIR)/libbackend.a
	python3 setup.py build_ext --inplace && rm -f backend.c && rm -Rf build

$(LIB_DIR)/libbackend.a:
	make -C $(LIB_DIR) backend.a

clean:
	rm *.so

