LIB_DIR=backend_dir
QUADPROG_DIR=quadprog-master
default: backend

backend: setup.py backend.pyx $(LIB_DIR)/libbackend.a $(QUADPROG_DIR)/'quadprog_polyhedron%.so' 
	python3 setup.py build_ext --inplace && rm -f backend.c && rm -Rf build	
	
$(LIB_DIR)/libbackend.a:
	make -C $(LIB_DIR) libbackend.a

$(QUADPROG_DIR)/'quadprog_polyhedron%.so':
	cd $(QUADPROG_DIR) && python3 setup.py build_ext --inplace && mv quadprog_polyhedron*.so .. && cd ..


clean:
	rm *.so
gpu: $(LIB_DIR)/libgpubackend.so
	python3 setup_gpu.py build_ext --inplace && rm -f gpubackend.cpp &&rm -Rf build

$(LIB_DIR)/libgpubackend.so:
	make $(LIB_DIR) gpu
