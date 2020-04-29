LIB_DIR=tltk_rob
QUADPROG_DIR=quadprog-master
default: backend

backend: setup.py $(LIB_DIR)/libbackend.a #$(QUADPROG_DIR)/'quadprog_polyhedron%.so' 
	python3 setup.py build_ext --inplace && rm -f backend.c && rm -Rf build	

$(LIB_DIR)/libbackend.a:
	make -C $(LIB_DIR) libbackend.a

#$(QUADPROG_DIR)/'quadprog_polyhedron%.so':
#	cd $(QUADPROG_DIR) && python3 setup.py build_ext --inplace && mv quadprog_polyhedron*.so .. && #cd ..


clean:
	rm -f *.so
	rm -f backend_dir/*.so backend_dir/*.a backend_dir/*.o
	rm -f -r __pycache__
	rm -f  quadprog-master/build/lib.linux-x86_64-3.8/*.so
	rm -f  quadprog-master/build/temp.linux-x86_64-3.8/quadprog/*.o
	rm -f -r ../__pycache__
	rm -f -r ../mkdocs/site
	rm -f -r build dist tltk_rob.egg-info
gpu: $(LIB_DIR)/libgpubackend.so
	python3 $(LIB_DIR)/setup_gpu.py build_ext --inplace && rm -f gpubackend.cpp &&rm -Rf build

$(LIB_DIR)/libgpubackend.so:
	make $(LIB_DIR) gpu

wheel:
	python3 setup.py sdist bdist_wheel
