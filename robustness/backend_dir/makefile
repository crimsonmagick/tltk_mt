CC = gcc

default: libbackend.a 

libbackend.a: backend.o
	ar rcs $@ $^

backend.o: backend.c backend.h
	$(CC) -c $<

clean:
	rm *.o *.a *.so

gpu:
	nvcc -shared -o libgpubackend.so gpubackend.cu --compiler-options '-fPIC'
gpu_test:
	nvcc -Xcompiler -fPIC -shared -o libgpubackend.so gpubackend.cu
