CC = gcc

default: libbackend.a 

libbackend.a: backend.o
	ar rcs $@ $^

backend.o: backend.c backend.h
	$(CC) -c -losqp -fopenmp -lm -Wall $<

clean:
	rm *.o *.a *.so

gpu:
	nvcc -shared -o libgpubackend.so gpubackend.cu --compiler-options '-fPIC'

