CC = gcc

default: libbackend.a

libbackend.a: backend.o
	ar rcs $@ $^

backend.o: backend.c backend.h
	$(CC) -c $<

clean:
	rm *.o *.a
