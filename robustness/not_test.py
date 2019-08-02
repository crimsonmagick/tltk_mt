import backend
import ctypes

test = [1,2,3]
arr = (ctypes.c_float * len(test))(*test)
print(backend.py_not(arr))
