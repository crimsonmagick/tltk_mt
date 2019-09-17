import ctypes
from ctypes import *
import numpy as np
lib = cdll.LoadLibrary('./backend_dir/libgpubackend.so')

left = np.array([1,234,34,543], dtype=np.float32)
right = np.array([3234,134,64,5465], dtype=np.float32)

p_left = left.ctypes.data_as(POINTER(c_float))
p_right = right.ctypes.data_as(POINTER(c_float))

lib.c_or_gpu.argtypes = [POINTER(c_float), POINTER(c_float), c_long]
lib.c_or_gpu.restype = c_void_p

lib.c_or_gpu(p_left,p_right,c_long(len(left)))

print(left)
