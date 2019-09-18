import ctypes as ct
import numpy as np
import sys
import os
import time
from array import array

try:
    lib = ct.cdll.LoadLibrary('./backend_dir/libgpubackend.so')
except:
    print(os.getcwd())
    lib = None
    
def c_gpu_or(left_robustness,right_robustness):
    if lib == None:
        print("No gpu shared object file found", file=sys.stderr)
        sys.exit(os.EX_OSFILE)

    np_left_robustness = np.array(left_robustness, dtype=np.float32)
    np_right_robustness = np.array(right_robustness, dtype=np.float32)
    # pointer_left = (ct.c_float * len(left_robustness))(*left_robustness)
    # pointer_right = (ct.c_float * len(right_robustness))(*right_robustness)

    
    pointer_left = np_left_robustness.ctypes.data_as(ct.POINTER(ct.c_float))
    pointer_right = np_right_robustness.ctypes.data_as(ct.POINTER(ct.c_float))
    
    lib.c_or_gpu.argtypes = [ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.c_long]
    lib.c_or_gpu.restype = ct.c_void_p
    
    t0 = time.time()
    lib.c_or_gpu(pointer_left,pointer_right,ct.c_long(len(left_robustness)))
    t1 = time.time()
    
    print("gpu or time:",t1 - t0)
    
    return np_left_robustness



