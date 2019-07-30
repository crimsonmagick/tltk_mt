import numpy as np
cdef extern from "backend.h":
    void c_not(float* robustness,long length)

def py_not(float[::1] robustness) -> float[::1]:
    c_not(&robustness[0],robustness.size)
    list_results = np.ndarray((robustness.size, ), 'f', robustness, order='C')
    return list_results
