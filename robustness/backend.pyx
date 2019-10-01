import numpy as np
cimport cython 
from cpython cimport array
from libc.stdlib cimport malloc, free
import array 

cdef extern from "backend.h":
    c_float higher_dim_pred(c_int n, c_int m, c_float* q,c_float* l, c_float* u, c_float **init_A, c_float **init_P)
    
cdef extern from "backend.h":
    void c_not(float* robustness,long length)

cdef extern from "backend.h":
    void c_or(float* left_robustness,float* right_robustness,long length)

cdef extern from "backend.h":
    void c_and(float* left_robustness,float* right_robustness,long length)

cdef extern from "backend.h":
    float* c_finally(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);

cdef extern from "backend.h":
    float* c_global(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);

cdef extern from "backend.h":
    float* c_until(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps, long length);

cdef extern from "backend.h":
    void c_one_dim_pred(float* traces, float A, float bound,long length);
    
#cdef extern from "gpubackend.h":
#    void c_or_gpu(float* left_robustness,float* right_robustness,long length)

#def py_or_gpu(list left_robustness,list right_robustness) -> float[::1]:
#    cdef float * c_left_robustness
#    cdef float * c_right_robustness
    
#    c_left_robustness = <float *>malloc(len(left_robustness)*cython.sizeof(float))
#    c_right_robustness = <float *>malloc(len(left_robustness)*cython.sizeof(float))
    
#    if c_left_robustness is NULL or c_right_robustness is NULL:
#        raise MemoryError()
        
#    for i in xrange(len(left_robustness)): #Sure this can be done better
#        c_left_robustness[i] = left_robustness[i]
#        c_right_robustness[i] = right_robustness[i]
    
#    c_or_gpu(c_left_robustness,c_right_robustness,len(left_robustness))
    
#    for i in xrange(len(left_robustness)): #Same here
#        left_robustness[i] = c_left_robustness[i]
#    #list_results = np.ndarray((len(left_robustness), ), 'f', c_left_robustness, order='C')
##    left_robustness = c_left_robustness[:len(left_robustness)]
#    with nogil:
#        free(c_left_robustness)
#        free(c_right_robustness)
    
#    return left_robustness

# c_int n, c_int m, c_float* q,c_float* l, c_float* u, c_float **init_A, c_float **init_P
def py_higher_dim(int n, int m, float* q, float* l, float* u, float** init_A, float** init_P) -> float[::1]:
    cdef float result
    result = higher_dim_pred(n, m, q, l, u, init_A, init_P)
    return result

def py_not(list robustness) -> float[::1]:
    cdef float * c_robustness
    c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))
#    cdef array.array c_array_robustness = array.array('f',robustness)
#    cdef float[:] c_robustness = c_array_robustness
    if c_robustness is NULL:
        raise MemoryError()
    
    for i in xrange(len(robustness)): #Sure this can be done better
        c_robustness[i] = robustness[i]
    
    c_not(c_robustness,len(robustness))
    
    for i in xrange(len(robustness)): #Same here
        robustness[i] = c_robustness[i]
    
    #list_results = np.ndarray((len(robustness), ), 'f', c_robustness, order='C')
    
    with nogil:
        free(c_robustness)
    
    return robustness

def py_and(list left_robustness,list right_robustness) -> float[::1]:
    cdef float * c_left_robustness
    cdef float * c_right_robustness
    
    c_left_robustness = <float *>malloc(len(left_robustness)*cython.sizeof(float))
    c_right_robustness = <float *>malloc(len(left_robustness)*cython.sizeof(float))
    
    if c_left_robustness is NULL or c_right_robustness is NULL:
        raise MemoryError()
        
    for i in xrange(len(left_robustness)): #Sure this can be done better
        c_left_robustness[i] = left_robustness[i]
        c_right_robustness[i] = right_robustness[i]
    
    c_and(c_left_robustness,c_right_robustness,len(left_robustness))
    
    for i in xrange(len(left_robustness)): #Same here
        left_robustness[i] = c_left_robustness[i]
    #list_results = np.ndarray((len(left_robustness), ), 'f', c_left_robustness, order='C')
#    left_robustness = c_left_robustness[:len(left_robustness)]
    with nogil:
        free(c_left_robustness)
        free(c_right_robustness)
    
    return left_robustness

def py_or(list left_robustness,list right_robustness) -> float[::1]:
    cdef float * c_left_robustness
    cdef float * c_right_robustness
    
    c_left_robustness = <float *>malloc(len(left_robustness)*cython.sizeof(float))
    c_right_robustness = <float *>malloc(len(left_robustness)*cython.sizeof(float))
    
    if c_left_robustness is NULL or c_right_robustness is NULL:
        raise MemoryError()
        
    for i in xrange(len(left_robustness)): #Sure this can be done better
        c_left_robustness[i] = left_robustness[i]
        c_right_robustness[i] = right_robustness[i]
    
    c_or(c_left_robustness,c_right_robustness,len(left_robustness))
    
    for i in xrange(len(left_robustness)): #Same here
        left_robustness[i] = c_left_robustness[i]
    #list_results = np.ndarray((len(left_robustness), ), 'f', c_left_robustness, order='C')
#    left_robustness = c_left_robustness[:len(left_robustness)]
    with nogil:
        free(c_left_robustness)
        free(c_right_robustness)
    
    return left_robustness


def py_finally(float lower_time_bound,float upper_time_bound,list robustness,list time_stamps) -> float[::1]:
    cdef float * c_robustness
    cdef float * c_time_stamps
    cdef float * c_results
    
    c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))
    c_time_stamps = <float *>malloc(len(time_stamps)*cython.sizeof(float))
    
    for i in xrange(len(robustness)): #Sure this can be done better
        c_robustness[i] = robustness[i]
        c_time_stamps[i] = time_stamps[i]
    
    c_results = c_finally(lower_time_bound,upper_time_bound,c_robustness,c_time_stamps,len(robustness))
    
    for i in xrange(len(robustness)): #Same here
        robustness[i] = c_results[i]
        
    #list_results = np.ndarray((len(robustness), ), 'f', c_robustness, order='C')
    
    with nogil:
        free(c_robustness)
        free(c_time_stamps)
        free(c_results)
    return robustness

def py_global(float lower_time_bound,float upper_time_bound,list robustness,list time_stamps) -> float[::1]:
    cdef float * c_robustness
    cdef float * c_time_stamps
    cdef float * c_results
    
    c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))
    c_time_stamps = <float *>malloc(len(time_stamps)*cython.sizeof(float))
    
    for i in xrange(len(robustness)): #Sure this can be done better
        c_robustness[i] = robustness[i]
        c_time_stamps[i] = time_stamps[i]
    
    c_results = c_global(lower_time_bound,upper_time_bound,c_robustness,c_time_stamps,len(robustness))
    
    for i in xrange(len(robustness)): #Same here
        robustness[i] = c_results[i]
        
    #list_results = np.ndarray((len(robustness), ), 'f', c_robustness, order='C')
    
    with nogil:
        free(c_robustness)
        free(c_time_stamps)
        free(c_results)
    return robustness


def py_until(float lower_time_bound,float upper_time_bound,list left_robustness,list right_robustness,list time_stamps) -> float[::1]:
    cdef float * c_left_robustness
    cdef float * c_right_robustness
    cdef float * c_time_stamps
    cdef float * c_results
    
    c_left_robustness = <float *>malloc(len(left_robustness)*cython.sizeof(float))
    c_right_robustness = <float *>malloc(len(right_robustness)*cython.sizeof(float))
    c_time_stamps = <float *>malloc(len(time_stamps)*cython.sizeof(float))
    
    for i in xrange(len(left_robustness)): #Sure this can be done better
        c_left_robustness[i] = left_robustness[i]
        c_right_robustness[i] = right_robustness[i]
        c_time_stamps[i] = time_stamps[i]
    
    c_results = c_until(lower_time_bound,upper_time_bound,c_left_robustness,c_right_robustness,c_time_stamps,len(left_robustness))
    
    for i in xrange(len(left_robustness)): #Same here
        left_robustness[i] = c_results[i]
        
    #list_results = np.ndarray((len(robustness), ), 'f', c_robustness, order='C')
    
    with nogil:
        free(c_left_robustness)
        free(c_right_robustness)
        free(c_time_stamps)
        free(c_results)
        
    return left_robustness




def py_one_dim_pred(list robustness, float A, float bound) -> float[::1]:
    cdef float * c_robustness
    c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))
#    cdef array.array c_array_robustness = array.array('f',robustness)
#    cdef float[:] c_robustness = c_array_robustness
    if c_robustness is NULL:
        raise MemoryError()
    
    for i in xrange(len(robustness)): #Sure this can be done better
        c_robustness[i] = robustness[i]
    
    c_one_dim_pred(c_robustness,A,bound,len(robustness))
    
    for i in xrange(len(robustness)): #Same here
        robustness[i] = c_robustness[i]
    
    #list_results = np.ndarray((len(robustness), ), 'f', c_robustness, order='C')
    
    with nogil:
        free(c_robustness)
    
    return robustness

