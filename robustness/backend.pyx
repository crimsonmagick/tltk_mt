#backend.pyx
#This file contains the wrapper code so python can run C code for calculuating MTL robustness

import numpy as np
cimport cython 
from cpython cimport array
from scipy.sparse import csc_matrix
from libc.stdlib cimport malloc, free
from ctypes import *
import array 

cdef extern from "backend.h":
    void c_not(float* robustness,long length)

cdef extern from "backend.h":
    void c_or(float* left_robustness,float* right_robustness,long length)

cdef extern from "backend.h":
    void c_and(float* left_robustness,float* right_robustness,long length)

cdef extern from "backend.h":
    void c_not_threaded(float* robustness,long length)

cdef extern from "backend.h":
    void c_or_threaded(float* left_robustness,float* right_robustness,long length)

cdef extern from "backend.h":
    void c_and_threaded(float* left_robustness,float* right_robustness,long length)


cdef extern from "backend.h":
    float* c_finally(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);

cdef extern from "backend.h":
    float* c_finally_no_malloc(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* finally_robustness ,long length);

cdef extern from "backend.h":
    float* c_finally_threaded(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);

cdef extern from "backend.h":
    float* c_finally_threaded_no_malloc(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, float* finally_robustness,long length);

cdef extern from "backend.h":
    float* c_global(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);
    
cdef extern from "backend.h":
    float* c_global_threaded(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);
    
cdef extern from "backend.h":
    float* c_global_threaded_no_malloc(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* global_robustness,long length);

cdef extern from "backend.h":
    float* c_until(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps, long length);
    
cdef extern from "backend.h":
    float* c_until_threaded(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps, long length);

cdef extern from "backend.h":
    void c_one_dim_pred(float* traces, float A, float bound,long length);

cdef extern from "backend.h":
    void c_one_dim_pred_threaded(float* traces, float A, float bound,long length);


#Wrapper for MTL not operation
#   robustness: A list of python floats
#   returns: A list of python floats
def py_not(list robustness) -> float[::1]:
    cdef float * c_robustness
    c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))

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

#Wrapper for MTL and operation
#   left_robustness: A list of python floats
#   right_robustness: A list of python floats
#   returns: A list of python floats
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

    with nogil:
        free(c_left_robustness)
        free(c_right_robustness)

    return left_robustness
    
def py_and_numpy(left_robustness,right_robustness) -> float[::1]:
    cdef float[:] c_left_robustness = left_robustness
    cdef float[:] c_right_robustness = right_robustness
        
    c_and(&c_left_robustness[0],&c_right_robustness[0],len(left_robustness))
    
    return c_left_robustness

def py_and_threaded_numpy(left_robustness,right_robustness) -> float[::1]:
    cdef float[:] c_left_robustness = left_robustness
    cdef float[:] c_right_robustness = right_robustness
        
    c_and_threaded(&c_left_robustness[0],&c_right_robustness[0],len(left_robustness))
    
    return c_left_robustness

def py_or_numpy(left_robustness,right_robustness) -> float[::1]:
    cdef float[:] c_left_robustness = left_robustness
    cdef float[:] c_right_robustness = right_robustness
        
    c_or(&c_left_robustness[0],&c_right_robustness[0],len(left_robustness))
    
    return c_left_robustness


def py_or_threaded_numpy(left_robustness,right_robustness) -> float[::1]:
    cdef float[:] c_left_robustness = left_robustness
    cdef float[:] c_right_robustness = right_robustness
    c_or_threaded(&c_left_robustness[0],&c_right_robustness[0],len(left_robustness))
    
    return c_left_robustness
    
def py_not_numpy(robustness) -> float[::1]:
    cdef float[:] c_robustness = robustness
        
    c_not(&c_robustness[0],len(robustness))
    
    return c_robustness


def py_not_threaded_numpy(robustness) -> float[::1]:
    cdef float[:] c_robustness = robustness
        
    c_not_threaded(&c_robustness[0],len(robustness))
    
    return c_robustness

#    return left_robustness

#Wrapper for the MTL or operation
#   left_robustness: A list of python floats
#   right_robustness: A list of python floats
#   returns: A list of python floats
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
    with nogil:
        free(c_left_robustness)
        free(c_right_robustness)
    
    return left_robustness

#Wrapper for the MTL finally operation
    #lower_time_bound: A python float
    #upper_time_bound: A pyhton float
    #robustness: A list containing floats
    #time_stamps: A list containing floats
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
    with nogil:
        free(c_robustness)
        free(c_time_stamps)
        free(c_results)
    return robustness

#Wrapper for the MTL finally operation that takes advantage of parallel processing
    #lower_time_bound: A python float
    #upper_time_bound: A pyhton float
    #robustness: A list containing floats
    #time_stamps: A list containing floats

def py_finally_threaded_numpy(float lower_time_bound,float upper_time_bound,robustness,time_stamps) -> float[::1]:
    cdef float[:] c_robustness = robustness
    cdef float[:] c_time_stamps = time_stamps
    cdef float[:] c_results = np.empty(len(robustness),dtype=np.float32)
    
    c_finally_threaded_no_malloc(lower_time_bound,upper_time_bound,&c_robustness[0],&c_time_stamps[0],&c_results[0],len(robustness))
    
    return c_results
    
def py_global_threaded_numpy(float lower_time_bound,float upper_time_bound,robustness,time_stamps) -> float[::1]:
    cdef float[:] c_robustness = robustness
    cdef float[:] c_time_stamps = time_stamps
    cdef float[:] c_results = np.empty(len(robustness),dtype=np.float32)
    
    c_global_threaded_no_malloc(lower_time_bound,upper_time_bound,&c_robustness[0],&c_time_stamps[0],&c_results[0],len(robustness))
    
    return c_results
    
def py_finally_numpy(float lower_time_bound,float upper_time_bound,robustness,time_stamps) -> float[::1]:
    cdef float[:] c_robustness = robustness
    cdef float[:] c_time_stamps = time_stamps
    cdef float[:] c_results = np.empty(len(robustness),dtype=np.float32)
    
    c_finally_no_malloc(lower_time_bound,upper_time_bound,&c_robustness[0],&c_time_stamps[0],&c_results[0],len(robustness))
    
    return c_results

def py_finally_threaded(float lower_time_bound,float upper_time_bound,list robustness,list time_stamps) -> float[::1]:
    cdef float * c_robustness
    cdef float * c_time_stamps
    cdef float * c_results
    
    c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))
    c_time_stamps = <float *>malloc(len(time_stamps)*cython.sizeof(float))
    
    for i in xrange(len(robustness)): 
        c_robustness[i] = robustness[i]
        c_time_stamps[i] = time_stamps[i]
    
    c_results = c_finally_threaded(lower_time_bound,upper_time_bound,c_robustness,c_time_stamps,len(robustness))
    
    for i in xrange(len(robustness)): 
        robustness[i] = c_results[i]
    with nogil:
        free(c_robustness)
        free(c_time_stamps)
        free(c_results)
    return robustness

#Wrapper for the MTL global operation
    #lower_time_bound: A python float
    #upper_time_bound: A pyhton float
    #robustness: A list containing floats
    #time_stamps: A list containing floats
def py_global(float lower_time_bound,float upper_time_bound,list robustness,list time_stamps) -> float[::1]:
    cdef float * c_robustness
    cdef float * c_time_stamps
    cdef float * c_results
    
    c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))
    c_time_stamps = <float *>malloc(len(time_stamps)*cython.sizeof(float))
    
    for i in xrange(len(robustness)): 
        c_robustness[i] = robustness[i]
        c_time_stamps[i] = time_stamps[i]
    
    c_results = c_global(lower_time_bound,upper_time_bound,c_robustness,c_time_stamps,len(robustness))
    
    for i in xrange(len(robustness)): 
        robustness[i] = c_results[i]
    with nogil:
        free(c_robustness)
        free(c_time_stamps)
        free(c_results)
    return robustness

#Wrapper for the MTL global operation that takes advantage of parallel processing
    #lower_time_bound: A python float
    #upper_time_bound: A pyhton float
    #robustness: A list containing floats
    #time_stamps: A list containing floats
def py_global_threaded(float lower_time_bound,float upper_time_bound,list robustness,list time_stamps) -> float[::1]:
    cdef float * c_robustness
    cdef float * c_time_stamps
    cdef float * c_results
    
    c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))
    c_time_stamps = <float *>malloc(len(time_stamps)*cython.sizeof(float))
    
    for i in xrange(len(robustness)): 
        c_robustness[i] = robustness[i]
        c_time_stamps[i] = time_stamps[i]
    
    c_results = c_global_threaded(lower_time_bound,upper_time_bound,c_robustness,c_time_stamps,len(robustness))
    
    for i in xrange(len(robustness)): 
        robustness[i] = c_results[i]
    with nogil:
        free(c_robustness)
        free(c_time_stamps)
        free(c_results)
    return robustness

#Wrapper for the MTL until operation (left_robustness[i] Until right_robustness[i])
    #lower_time_bound: A python float
    #upper_time_bound: A pyhton float
    #left_robustness: A list containing floats
    #right_robustness: A list containing floats
    #time_stamps: A list containing floats
def py_until(float lower_time_bound,float upper_time_bound,list left_robustness,list right_robustness,list time_stamps) -> float[::1]:
    cdef float * c_left_robustness
    cdef float * c_right_robustness
    cdef float * c_time_stamps
    cdef float * c_results
    
    c_left_robustness = <float *>malloc(len(left_robustness)*cython.sizeof(float))
    c_right_robustness = <float *>malloc(len(right_robustness)*cython.sizeof(float))
    c_time_stamps = <float *>malloc(len(time_stamps)*cython.sizeof(float))
    
    for i in xrange(len(left_robustness)):
        c_left_robustness[i] = left_robustness[i]
        c_right_robustness[i] = right_robustness[i]
        c_time_stamps[i] = time_stamps[i]
    
    c_results = c_until(lower_time_bound,upper_time_bound,c_left_robustness,c_right_robustness,c_time_stamps,len(left_robustness))
    
    for i in xrange(len(left_robustness)): #Same here
        left_robustness[i] = c_results[i]
    with nogil:
        free(c_left_robustness)
        free(c_right_robustness)
        free(c_time_stamps)
        free(c_results)
        
    return left_robustness

#Wrapper for the MTL until operation that takes advantage of parallel processing (left_robustness[i] Until right_robustness[i])
    #lower_time_bound: A python float
    #upper_time_bound: A pyhton float
    #left_robustness: A list containing floats
    #right_robustness: A list containing floats
    #time_stamps: A list containing floats
def py_until_threaded(float lower_time_bound,float upper_time_bound,list left_robustness,list right_robustness,list time_stamps) -> float[::1]:
    cdef float * c_left_robustness
    cdef float * c_right_robustness
    cdef float * c_time_stamps
    cdef float * c_results
    
    c_left_robustness = <float *>malloc(len(left_robustness)*cython.sizeof(float))
    c_right_robustness = <float *>malloc(len(right_robustness)*cython.sizeof(float))
    c_time_stamps = <float *>malloc(len(time_stamps)*cython.sizeof(float))
    
    for i in xrange(len(left_robustness)): 
        c_left_robustness[i] = left_robustness[i]
        c_right_robustness[i] = right_robustness[i]
        c_time_stamps[i] = time_stamps[i]
    
    c_results = c_until_threaded(lower_time_bound,upper_time_bound,c_left_robustness,c_right_robustness,c_time_stamps,len(left_robustness))
    
    for i in xrange(len(left_robustness)):
        left_robustness[i] = c_results[i]
    with nogil:
        free(c_left_robustness)
        free(c_right_robustness)
        free(c_time_stamps)
        free(c_results)
        
    return left_robustness
#Wrapper for a one dimisional polyheadron predicate A*robustness[i] <= bound
#   robustness: A list of python floats
#   A: A python float
#   bound: A python float
def py_one_dim_pred(list robustness, float A, float bound) -> float[::1]:
    cdef float * c_robustness
    c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))
    if c_robustness is NULL:
        raise MemoryError()
    
    for i in xrange(len(robustness)): #Sure this can be done better
        c_robustness[i] = robustness[i]
    
    c_one_dim_pred(c_robustness,A,bound,len(robustness))
    
    for i in xrange(len(robustness)): #Same here
        robustness[i] = c_robustness[i]
    with nogil:
        free(c_robustness)
    
    return robustness


def py_one_dim_pred_numpy(robustness, float A, float bound) -> float[::1]:
    cdef float[:] c_robustness = robustness

        
    c_one_dim_pred(&c_robustness[0],A,bound,len(robustness))
    

    
    return robustness

def py_one_dim_pred_threaded_numpy(robustness, float A, float bound) -> float[::1]:
    cdef float[:] c_robustness = robustness

        
    c_one_dim_pred_threaded(&c_robustness[0],A,bound,len(robustness))
    

    
    return robustness
