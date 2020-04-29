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
    void c_next_no_malloc(float* robustness,long length)

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
    float* c_global_no_malloc(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* global_robustness ,long length);
    
cdef extern from "backend.h":
    float* c_global_threaded(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);
    
cdef extern from "backend.h":
    float* c_global_threaded_no_malloc(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* global_robustness,long length);

cdef extern from "backend.h":
    float* c_until(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps, long length);
    
cdef extern from "backend.h":
    float* c_until_no_malloc(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness,float* until_robustness, float* time_stamps, long length);
    
cdef extern from "backend.h":
    float* c_until_threaded(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps, long length);

cdef extern from "backend.h":
    float* c_until_threaded_no_malloc(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps,float* until_robustness ,long length);

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
    

def py_next_numpy(robustness):
    cdef float[:] c_robustness = robustness
    c_next_no_malloc(&c_robustness[0],len(robustness))
    return c_robustness

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

def py_global_numpy(float lower_time_bound,float upper_time_bound,robustness,time_stamps) -> float[::1]:
    cdef float[:] c_robustness = robustness
    cdef float[:] c_time_stamps = time_stamps
    cdef float[:] c_results = np.empty(len(robustness),dtype=np.float32)
    
    c_global_no_malloc(lower_time_bound,upper_time_bound,&c_robustness[0],&c_time_stamps[0],&c_results[0],len(robustness))
    
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
    

    
def py_until_threaded_numpy(float lower_time_bound,float upper_time_bound,left_robustness,right_robustness,time_stamps) -> float[::1]:
    cdef float[:] c_left_robustness = left_robustness
    cdef float[:] c_right_robustness = right_robustness
    cdef float[:] c_time_stamps = time_stamps
    cdef float[:] c_results = np.empty(len(left_robustness),dtype=np.float32)
    
    c_until_threaded_no_malloc(lower_time_bound,upper_time_bound,&c_left_robustness[0],&c_right_robustness[0],&c_time_stamps[0],&c_results[0],len(left_robustness))
    
    return c_results
    
def py_until_numpy(float lower_time_bound,float upper_time_bound,left_robustness,right_robustness,time_stamps) -> float[::1]:
    cdef float[:] c_left_robustness = left_robustness
    cdef float[:] c_right_robustness = right_robustness
    cdef float[:] c_time_stamps = time_stamps
    cdef float[:] c_results = np.empty(len(left_robustness),dtype=np.float32)
    
    c_until_no_malloc(lower_time_bound,upper_time_bound,&c_left_robustness[0],&c_right_robustness[0],&c_time_stamps[0],&c_results[0],len(left_robustness))
    
    return c_results

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

#_______________MTL.py_______________________
GPU_LIB_FOUND = True
try:
    import gpubackend
except:
    GPU_LIB_FOUND = False
    #print("WARNING: No gpu libary found")
import ctypes
# import cvxpy as cp
import numpy as np
import scipy as sp
from multiprocessing import Pool
from multiprocessing import cpu_count
from time import time
#import quadprog_polyhedron

# trace[name] <= bound
# Time bounds inclusive
# 0 robustness is a failure (will add an option to choose later)

class Predicate:
    def __init__(self,variable_name,A_Matrix,bound,process_type = 'cpu',thread_pool = False):
        self.variable_name = variable_name
        self.value = None
        self.robustness_array = None
        self.bound = bound
        self.robustness = 0
        self.A_Matrix = A_Matrix
        self.thread_pool = thread_pool
        self.process_type = process_type
        
    
    def eval_interval(self,traces,time_stamps):
        if type(self.variable_name) != list:
            trace = traces[self.variable_name]
        else:
            iterts = []
            for name in self.variable_name:
                iterts.append(traces[name])
            trace = np.array(list(zip(*iterts)),dtype=np.float64)

        predicate_robustness = []
        np_A_Matrix = np.array(self.A_Matrix)
        
        if type(self.robustness_array) != type(None):
            return self.robustness_array
        
        if self.thread_pool == False:
            if ((len(trace.shape) == 1) and (type(self.A_Matrix) == int or type(self.A_Matrix) == float)):
                if self.process_type == 'cpu':
                    predicate_robustness = py_one_dim_pred_numpy(trace, self.A_Matrix, self.bound)
                    #predicate_robustness = py_one_dim_pred(list(trace), self.A_Matrix, self.bound)
                elif self.process_type == 'cpu_threaded':
                    predicate_robustness = py_one_dim_pred_threaded_numpy(trace, self.A_Matrix, self.bound)
                else:
                    predicate_robustness = gpubackend.py_one_dim_pred_numpy_gpu(list(trace), self.A_Matrix, self.bound)   
            else:
 
                #traces = np.transpose(np.array(traces)).tolist()
                #predicate_robustness = py_higher_dim(trace_size, n, m, q, l, u, init_A, init_P, traces, length, results)
                #if self.process_type == 'cpu':
                    predicate_robustness = solve_polyhedron_test(self.A_Matrix,self.bound,trace)
                #else:
                    #predicate_robustness = quadprog_polyhedron.solve_polyhedron_threaded(self.A_Matrix,self.bound,trace)
                    
                
                # for value in trace:
                    # np_value = np.array(value)
                    # # if np_value.size == 1 and np_A_Matrix.size == 1:
                        # # predicate_robustness.append(value * self.A_Matrix - self.bound)
                    # if (self.A_Matrix*np_value <= self.bound).all():            #calculate depth if np_value is in A*x <= b
                        # x = cp.Variable(np_value.size)
                        # objective = cp.Minimize(cp.norm(x - np_value))
                        # constraints = [self.A_Matrix*x >= self.bound]
                        # prob = cp.Problem(objective, constraints)
                        # predicate_robustness.append(-prob.solve(solver=cp.ECOS)) #ECOS has fastest time of ones tested. Going to add this as an argument eventually
                    # else:                                               #calculate distance if np_value is not in A*x <= b
                        # x = cp.Variable(np_value.size)
                        # objective = cp.Minimize(cp.norm(x - np_value))
                        # constraints = [self.A_Matrix*x <= self.bound]
                        # prob = cp.Problem(objective, constraints)
                        # predicate_robustness.append(prob.solve(solver=cp.ECOS))
        # else:
                                                                    # #I wanted to pass the Pool() as an argument but that weirdly causes an error. I then tryied making the thread pool in the object init but the same error happens. Not sure how to make the thread pool creation only happen once if the object is used more than once
            # with Pool(cpu_count()) as p:
                # t0 = time()
                # predicate_robustness = p.map(self.optimize_polyhedron, trace)
                # t1 = time()
                # print('Predicate time: ', t1 - t0)
                # p.close()
                # p.join()
        self.robustness = predicate_robustness[0]
        self.robustness_array = predicate_robustness
        return predicate_robustness
        
class Next:
    def __init__(self,subformula):
        self.subformula = subformula
        self.robustness = None
    def eval_interval(self,traces,time_stamps):
        subformula_robustness = self.subformula.eval_interval(traces,time_stamps)
        next_robustness = py_next_numpy(subformula_robustness)
        self.robustness = next_robustness[0]
        return next_robustness

class Global:
    def __init__(self,lower_time_bound,upper_time_bound,subformula = None, process_type = 'cpu_threaded'):
        self.value = True
        self.subformula = subformula
        self.truth_value_history = []
        self.robustness = float('inf')
        self.upper_time_bound = upper_time_bound
        self.lower_time_bound = lower_time_bound
        self.process_type = process_type
        
    def eval_interval(self,traces,time_stamps): 
        subformula_robustness = self.subformula.eval_interval(traces,time_stamps)
        globally_robustness = []
        max_robustness = float('-inf')
        
        #subformula_robustness.reverse()
        #time_stamps.reverse()
        
        if self.process_type == 'cpu':
            globally_robustness = py_global_numpy(self.lower_time_bound,self.upper_time_bound,subformula_robustness,time_stamps)
        elif self.process_type == 'cpu_threaded':
            #globally_robustness = py_global_threaded(self.lower_time_bound,self.upper_time_bound,list(subformula_robustness),list(time_stamps))
            globally_robustness = py_global_threaded_numpy(self.lower_time_bound,self.upper_time_bound,subformula_robustness,time_stamps)
        else:
            #print("global GPU computation")
            globally_robustness = gpubackend.py_global_numpy_gpu(self.lower_time_bound,self.upper_time_bound,subformula_robustness,time_stamps)
        t1 = time()
        #print("Global time: ", t1 - t0)
        self.robustness = globally_robustness[0]
        if self.robustness > 0:
            self.value = True
        #globally_robustness.reverse()
        return globally_robustness
    
    def add_subformula(self,subformula):
        self.subformula = subformula
    def get_subformula(self):
        return self.subformula

class Finally:
    def __init__(self,lower_time_bound,upper_time_bound,subformula = None, process_type = 'cpu_threaded'):
        self.subformula = subformula
        self.truth_value_history = []
        self.robustness = float('-inf')
        self.upper_time_bound = upper_time_bound
        self.lower_time_bound = lower_time_bound
        self.process_type = process_type
        
    def eval_interval(self,traces,time_stamps): 
        subformula_robustness = self.subformula.eval_interval(traces,time_stamps)
        finally_robustness = []
        max_robustness = float('-inf')
        #subformula_robustness.reverse()
        #time_stamps.reverse()
        t0 = time()
        if self.process_type == 'cpu':
            #print("finally CPU computation")
            #finally_robustness = py_finally(self.lower_time_bound,self.upper_time_bound,list(subformula_robustness),list(time_stamps))
            finally_robustness = py_finally_numpy(self.lower_time_bound,self.upper_time_bound,subformula_robustness,time_stamps)
        elif self.process_type == 'cpu_threaded':
            finally_robustness = py_finally_threaded_numpy(self.lower_time_bound,self.upper_time_bound,subformula_robustness,time_stamps)
            #finally_robustness = py_finally_threaded(self.lower_time_bound,self.upper_time_bound,list(subformula_robustness),list(time_stamps))
        else:
            finally_robustness = gpubackend.py_finally_numpy_gpu(self.lower_time_bound,self.upper_time_bound,subformula_robustness,time_stamps)

        #t1 = time()
        #print('Finally time:', t1 - t0)
        self.robustness = finally_robustness[0]
        
        if self.robustness > 0:
            self.value = True
        #finally_robustness.reverse()

        return  finally_robustness
        
        
    def add_subformula(self,subformula):
        self.subformula = subformula
    def get_subformula(self):
        return self.subformula

class Not:
    def __init__(self,subformula = None, process_type = "cpu_threaded"):
        self.subformula = subformula
        self.truth_value_history = []
        self.robustness = 0
        self.value = None
        self.process_type = process_type


    def eval_interval(self,traces,time_stamps): 
        subformula_robustness = self.subformula.eval_interval(traces,time_stamps)
        not_robustness = []
        # t0 = time()
        #not_robustness = [i * -1 for i in subformula_robustness]
        #c_subformula_robustness = (ctypes.c_float * len(subformula_robustness))(*subformula_robustness)
        # not_robustness = py_not(subformula_robustness)
        # t1 = time()
        # print('Not time: ', t1 - t0)
        if self.process_type == "cpu":
            not_robustness = py_not_numpy(subformula_robustness)
        elif self.process_type == "cpu_threaded":
            not_robustness = py_not_threaded_numpy(subformula_robustness)
        else:
            not_robustness = gpubackend.py_not_numpy_gpu(subformula_robustness)
        self.robustness = -self.subformula.robustness 
        
        return not_robustness


    def add_subformula(self,subformula):
        self.subformula = subformula
    def get_subformula(self):
        return self.subformula

class And:
    def __init__(self,left_subformula = None,right_subformula = None, process_type = 'cpu_threaded'):
        self.left_subformula = left_subformula
        self.right_subformula = right_subformula
        self.truth_value_history = []
        self.robustness = 0
        self.value = None
        self.process_type = process_type

    def eval_interval(self,traces,time_stamps):
        left_subformula_robustness = self.left_subformula.eval_interval(traces,time_stamps)
        right_subformula_robustness = self.right_subformula.eval_interval(traces,time_stamps)
        and_robustness = []
        # t0 = time()
        #for left_robustness,right_robustness in zip(left_subformula_robustness,right_subformula_robustness):
            
        #    and_robustness.append(min(left_robustness,right_robustness))
            
        #c_left_subformula_robustness = (ctypes.c_float * len(left_subformula_robustness))(*left_subformula_robustness)
        # t1 = time()
        # print('And time: ',t1-t0)
        if self.process_type == "cpu":
            and_robustness = py_and_numpy(left_subformula_robustness,right_subformula_robustness)
        elif self.process_type == "cpu_threaded":
            and_robustness = py_and_threaded_numpy(left_subformula_robustness,right_subformula_robustness)
        else:
            #print("GPU for AND")
            and_robustness = gpubackend.py_and_numpy_gpu(left_subformula_robustness,right_subformula_robustness)
        self.robustness = min(self.left_subformula.robustness,self.right_subformula.robustness)
        return and_robustness

class Or:
    def __init__(self,left_subformula = None,right_subformula = None, process_type = "cpu_threaded"):
        self.left_subformula = left_subformula
        self.right_subformula = right_subformula
        self.truth_value_history = []
        self.robustness = 0
        self.value = None
        self.process_type = process_type

    def eval_interval(self,traces,time_stamps): 
        left_subformula_robustness = self.left_subformula.eval_interval(traces,time_stamps)
        right_subformula_robustness = self.right_subformula.eval_interval(traces,time_stamps)
        or_robustness = []
        t0 = time()
        # for left_robustness,right_robustness in zip(left_subformula_robustness,right_subformula_robustness):
            
            # or_robustness.append(max(left_robustness,right_robustness))
        if self.process_type == "cpu":
            or_robustness = py_or_numpy(left_subformula_robustness,right_subformula_robustness)
        elif self.process_type == "cpu_threaded":
            or_robustness = py_or_threaded_numpy(left_subformula_robustness,right_subformula_robustness)
        else:
            #print("GPU for OR")
            or_robustness = gpubackend.py_or_numpy_gpu(left_subformula_robustness,right_subformula_robustness)
        
        t1 = time()
        #print('Or time: ', t1 - t0)
        
        
        self.robustness = max(self.left_subformula.robustness,self.right_subformula.robustness)
        return or_robustness

class Implication:
    def __init__(self,left_subformula = None,right_subformula = None, process_type = 'cpu'):
        self.left_subformula = left_subformula
        self.right_subformula = right_subformula
        self.truth_value_history = []
        self.robustness = 0
        self.value = None
        self.process_type = process_type

    def eval_interval(self,traces,time_stamps): 
        left_subformula_robustness = self.left_subformula.eval_interval(traces,time_stamps)
        right_subformula_robustness = self.right_subformula.eval_interval(traces,time_stamps)
        or_robustness = []

        for left_robustness,right_robustness in zip(left_subformula_robustness,right_subformula_robustness):
            
            or_robustness.append(max((-1*left_robustness),right_robustness))
        
            if (not left_robustness > 0) or right_robustness > 0:
                self.value = True
            else:
                self.value = False
        self.robustness = max(-self.left_subformula.robustness,right_subformula.robustness)

        return or_robustness

class Until:
    def __init__(self,lower_time_bound,upper_time_bound,left_subformula = None,right_subformula = None,process_type = 'cpu_threaded'):
        self.left_subformula = left_subformula
        self.right_subformula = right_subformula
        self.upper_time_bound = upper_time_bound
        self.lower_time_bound = lower_time_bound
        self.truth_value_history = []
        self.robustness = 0
        self.value = True
        self.right_subformula_true = False
        self.process_type = process_type 


    def eval_interval(self,traces,time_stamps): 
        left_subformula_robustness = self.left_subformula.eval_interval(traces,time_stamps)
        right_subformula_robustness = self.right_subformula.eval_interval(traces,time_stamps)
    
        until_robustness = []
        left_subformula_robustness_history = []
        inner_formula_min = []
        last_robustness = float('-inf')
        if self.process_type == "cpu":
            until_robustness = until_robustness = py_until_numpy(self.lower_time_bound,self.upper_time_bound,left_subformula_robustness,right_subformula_robustness,time_stamps)
        elif self.process_type == "cpu_threaded":
            until_robustness = until_robustness = py_until_threaded_numpy(self.lower_time_bound,self.upper_time_bound,left_subformula_robustness,right_subformula_robustness,time_stamps)
        else:
            until_robustness = gpubackend.py_until_numpy_gpu(self.lower_time_bound,self.upper_time_bound,left_subformula_robustness,right_subformula_robustness,list(time_stamps))
        self.robustness = until_robustness[0]
        return until_robustness

#_______________quadprog.pyx______________

cdef extern:
    void wrap_polyhedron_two(double** traces,double* C,double* b,double* results,int m,int n,long length)

cdef extern:
    void wrap_polyhedron_threaded(double** traces,double* C,double* b,double* results,int m,int n,long length)
    
def solve_polyhedron_test(C, b, traces):

    n3, m1 = C.shape[1], C.shape[0]
    
    cdef double** traces_
    cdef long length = len(traces)
    c_results = <double *>malloc(len(traces)*cython.sizeof(double))
    traces_ = <double **>malloc(len(traces)*cython.sizeof(c_results))
    
    for time_step in xrange(len(traces)):
        traces_[time_step] = <double *>malloc(len(traces[0]) * cython.sizeof(double))
        for i in xrange(len(traces[0])):
            #print(traces[time_step][i], end=",")
            traces_[time_step][i] = traces[time_step][i]
            #sys.stdout.write("%lf," % traces[time_step][i])
            
    C.transpose()
    b.transpose()
    
    cdef double[::1, :] C_ = np.array(C, copy=True, order='F')
    cdef double[::1] b_ = np.array(b, copy=True, order='F')
    cdef double[:] results = np.empty(length)

    wrap_polyhedron_two(traces_,&C_[0,0],&b_[0],&results[0],m1,n3,length)
    

    return np.array(results,dtype=np.float32)
    
