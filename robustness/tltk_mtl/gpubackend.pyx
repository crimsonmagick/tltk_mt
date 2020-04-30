import numpy as np
cimport numpy as np
cimport cython 
from cpython cimport array
from libc.stdlib cimport malloc, free
import array 

cdef extern from "backend_dir/gpubackend.h":
	void predicate_setup(float* cpu_traces, float A, float bound,long length)

cdef extern from "backend_dir/gpubackend.h":
	void c_or_gpu(float* left_robustness, float* right_robustness, long length)

cdef extern from "backend_dir/gpubackend.h":	
	void c_not_gpu(float* robustness, long length)

cdef extern from "backend_dir/gpubackend.h":
	void predicate_setup(float* cpu_traces, float A, float bound,long length)

cdef extern from "backend_dir/gpubackend.h":
	void c_and_gpu(float* left_robustness, float* right_robustness, long length)

cdef extern from "backend_dir/gpubackend.h":
	void c_finally_gpu(float* cpu_traces, float* cpu_time_stamps, float* results,float lower_time_bound, float upper_time_bound,long length)

cdef extern from "backend_dir/gpubackend.h":
	void c_global_gpu(float* cpu_traces, float* cpu_time_stamps, float* results,float lower_time_bound, float upper_time_bound,long length)

cdef extern from "backend_dir/gpubackend.h":
	float* c_until_gpu(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps,float* results, long length)

# void predicate_setup(float* cpu_traces, float A, float bound,long length)
def py_one_dim_pred_gpu(list robustness, float A, float bound) -> float[::1]:
	cdef float * c_robustness
	c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))
	if c_robustness is NULL:
		raise MemoryError()

	for i in xrange(len(robustness)): #Sure this can be done better
		c_robustness[i] = robustness[i]

	predicate_setup(c_robustness,A,bound,len(robustness))
    
	for i in xrange(len(robustness)): #Same here
		robustness[i] = c_robustness[i]
    
	with nogil:
		free(c_robustness)
    
	return robustness

def py_one_dim_pred_numpy_gpu(robustness, float A, float bound) -> float[::1]:
	cdef float[:] c_robustness = robustness      
	predicate_setup(&c_robustness[0],A,bound,len(robustness))
	return robustness

def py_or_gpu(list left_robustness,list right_robustness) -> float[::1]:
	cdef float * c_left_robustness
	cdef float * c_right_robustness
	c_left_robustness = <float *>malloc(len(left_robustness)*cython.sizeof(float))
	c_right_robustness = <float *>malloc(len(left_robustness)*cython.sizeof(float))
	if c_left_robustness is NULL or c_right_robustness is NULL:
		raise MemoryError()
	for i in xrange(len(left_robustness)): #Sure this can be done better
		c_left_robustness[i] = left_robustness[i]
		c_right_robustness[i] = right_robustness[i]
	c_or_gpu(c_left_robustness,c_right_robustness,len(left_robustness))
	for i in xrange(len(left_robustness)): #Same here
		left_robustness[i] = c_left_robustness[i]
	with nogil:
		free(c_left_robustness)
		free(c_right_robustness)
	return left_robustness

def py_or_numpy_gpu(left_robustness,right_robustness) -> float[::1]:
	cdef float[:] c_left_robustness = left_robustness
	cdef float[:] c_right_robustness = right_robustness

	c_or_gpu(&c_left_robustness[0],&c_right_robustness[0],len(left_robustness))

	return c_left_robustness

def py_and_gpu(list left_robustness,list right_robustness) -> float[::1]:
	cdef float * c_left_robustness
	cdef float * c_right_robustness
    
	c_left_robustness = <float *>malloc(len(left_robustness)*cython.sizeof(float))
	c_right_robustness = <float *>malloc(len(left_robustness)*cython.sizeof(float))
    
	if c_left_robustness is NULL or c_right_robustness is NULL:
		raise MemoryError()
        
	for i in xrange(len(left_robustness)): #Sure this can be done better
		c_left_robustness[i] = left_robustness[i]
		c_right_robustness[i] = right_robustness[i]
    
	c_and_gpu(c_left_robustness,c_right_robustness,len(left_robustness))
    
	for i in xrange(len(left_robustness)): #Same here
		left_robustness[i] = c_left_robustness[i]
    #list_results = np.ndarray((len(left_robustness), ), 'f', c_left_robustness, order='C')
#    left_robustness = c_left_robustness[:len(left_robustness)]
	with nogil:
		free(c_left_robustness)
		free(c_right_robustness)
    
	return left_robustness

def py_and_numpy_gpu(left_robustness,right_robustness) -> float[::1]:
	cdef float[:] c_left_robustness = left_robustness
	cdef float[:] c_right_robustness = right_robustness

	c_and_gpu(&c_left_robustness[0],&c_right_robustness[0],len(left_robustness))

	return c_left_robustness

def py_not_gpu(list robustness) -> float[::1]:
	cdef float * c_robustness
	c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))
#    cdef array.array c_array_robustness = array.array('f',robustness)
#    cdef float[:] c_robustness = c_array_robustness
	if c_robustness is NULL:
		raise MemoryError()
    
	for i in xrange(len(robustness)): #Sure this can be done better
		c_robustness[i] = robustness[i]
    
	c_not_gpu(c_robustness,len(robustness))
    
	for i in xrange(len(robustness)): #Same here
		robustness[i] = c_robustness[i]
    
    
	with nogil:
		free(c_robustness)
    
	return robustness

def py_not_numpy_gpu(robustness) -> float[::1]:
	cdef float[:] c_robustness = robustness

	c_not_gpu(&c_robustness[0],len(robustness))

	return c_robustness

def py_finally_gpu(float lower_time_bound,float upper_time_bound,list robustness,list time_stamps) -> float[::1]:
# void c_finally_gpu(float* cpu_traces, float* cpu_time_stamps, float* results,float lower_time_bound, float upper_time_bound,long length)
	cdef float * c_robustness
	cdef float * c_time_stamps
	cdef float * c_results
    
	c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))
	c_results = <float *>malloc(len(robustness)*cython.sizeof(float))
	c_time_stamps = <float *>malloc(len(time_stamps)*cython.sizeof(float))
	for i in xrange(len(robustness)): #Sure this can be done better
		c_robustness[i] = robustness[i]
		c_time_stamps[i] = time_stamps[i]
	c_finally_gpu(c_robustness, c_time_stamps, c_results, lower_time_bound,upper_time_bound,len(robustness))
	for i in xrange(len(robustness)): #Same here
		robustness[i] = c_results[i]
        
    #list_results = np.ndarray((len(robustness), ), 'f', c_robustness, order='C')
    
	with nogil:
		free(c_robustness)
		free(c_time_stamps)
		free(c_results)
	return robustness

def py_finally_numpy_gpu(float lower_time_bound,float upper_time_bound,robustness,time_stamps) -> float[::1]:
	cdef float[:] c_robustness = robustness
	cdef float[:] c_time_stamps = time_stamps
	cdef float[:] c_results = np.empty(len(robustness),dtype=np.float32)
	#c_finally_gpu(c_robustness, c_time_stamps, c_results, lower_time_bound,upper_time_bound,len(robustness))
	c_finally_gpu(&c_robustness[0],&c_time_stamps[0],&c_results[0],lower_time_bound,upper_time_bound,len(robustness))
	#c_finally_gpu(lower_time_bound,upper_time_bound,&c_robustness[0],&c_time_stamps[0],&c_results[0],len(robustness))

	return c_results

def py_global_gpu(float lower_time_bound,float upper_time_bound,list robustness,list time_stamps) -> float[::1]:
# c_global_gpu(float* cpu_traces, float* cpu_time_stamps, float* results,float lower_time_bound, float upper_time_bound,long length)
	cdef float * c_robustness
	cdef float * c_time_stamps
	cdef float * c_results
    
	c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))
	c_results = <float *>malloc(len(robustness)*cython.sizeof(float))
	c_time_stamps = <float *>malloc(len(time_stamps)*cython.sizeof(float))
    
	for i in xrange(len(robustness)): #Sure this can be done better
		c_robustness[i] = robustness[i]
		c_time_stamps[i] = time_stamps[i]
    
	c_global_gpu(c_robustness, c_time_stamps, c_results, lower_time_bound,upper_time_bound,len(robustness))
    
	for i in xrange(len(robustness)): #Same here
		robustness[i] = c_results[i]
        
    #list_results = np.ndarray((len(robustness), ), 'f', c_robustness, order='C')
    
	with nogil:
		free(c_robustness)
		free(c_time_stamps)
		free(c_results)
	return robustness

def py_global_numpy_gpu(float lower_time_bound,float upper_time_bound,robustness,time_stamps) -> float[::1]:
	cdef float[:] c_robustness = robustness
	cdef float[:] c_time_stamps = time_stamps
	cdef float[:] c_results = np.empty(len(robustness),dtype=np.float32)

	c_global_gpu(&c_robustness[0],&c_time_stamps[0],&c_results[0],lower_time_bound,upper_time_bound,len(robustness))
	
	return c_results

# c_until_gpu(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps,float* results, long length)
def py_until_gpu(float lower_time_bound,float upper_time_bound,list left_robustness,list right_robustness,list time_stamps) -> float[::1]:
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
    
	c_results = c_until_gpu(lower_time_bound,upper_time_bound,c_left_robustness,c_right_robustness,c_time_stamps, c_results, len(left_robustness))
    
	for i in xrange(len(left_robustness)): #Same here
		left_robustness[i] = c_results[i]
        
    #list_results = np.ndarray((len(robustness), ), 'f', c_robustness, order='C')
    
	with nogil:
		free(c_left_robustness)
		free(c_right_robustness)
		free(c_time_stamps)
		free(c_results)
        
	return left_robustness

def py_until_numpy_gpu(float lower_time_bound,float upper_time_bound,left_robustness,right_robustness,time_stamps) -> float[::1]:
	cdef float[:] c_left_robustness = left_robustness
	cdef float[:] c_right_robustness = right_robustness
	cdef float[:] c_time_stamps = time_stamps
	cdef float[:] c_results = np.empty(len(left_robustness),dtype=np.float32)

	c_until_gpu(lower_time_bound,upper_time_bound,&c_left_robustness[0],&c_right_robustness[0],&c_time_stamps[0],&c_results[0],len(left_robustness))

	return c_results
