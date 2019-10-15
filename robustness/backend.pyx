import numpy as np
cimport cython 
from cpython cimport array
from scipy.sparse import csc_matrix
from libc.stdlib cimport malloc, free
from ctypes import *
import array 

cdef extern from "backend.h":
    double* higher_dim_pred(long trace_size, long long int n, long long int m, double* q,double* l, double* u, int A_nnz, int P_nnz, double** traces, long length, double* P_data, long long int* P_indices, long long int* P_indptr, double* A_data, long long int* A_indices, long long int* A_indptr,double* init_A);
    
cdef extern from "backend.h":
    double* higher_dim_pred_threaded(long trace_size, long long int n, long long int m, double* q,double* l, double* u, int A_nnz, int P_nnz, double** traces, long length, double* P_data, long long int* P_indices, long long int* P_indptr, double* A_data, long long int* A_indices, long long int* A_indptr, double* init_A)
    
cdef extern from "backend.h":
    void c_not(float* robustness,long length)

cdef extern from "backend.h":
    void c_or(float* left_robustness,float* right_robustness,long length)

cdef extern from "backend.h":
    void c_and(float* left_robustness,float* right_robustness,long length)

cdef extern from "backend.h":
    float* c_finally(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);
    
cdef extern from "backend.h":
    float* c_finally_threaded(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);

cdef extern from "backend.h":
    float* c_global(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);
    
cdef extern from "backend.h":
    float* c_global_threaded(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);

cdef extern from "backend.h":
    float* c_until(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps, long length);
    
cdef extern from "backend.h":
    float* c_until_threaded(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps, long length);

cdef extern from "backend.h":
    void c_one_dim_pred(float* traces, float A, float bound,long length);

def py_higher_dim(long trace_size, long long int n, long long int m, list q, list l, list u, list init_A, list init_P, list traces, long length, list results) -> double[::1]:
    cdef double* c_q
    cdef double* c_l
    cdef double* c_u
    cdef double *A
    cdef double *P_data_c
    cdef double *A_data_c
    cdef long long int *P_indices_c
    cdef long long int *P_indptr_c
    cdef long long int *A_indices_c
    cdef long long int *A_indptr_c
    cdef double** P
    cdef double** c_traces
    cdef float* c_traces_onedim
    cdef double* result
    cdef int P_nnz, A_nnz
    
    c_q = <double *>malloc(len(q) * cython.sizeof(double))
    c_l = <double *>malloc(len(l) * cython.sizeof(double))
    c_u = <double *>malloc(len(u) * cython.sizeof(double))
    result = <double *>malloc(length*cython.sizeof(double));
    c_traces = <double **>malloc(trace_size * cython.sizeof(c_q))
    #print(traces[0])
    for i in xrange(trace_size):
        c_traces[i] = <double*>malloc(length*cython.sizeof(double))
        for j in xrange(length):
            c_traces[i][j] = traces[i][j]
    A = <double*>malloc(n*m*cython.sizeof(double))
    counter = 0
    for i in xrange(m):
        for j in xrange(n):
            if m > 1:
                A[counter] = init_A[i][j]
            else:
                A[counter] = init_A[j]
            counter += 1
            
    P_csc = csc_matrix(init_P, dtype=np.float64)   
            
    P_data = P_csc.data
    P_data_c = <double *>malloc(np.size(P_data)*cython.sizeof(double))
    for i in xrange(np.size(P_data)):
        P_data_c[i] = P_data[i]
        
    P_indices = P_csc.indices
    P_indices_c = <long long int*>malloc(np.size(P_indices)*cython.sizeof(n))
    for i in xrange(np.size(P_indices)):
        P_indices_c[i] = P_indices[i]
        
    P_indptr = P_csc.indptr
    P_indptr_c = <long long int*>malloc(np.size(P_indptr)*cython.sizeof(n))
    for i in xrange(np.size(P_indptr)):
        P_indptr_c[i] = P_indptr[i]
        
    A_csc = csc_matrix(init_A, dtype=np.float64)
        
    A_data = A_csc.data
    A_data_c = <double *>malloc(np.size(A_data)*cython.sizeof(double))
    for i in xrange(np.size(A_data)):
        A_data_c[i] = A_data[i]
        
    A_indices = A_csc.indices
    A_indices_c = <long long int*>malloc(np.size(A_indices)*cython.sizeof(n))
    for i in xrange(np.size(A_indices)):
        A_indices_c[i] = A_indices[i]
        
    A_indptr = A_csc.indptr
    A_indptr_c = <long long int*>malloc(np.size(A_indptr)*cython.sizeof(n))
    for i in xrange(np.size(A_indptr)):
        A_indptr_c[i] = A_indptr[i]
    
    for i in xrange(len(q)):
        c_q[i] = q[i]
    for i in xrange(len(l)):
        c_l[i] = l[i]
    for i in xrange(len(u)):
        c_u[i] = u[i]
        
    P_nnz = np.count_nonzero(init_P)
    A_nnz = np.count_nonzero(init_A)
    '''
    print(P_data)
    print(P_indices)
    print(P_indptr)
    print(A_data)
    print(A_indices)
    print(A_indptr)
    '''
    result = higher_dim_pred(trace_size, n, m, c_q, c_l, c_u, A_nnz, P_nnz, c_traces, length, P_data_c, P_indices_c, P_indptr_c, A_data_c, A_indices_c, A_indptr_c, A)
    
    for i in xrange(length):
        results[i] = result[i]
    with nogil:
        free(c_q)
        free(c_l)
        free(c_u)
        #for i in xrange(length):
        #    free(c_traces[i])
        #free(c_traces)
        free(A)  
    return results
    
def py_higher_dim_threaded(long trace_size, long long int n, long long int m, list q, list l, list u, list init_A, list init_P, list traces, long length, list results) -> double[::1]:
    cdef double* c_q
    cdef double* c_l
    cdef double* c_u
    cdef double *A
    cdef double *P_data_c
    cdef double *A_data_c
    cdef long long int *P_indices_c
    cdef long long int *P_indptr_c
    cdef long long int *A_indices_c
    cdef long long int *A_indptr_c
    cdef double** P
    cdef double** c_traces
    cdef float* c_traces_onedim
    cdef double* result
    cdef int P_nnz, A_nnz
    
    c_q = <double *>malloc(len(q) * cython.sizeof(double))
    c_l = <double *>malloc(len(l) * cython.sizeof(double))
    c_u = <double *>malloc(len(u) * cython.sizeof(double))
    result = <double *>malloc(length*cython.sizeof(double));
    c_traces = <double **>malloc(trace_size * cython.sizeof(c_q))
    #print(traces[0])
    for i in xrange(trace_size):
        c_traces[i] = <double*>malloc(length*cython.sizeof(double))
        for j in xrange(length):
            c_traces[i][j] = traces[i][j]
    A = <double*>malloc(n*m*cython.sizeof(double))
    counter = 0
    for i in xrange(m):
        for j in xrange(n):
            if m > 1:
                A[counter] = init_A[i][j]
            else:
                A[counter] = init_A[j]
            counter += 1
            
    P_csc = csc_matrix(init_P, dtype=np.float64)   
            
    P_data = P_csc.data
    P_data_c = <double *>malloc(np.size(P_data)*cython.sizeof(double))
    for i in xrange(np.size(P_data)):
        P_data_c[i] = P_data[i]
        
    P_indices = P_csc.indices
    P_indices_c = <long long int*>malloc(np.size(P_indices)*cython.sizeof(n))
    for i in xrange(np.size(P_indices)):
        P_indices_c[i] = P_indices[i]
        
    P_indptr = P_csc.indptr
    P_indptr_c = <long long int*>malloc(np.size(P_indptr)*cython.sizeof(n))
    for i in xrange(np.size(P_indptr)):
        P_indptr_c[i] = P_indptr[i]
        
    A_csc = csc_matrix(init_A, dtype=np.float64)
        
    A_data = A_csc.data
    A_data_c = <double *>malloc(np.size(A_data)*cython.sizeof(double))
    for i in xrange(np.size(A_data)):
        A_data_c[i] = A_data[i]
        
    A_indices = A_csc.indices
    A_indices_c = <long long int*>malloc(np.size(A_indices)*cython.sizeof(n))
    for i in xrange(np.size(A_indices)):
        A_indices_c[i] = A_indices[i]
        
    A_indptr = A_csc.indptr
    A_indptr_c = <long long int*>malloc(np.size(A_indptr)*cython.sizeof(n))
    for i in xrange(np.size(A_indptr)):
        A_indptr_c[i] = A_indptr[i]
    
    for i in xrange(len(q)):
        c_q[i] = q[i]
    for i in xrange(len(l)):
        c_l[i] = l[i]
    for i in xrange(len(u)):
        c_u[i] = u[i]
        
    P_nnz = np.count_nonzero(init_P)
    A_nnz = np.count_nonzero(init_A)
    result = higher_dim_pred_threaded(trace_size, n, m, c_q, c_l, c_u, A_nnz, P_nnz, c_traces, length, P_data_c, P_indices_c, P_indptr_c, A_data_c, A_indices_c, A_indptr_c, A)
    
    for i in xrange(length):
        results[i] = result[i]
    with nogil:
        free(c_q)
        free(c_l)
        free(c_u)
        #for i in xrange(length):
        #    free(c_traces[i])
        #free(c_traces)
        free(A)  
    return results



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
    
def py_finally_threaded(float lower_time_bound,float upper_time_bound,list robustness,list time_stamps) -> float[::1]:
    cdef float * c_robustness
    cdef float * c_time_stamps
    cdef float * c_results
    
    c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))
    c_time_stamps = <float *>malloc(len(time_stamps)*cython.sizeof(float))
    
    for i in xrange(len(robustness)): #Sure this can be done better
        c_robustness[i] = robustness[i]
        c_time_stamps[i] = time_stamps[i]
    
    c_results = c_finally_threaded(lower_time_bound,upper_time_bound,c_robustness,c_time_stamps,len(robustness))
    
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
    
def py_global_threaded(float lower_time_bound,float upper_time_bound,list robustness,list time_stamps) -> float[::1]:
    cdef float * c_robustness
    cdef float * c_time_stamps
    cdef float * c_results
    
    c_robustness = <float *>malloc(len(robustness)*cython.sizeof(float))
    c_time_stamps = <float *>malloc(len(time_stamps)*cython.sizeof(float))
    
    for i in xrange(len(robustness)): #Sure this can be done better
        c_robustness[i] = robustness[i]
        c_time_stamps[i] = time_stamps[i]
    
    c_results = c_global_threaded(lower_time_bound,upper_time_bound,c_robustness,c_time_stamps,len(robustness))
    
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

def py_until_threaded(float lower_time_bound,float upper_time_bound,list left_robustness,list right_robustness,list time_stamps) -> float[::1]:
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
    
    c_results = c_until_threaded(lower_time_bound,upper_time_bound,c_left_robustness,c_right_robustness,c_time_stamps,len(left_robustness))
    
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

