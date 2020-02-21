import sys
import backend
GPU_LIB_FOUND = True
try:
    import gpubackend
except:
    GPU_LIB_FOUND = False
    print("WARNING: No gpu libary found")
import ctypes
# import cvxpy as cp
import numpy as np
import scipy as sp
from multiprocessing import Pool
from multiprocessing import cpu_count
from time import time
import quadprog_polyhedron

# trace[name] <= bound
# Time bounds inclusive
# 0 robustness is a failure (will add an option to choose later)

class Predicate:
    def __init__(self,variable_name,A_Matrix,bound,process_type = 'cpu_threaded',thread_pool = False):
        self.variable_name = variable_name
        self.value = None
        self.robustness_array = None
        self.bound = bound
        self.robustness = 0
        self.A_Matrix = A_Matrix
        self.thread_pool = thread_pool
        self.process_type = process_type
        
    # def optimize_polyhedron(self,trace):
        # np_value = np.array(trace)
        # np_A_Matrix = np.array(self.A_Matrix)
        
        # x = cp.Variable(np_value.size)
        # objective = cp.Minimize(cp.norm(x - np_value))
        # if np_value.size == 1 and np_A_Matrix.size == 1:
            # return trace * self.A_Matrix - self.bound
        # elif (self.A_Matrix*np_value <= self.bound).all():                        #calculate depth if np_value is in A*x <= b
            # constraints = [self.A_Matrix*x >= self.bound]
            # prob = cp.Problem(objective, constraints)
            # return -prob.solve(solver=cp.ECOS)    #ECOS has fastest time of ones tested. Going to add this as an argument eventually
        # else:                                                           #calculate distance if np_value is not in A*x <= b
            # constraints = [self.A_Matrix*x <= self.bound]
            # prob = cp.Problem(objective, constraints)
            # return prob.solve(solver=cp.ECOS)
    
    def eval_interval(self,traces,time_stamps):
        trace = traces[self.variable_name]
        predicate_robustness = []
        np_A_Matrix = np.array(self.A_Matrix)
        
        if type(self.robustness_array) != type(None):
            return self.robustness_array
        
        if self.thread_pool == False:
            if ((len(trace.shape) == 1) and (type(self.A_Matrix) == int or type(self.A_Matrix) == float)):
                if self.process_type == 'cpu':
                    predicate_robustness = backend.py_one_dim_pred_numpy(trace, self.A_Matrix, self.bound)
                    #predicate_robustness = backend.py_one_dim_pred(list(trace), self.A_Matrix, self.bound)
                elif self.process_type == 'cpu_threaded':
                    predicate_robustness = backend.py_one_dim_pred_threaded_numpy(trace, self.A_Matrix, self.bound)
                else:
                    predicate_robustness = gpubackend.py_one_dim_pred_numpy_gpu(list(trace), self.A_Matrix, self.bound)   
            else:
 
                #traces = np.transpose(np.array(traces)).tolist()
                #predicate_robustness = backend.py_higher_dim(trace_size, n, m, q, l, u, init_A, init_P, traces, length, results)
                predicate_robustness = quadprog_polyhedron.solve_polyhedron_numpy(self.A_Matrix,self.bound,trace)
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
            globally_robustness = backend.py_global_numpy(self.lower_time_bound,self.upper_time_bound,subformula_robustness,time_stamps)
        elif self.process_type == 'cpu_threaded':
            #globally_robustness = backend.py_global_threaded(self.lower_time_bound,self.upper_time_bound,list(subformula_robustness),list(time_stamps))
            globally_robustness = backend.py_global_threaded_numpy(self.lower_time_bound,self.upper_time_bound,subformula_robustness,time_stamps)
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
            #finally_robustness = backend.py_finally(self.lower_time_bound,self.upper_time_bound,list(subformula_robustness),list(time_stamps))
            finally_robustness = backend.py_finally_numpy(self.lower_time_bound,self.upper_time_bound,subformula_robustness,time_stamps)
        elif self.process_type == 'cpu_threaded':
            finally_robustness = backend.py_finally_threaded_numpy(self.lower_time_bound,self.upper_time_bound,subformula_robustness,time_stamps)
            #finally_robustness = backend.py_finally_threaded(self.lower_time_bound,self.upper_time_bound,list(subformula_robustness),list(time_stamps))
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
        # not_robustness = backend.py_not(subformula_robustness)
        # t1 = time()
        # print('Not time: ', t1 - t0)
        if self.process_type == "cpu":
            not_robustness = backend.py_not_numpy(subformula_robustness)
        elif self.process_type == "cpu_threaded":
            not_robustness = backend.py_not_threaded_numpy(subformula_robustness)
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
            and_robustness = backend.py_and_numpy(left_subformula_robustness,right_subformula_robustness)
        elif self.process_type == "cpu_threaded":
            and_robustness = backend.py_and_threaded_numpy(left_subformula_robustness,right_subformula_robustness)
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
            or_robustness = backend.py_or_numpy(left_subformula_robustness,right_subformula_robustness)
        elif self.process_type == "cpu_threaded":
            or_robustness = backend.py_or_threaded_numpy(left_subformula_robustness,right_subformula_robustness)
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
            until_robustness = until_robustness = backend.py_until_numpy(self.lower_time_bound,self.upper_time_bound,left_subformula_robustness,right_subformula_robustness,time_stamps)
        elif self.process_type == "cpu_threaded":
            until_robustness = until_robustness = backend.py_until_threaded_numpy(self.lower_time_bound,self.upper_time_bound,left_subformula_robustness,right_subformula_robustness,time_stamps)
        else:
            until_robustness = gpubackend.py_until_numpy_gpu(self.lower_time_bound,self.upper_time_bound,left_subformula_robustness,right_subformula_robustness,list(time_stamps))
        self.robustness = until_robustness[0]
        return until_robustness

