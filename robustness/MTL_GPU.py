import sys
import cvxpy as cp
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
from numba import vectorize, guvectorize
#from quadprog import solve_qp
# trace[name] <= bound
# Time bounds inclusive
# 0 robustness is a failure (will add an option to choose later)



class Predicate:
    def __init__(self,variable_name,A_Matrix,bound,thread_pool = 'cpu'):
        self.variable_name = variable_name
        self.value = None
        self.truth_value_history = []
        self.bound = bound
        self.robustness = 0
        self.A_Matrix = A_Matrix
        self.thread_pool = thread_pool
       
    def optimize_polyhedron(self,trace):
        if trace.size == 1 and type(self.A_Matrix) is int:
            return trace * self.A_Matrix - self.bound
        else:
            np_value = np.array(trace)
            x = cp.Variable(np_value.size)
            objective = cp.Minimize(cp.norm(x - np_value))
            if (self.A_Matrix*np_value <= self.bound).all():                        #calculate depth if np_value is in A*x <= b
                constraints = [self.A_Matrix*x >= self.bound]
                prob = cp.Problem(objective, constraints)
                return -prob.solve(solver=cp.ECOS)    #ECOS has fastest time of ones tested. Going to add this as an argument eventually
            else:                                                           #calculate distance if np_value is not in A*x <= b
                constraints = [self.A_Matrix*x <= self.bound]
                prob = cp.Problem(objective, constraints)
                return prob.solve(solver=cp.ECOS)
    
        
    def eval_interval(self,traces,time_stamps):
        trace = traces[self.variable_name]
        predicate_robustness = []
        if self.thread_pool == 'false':
            for value in trace:
                np_value = np.array(value)
                x = cp.Variable(np_value.size)
                objective = cp.Minimize(cp.norm(x - np_value))
                if (self.A_Matrix*np_value <= self.bound).all():            #calculate depth if np_value is in A*x <= b
                    constraints = [self.A_Matrix*x >= self.bound]
                    prob = cp.Problem(objective, constraints)
                    predicate_robustness.append(-prob.solve(solver=cp.ECOS)) #ECOS has fastest time of ones tested. Going to add this as an argument eventually
                else:                                               #calculate distance if np_value is not in A*x <= b
                    constraints = [self.A_Matrix*x <= self.bound]
                    prob = cp.Problem(objective, constraints)
                    predicate_robustness.append(prob.solve(solver=cp.ECOS))
        elif self.thread_pool == 'cpu':                                                                    #I wanted to pass the Pool() as an argument but that weirdly causes and error 
                p = Pool(cpu_count()) 
                predicate_robustness = p.map(self.optimize_polyhedron, trace)
                p.close()
        elif self.thread_pool == 'gpu':
            #gpu_A_Matrix = np.float32(self.A_Matrix)
            #gpu_bound = np.float32(self.bound)
            gpu_A_Matrix = np.float32(self.A_Matrix)
            gpu_bound = np.float32(self.bound)
            @vectorize(['float32(float32,float32,float32)'], target='cuda')
            def optimize_polyhedron_gpu(gpu_A_Matrix,gpu_bound,trace):
                return trace * gpu_A_Matrix - gpu_bound
            
            predicate_robustness = optimize_polyhedron_gpu(gpu_A_Matrix,gpu_bound,np.float32(trace))
        return predicate_robustness
        


class Global:
    def __init__(self,lower_time_bound,upper_time_bound,subformula = None,thread_pool = True):
        self.value = True
        self.subformula = subformula
        self.truth_value_history = []
        self.robustness = float('inf')
        self.upper_time_bound = upper_time_bound
        self.lower_time_bound = lower_time_bound
        self.time_stamps = None
        self.subformula_robustness = None
        self.thread_pool = thread_pool
        
    def calculate_sub_interval(self, current_time_step):
        current_time_stamp = self.time_stamps[current_time_step]
        lower_bound = current_time_stamp + self.lower_time_bound
        upper_bound = current_time_stamp + self.upper_time_bound
        #lower_bound_index, upper_bound_index = self.binary_search(self.time_stamps,current_time_step,lower_bound,upper_bound)
        
        lower_bound_index, upper_bound_index = np.searchsorted(self.time_stamps[current_time_step:], [lower_bound, upper_bound])
        lower_bound_index = lower_bound_index + current_time_step
        upper_bound_index = upper_bound_index - 1 + current_time_step

        if lower_bound_index == None:
            return self.subformula_robustness[-1]
        elif lower_bound_index == upper_bound_index:
            return self.subformula_robustness[lower_bound_index]
        else:
            return min(self.subformula_robustness[lower_bound_index:upper_bound_index+1])


    def eval_interval(self,traces,time_stamps): 
        subformula_robustness = self.subformula.eval_interval(traces,time_stamps)
        self.subformula_robustness = subformula_robustness #Clean this up to be an object field only later
        self.time_stamps = time_stamps
        globally_robustness = []
        min_robustness = float('inf')
        #subformula_robustness.reverse()
        #time_stamps.reverse()
        if self.thread_pool == False or (self.lower_time_bound == 0 and self.upper_time_bound == float('inf')):
            for current_time_step,robustness in reversed(list(enumerate(subformula_robustness))):
                if self.lower_time_bound == 0 and self.upper_time_bound == float('inf'):
                    globally_robustness.append(min(min_robustness,robustness))
                    if min_robustness > robustness :
                            min_robustness = robustness
                else:
                    current_time_stamp = time_stamps[current_time_step] 
                    lower_bound = current_time_stamp + self.lower_time_bound
                    upper_bound = current_time_stamp + self.upper_time_bound
                    
                    lower_bound_index, upper_bound_index = np.searchsorted(time_stamps[current_time_step:], [lower_bound, upper_bound])
                    lower_bound_index = lower_bound_index + current_time_step
                    upper_bound_index = upper_bound_index - 1 + current_time_step

                    if lower_bound_index == None:
                        globally_robustness.append(subformula_robustness[-1])
                    elif lower_bound_index == upper_bound_index:
                        globally_robustness.append(subformula_robustness[lower_bound_index])
                    else:
                        globally_robustness.append(min(subformula_robustness[lower_bound_index:upper_bound_index+1]))
        else:
            with Pool(cpu_count()) as p:
                globally_robustness = p.map(self.calculate_sub_interval, range(len(time_stamps)))
                p.close()
                p.join()

        self.robustness = min(globally_robustness)
        if self.robustness > 0:
            self.value = True
        globally_robustness.reverse()
        #print(globally_robustness[:5])
        return globally_robustness
    
    def add_subformula(self,subformula):
        self.subformula = subformula
    def get_subformula(self):
        return self.subformula

class Finally:
    def __init__(self,lower_time_bound,upper_time_bound,subformula = None,thread_pool = 'cpu'):
        self.value = None
        self.subformula = subformula
        self.truth_value_history = []
        self.robustness = float('-inf')
        self.upper_time_bound = upper_time_bound
        self.lower_time_bound = lower_time_bound
        self.subformula_robustness = None
        self.time_stamps = None
        self.thread_pool = thread_pool
        
    def calculate_sub_interval(self, current_time_step):
        current_time_stamp = self.time_stamps[current_time_step]
        lower_bound = current_time_stamp + self.lower_time_bound
        upper_bound = current_time_stamp + self.upper_time_bound
        #lower_bound_index, upper_bound_index = self.binary_search(self.time_stamps,current_time_step,lower_bound,upper_bound)
        
        lower_bound_index, upper_bound_index = np.searchsorted(self.time_stamps[current_time_step:], [lower_bound, upper_bound])
        lower_bound_index = lower_bound_index + current_time_step
        upper_bound_index = upper_bound_index - 1 + current_time_step
        if lower_bound_index == None:
            return self.subformula_robustness[-1]
        elif lower_bound_index == upper_bound_index:
            return self.subformula_robustness[lower_bound_index]
        else:
            return max(self.subformula_robustness[lower_bound_index:upper_bound_index+1])
        
        
    def eval_interval(self,traces,time_stamps): 
        subformula_robustness = self.subformula.eval_interval(traces,time_stamps)
        self.subformula_robustness = subformula_robustness #Clean this up to be an object field only later
        self.time_stamps = time_stamps
        finally_robustness = []
        max_robustness = float('-inf')

        if (self.thread_pool != 'cpu' and self.thread_pool != 'gpu') or (self.lower_time_bound == 0 and self.upper_time_bound == float('inf')):
            for current_time_step,robustness in reversed(list(enumerate(subformula_robustness))):
                current_time_stamp = time_stamps[current_time_step] 
                lower_bound = current_time_stamp + self.lower_time_bound
                upper_bound = current_time_stamp + self.upper_time_bound
                lower_bound_index = None
                if self.lower_time_bound == 0 and self.upper_time_bound == float('inf'):
                    finally_robustness.append(max(max_robustness,robustness))
                    if robustness > max_robustness:
                        max_robustness = robustness
                else:
                    lower_bound_index, upper_bound_index = np.searchsorted(self.time_stamps[current_time_step:], [lower_bound, upper_bound])
                    lower_bound_index = lower_bound_index + current_time_step
                    upper_bound_index = upper_bound_index - 1 + current_time_step

                    if lower_bound_index == None:
                        finally_robustness.append(subformula_robustness[-1])
                    elif lower_bound_index == upper_bound_index:
                        finally_robustness.append(subformula_robustness[lower_bound_index])
                    else:
                        finally_robustness.append(max(subformula_robustness[lower_bound_index:upper_bound_index+1]))
        
        elif self.thread_pool == 'cpu':
            with Pool(cpu_count()) as p:
                finally_robustness = p.map(self.calculate_sub_interval, range(len(time_stamps)))
                p.close()
                p.join()
        elif self.thread_pool == 'gpu':
            np_lower_time_bound = np.float32(self.lower_time_bound)
            np_upper_time_bound = np.float32(self.upper_time_bound)
            np_time_stamps = np.float32(time_stamps)
            np_subformula_robustness = np.float32(subformula_robustness)
            # def calculate_sub_interval_gpu(lower_bound_time, upper_bound_time,time_stamps, subformula_robustness):
            @guvectorize(['(float32,float32,float32[:],float32[:],float32[:])'],'(),(),(n),(n) -> (n)',target='cuda')
            def calculate_sub_interval_gpu(lower_time_bound, upper_time_bound,time_stamps, subformula_robustness,finally_robustness):
                for current_time_step in range(time_stamps.shape[0]):
                    current_time_stamp = time_stamps[current_time_step] 
                    lower_bound = current_time_stamp + lower_time_bound
                    upper_bound = current_time_stamp + upper_time_bound
                 
                    # lower_bound_index = np.searchsorted(time_stamps[current_time_step:], lower_bound)
                    # upper_bound_index = np.searchsorted(time_stamps[current_time_step:], upper_bound)
                    # lower_bound_index = lower_bound_index + current_time_step
                    # upper_bound_index = upper_bound_index - 1 + current_time_step
                    
                    if lower_time_bound == 0:
                        lower_bound_index = current_time_step
                    else:
                        for time_stamp_index  in range(time_stamps.shape[0]):          #this is very bad and is just a place holder
                            if lower_bound <= time_stamps[time_stamp_index]:
                                lower_bound_index = time_stamp_index
                                break
            
                    for time_stamp_index  in range(time_stamps.shape[0]):          #this is very bad and is just a place holder
                        if time_stamps[time_stamp_index] <= upper_bound:
                            upper_bound_index = time_stamp_index
                            break
                    #lower_search_index = np.int64(0)
                    # upper_search_index = finally_robustness.shape[0] - 1
                    # while True:
                        # search_index = np.int64((lower_search_index + upper_search_index)/2)
                        
                        # if upper_bound <= finally_robustness[search_index]:
                            # if search_index - 1 >= 0:
                                # if finally_robustness[search_index - 1] < upper_bound:
                                    # upper_bound_index = search_index
                            # else:
                                # upper_search_index = search_index
                        # else:
                            # lower_search_index = search_index
                             
                             
                    if lower_bound_index == upper_bound_index:
                        finally_robustness[current_time_step] = subformula_robustness[lower_bound_index]
                    else:
                        max_robustness = subformula_robustness[lower_bound_index]
                        for i in range(lower_bound_index,upper_bound_index+1):
                            if max_robustness < finally_robustness[i]:
                                finally_robustness[current_time_step] = max_robustness
                    
            
            finally_robustness = calculate_sub_interval_gpu(np_lower_time_bound, np_upper_time_bound,np_time_stamps, np_subformula_robustness)
                    
        self.robustness = max(finally_robustness)
        if self.robustness > 0:
            self.value = True
            
        finally_robustness =  list(finally_robustness)
        finally_robustness.reverse()
        #print(finally_robustness[:5])
        #print(finally_robustness)
        return  finally_robustness
        
        
    def add_subformula(self,subformula):
        self.subformula = subformula
    def get_subformula(self):
        return self.subformula

class Not:
    def __init__(self,subformula = None,thread_pool = 'false'):
        self.subformula = subformula
        self.truth_value_history = []
        self.robustness = 0
        self.value = None
        self.thread_pool = thread_pool

    def eval_interval(self,traces,time_stamps): 
        subformula_robustness = self.subformula.eval_interval(traces,time_stamps)
        not_robustness = []
        if self.thread_pool == 'false':
            for robustness in subformula_robustness:
                not_robustness.append(robustness * -1)
        elif self.thread_pool == 'gpu':
            @vectorize(['float32(float32)'], target='cuda')
            def not_gpu(robustness):
                return -1*robustness
            not_robustness = not_gpu(np.float32(subformula_robustness))
        self.robustness = not_robustness[0]

                
        return not_robustness


    def add_subformula(self,subformula):
        self.subformula = subformula
    def get_subformula(self):
        return self.subformula

class And:
    def __init__(self,left_subformula = None,right_subformula = None,thread_pool = 'false'):
        self.left_subformula = left_subformula
        self.right_subformula = right_subformula
        self.truth_value_history = []
        self.robustness = 0
        self.value = None
        self.thread_pool = thread_pool

    def eval_interval(self,traces,time_stamps):
        left_subformula_robustness = self.left_subformula.eval_interval(traces,time_stamps)
        right_subformula_robustness = self.right_subformula.eval_interval(traces,time_stamps)
        and_robustness = []
        if self.thread_pool == 'false':
            for left_robustness,right_robustness in zip(left_subformula_robustness,right_subformula_robustness):
                
                and_robustness.append(min(left_robustness,right_robustness))
            
                if left_robustness > 0 and right_robustness > 0:
                    self.value = True
                else:
                    self.value = False
        elif self.thread_pool == 'gpu':
            @vectorize(['float32(float32,float32)'], target='cuda')
            def and_gpu(left_robustness,right_robustness):
                return min(left_robustness,right_robustness)
            and_robustness = and_gpu(np.float32(left_subformula_robustness),np.float32(right_subformula_robustness))
        self.robustness = min(self.left_subformula.robustness,self.right_subformula.robustness)
        return and_robustness

class Or:
    def __init__(self,left_subformula = None,right_subformula = None):
        self.left_subformula = left_subformula
        self.right_subformula = right_subformula
        self.truth_value_history = []
        self.robustness = 0
        self.value = None

    def eval(self,trace,time_stamp):
        if self.left_subformula == None or self.right_subformula == None: 
            print('Each branch of MTL tree needs to be terminated with a proposition',file=sys.stderr)
            sys.exit()
        left_robustness = self.left_subformula.eval(trace,time_stamp)
        right_robustness = self.right_subformula.eval(trace,time_stamp)

        if left_robustness > 0 or right_robustness > 0:
            self.value = True
        else:
            self.value = False
        self.truth_value_history.append(self.value)

        if left_robustness > right_robustness:
            self.robustness = left_robustness
        else:
            self.robustness = right_robustness
            
        return self.robustness

    def eval_interval(self,traces,time_stamps): 
        left_subformula_robustness = self.left_subformula.eval_interval(traces,time_stamps)
        right_subformula_robustness = self.right_subformula.eval_interval(traces,time_stamps)
        or_robustness = []

        for left_robustness,right_robustness in zip(left_subformula_robustness,right_subformula_robustness):
            
            or_robustness.append(max(left_robustness,right_robustness))
        
            if left_robustness > 0 or right_robustness > 0:
                self.value = True
            else:
                self.value = False
        self.robustness = max(self.left_subformula.robustness,self.right_subformula.robustness)
        return or_robustness

class Implication:
    def __init__(self,left_subformula = None,right_subformula = None):
        self.left_subformula = left_subformula
        self.right_subformula = right_subformula
        self.truth_value_history = []
        self.robustness = 0
        self.value = None

    def eval(self,trace,time_stamp):
        if self.left_subformula == None or self.right_subformula == None: 
            print('Each branch of MTL tree needs to be terminated with a proposition',file=sys.stderr)
            sys.exit()
        #could use thread pool if more expensive distance operation are added
        left_robustness = self.left_subformula.eval(trace,time_stamp)
        right_robustness = self.right_subformula.eval(trace,time_stamp)
        
        left_robustness *= -1           #apply not to the left side of the implies
        
        if left_robustness > 0 or right_robustness > 0:
            self.value = True
        else:
            self.value = False
        self.truth_value_history.append(self.value)
        
        if left_robustness > right_robustness:
            self.robustness = left_robustness
        else:
            self.robustness = right_robustness
            
        return self.robustness

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
    def __init__(self,lower_time_bound,upper_time_bound,left_subformula = None,right_subformula = None):
        self.left_subformula = left_subformula
        self.right_subformula = right_subformula
        self.upper_time_bound = upper_time_bound
        self.lower_time_bound = lower_time_bound
        self.truth_value_history = []
        self.robustness = 0
        self.value = True
        self.right_subformula_true = False

    def eval(self, trace, time_stamp):
        if self.left_subformula == None or self.right_subformula == None:
            print('Each branch of MTL tree needs to be terminated with a proposition',file=sys.stderr)
            sys.exit()

        left_robustness = self.left_subformula.eval(trace,time_stamp)
        right_robustness = self.right_subformula.eval(trace,time_stamp)

        if self.right_subformula_true and left_robustness > 0:
            self.value = True
        else:
            self.value = False
            self.right_subformula_true = False

        if right_robustness > 0:
            self.right_subformula_true = True

        if self.value:
            return 1
        else:
            return -1

    def eval_interval(self,traces,time_stamps): 
        left_subformula_robustness = self.left_subformula.eval_interval(traces,time_stamps)
        right_subformula_robustness = self.right_subformula.eval_interval(traces,time_stamps)
        until_robustness = []
        left_subformula_robustness_history = []
        inner_formula_min = []
        last_robustness = float('-inf')

        if self.lower_time_bound == 0 and self.upper_time_bound == float('inf'):
            for left_robustness,right_robustness in reversed(list(zip(left_subformula_robustness,right_subformula_robustness))):
                last_robustness = max(min(last_robustness,left_robustness),right_robustness)
                until_robustness.insert(0,last_robustness)
            self.robustness = until_robustness[0]
        else:
            for current_time_step,data in reversed(list((enumerate(zip(left_subformula_robustness,time_stamps))))):
            #for current_index,left_robustness in enumerate(list(reversed(list(zip(left_subformula_robustness,right_subformula_robustness,time_stamps))))):
                
                left_robustness,current_time_stamp = data

                current_time_stamp = time_stamps[current_time_step] 
                lower_bound = current_time_stamp + self.lower_time_bound
                upper_bound = current_time_stamp + self.upper_time_bound
                lower_bound_index = None
                
                for time_stamp_index ,time_stamp in enumerate(time_stamps):          #this is very bad and is just a place holder
                    if lower_bound <= time_stamp:
                        lower_bound_index = time_stamp_index
                        break
                
                for time_stamp_index ,time_stamp in reversed(list(enumerate(time_stamps))):          #this is very bad and is just a place holder
                    if time_stamp <= upper_bound:
                        upper_bound_index = time_stamp_index
                        break

                if lower_bound_index == current_time_step:
                    min_robustness = left_subformula_robustness[lower_bound_index]
                else:
                    min_robustness = min(left_subformula_robustness[current_time_step:lower_bound_index+1])
                    
                
                for left_robustness_bounded ,right_robustness_bounded in zip(left_subformula_robustness[lower_bound_index:upper_bound_index+1],right_subformula_robustness[lower_bound_index:upper_bound_index+1]):
                    last_robustness = max(last_robustness,min(right_robustness_bounded,min_robustness))
                    min_robustness = min(min_robustness,left_robustness_bounded)
                until_robustness.insert(0,last_robustness)
                last_robustness = float('-inf')

        self.robustness = until_robustness[0]
        return until_robustness

