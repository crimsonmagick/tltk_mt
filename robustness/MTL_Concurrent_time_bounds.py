import sys
import cvxpy as cp
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
# trace[name] <= bound
# Time bounds inclusive
# 0 robustness is a failure (will add an option to choose later)
class Predicate:
    def __init__(self,variable_name,A_Matrix,bound,thread_pool = True):
        self.variable_name = variable_name
        self.value = None
        self.truth_value_history = []
        self.bound = bound
        self.robustness = 0
        self.A_Matrix = A_Matrix
        self.thread_pool = thread_pool
        
    def optimize_polyhedron(self,trace):
        np_value = np.array(trace)
        np_A_Matrix = np.array(self.A_Matrix)
        
        x = cp.Variable(np_value.size)
        objective = cp.Minimize(cp.norm(x - np_value))
        if np_value.size == 1 and np_A_Matrix.size == 1:
            return trace * self.A_Matrix - self.bound
        elif (self.A_Matrix*np_value <= self.bound).all():                        #calculate depth if np_value is in A*x <= b
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
        np_A_Matrix = np.array(self.A_Matrix)
        if self.thread_pool == False:
            for value in trace:
                np_value = np.array(value)
                x = cp.Variable(np_value.size)
                objective = cp.Minimize(cp.norm(x - np_value))
                if np_value.size == 1 and np_A_Matrix.size == 1:
                    return trace * self.A_Matrix - self.bound
                if (self.A_Matrix*np_value <= self.bound).all():            #calculate depth if np_value is in A*x <= b
                    constraints = [self.A_Matrix*x >= self.bound]
                    prob = cp.Problem(objective, constraints)
                    predicate_robustness.append(-prob.solve(solver=cp.ECOS)) #ECOS has fastest time of ones tested. Going to add this as an argument eventually
                else:                                               #calculate distance if np_value is not in A*x <= b
                    constraints = [self.A_Matrix*x <= self.bound]
                    prob = cp.Problem(objective, constraints)
                    predicate_robustness.append(prob.solve(solver=cp.ECOS))
        else:
                                                                    #I wanted to pass the Pool() as an argument but that weirdly causes an error. I then tryied making the thread pool in the object init but the same error happens. Not sure how to make the thread pool creation only happen once if the object is used more than once
            with Pool(cpu_count()) as p:
                predicate_robustness = p.map(self.optimize_polyhedron, trace)
                p.close()
                p.join()
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
        return globally_robustness
    
    def add_subformula(self,subformula):
        self.subformula = subformula
    def get_subformula(self):
        return self.subformula

class Finally:
    def __init__(self,lower_time_bound,upper_time_bound,subformula = None,thread_pool = True):
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

        if self.thread_pool == False or (self.lower_time_bound == 0 and self.upper_time_bound == float('inf')):
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
        else:
            with Pool(cpu_count()) as p:
                finally_robustness = p.map(self.calculate_sub_interval, range(len(time_stamps)))
                p.close()
                p.join()
                    
        self.robustness = max(finally_robustness)
        if self.robustness > 0:
            self.value = True
        finally_robustness.reverse()
        return  subformula_robustness
        
        
    def add_subformula(self,subformula):
        self.subformula = subformula
    def get_subformula(self):
        return self.subformula

class Not:
    def __init__(self,subformula = None):
        self.subformula = subformula
        self.truth_value_history = []
        self.robustness = 0
        self.value = None


    def eval_interval(self,traces,time_stamps): 
        subformula_robustness = self.subformula.eval_interval(traces,time_stamps)
        not_robustness = []

        for robustness in subformula_robustness:
            
            not_robustness.append(robustness * -1)
        
            if robustness > 0:
                self.value = True
            else:
                self.value = False
        self.robustness = -self.subformula.robustness 
                
        return not_robustness


    def add_subformula(self,subformula):
        self.subformula = subformula
    def get_subformula(self):
        return self.subformula

class And:
    def __init__(self,left_subformula = None,right_subformula = None):
        self.left_subformula = left_subformula
        self.right_subformula = right_subformula
        self.truth_value_history = []
        self.robustness = 0
        self.value = None


    def eval_interval(self,traces,time_stamps):
        left_subformula_robustness = self.left_subformula.eval_interval(traces,time_stamps)
        right_subformula_robustness = self.right_subformula.eval_interval(traces,time_stamps)
        and_robustness = []

        for left_robustness,right_robustness in zip(left_subformula_robustness,right_subformula_robustness):
            
            and_robustness.append(min(left_robustness,right_robustness))
        
            if left_robustness > 0 and right_robustness > 0:
                self.value = True
            else:
                self.value = False
        self.robustness = min(self.left_subformula.robustness,self.right_subformula.robustness)
        return and_robustness

class Or:
    def __init__(self,left_subformula = None,right_subformula = None):
        self.left_subformula = left_subformula
        self.right_subformula = right_subformula
        self.truth_value_history = []
        self.robustness = 0
        self.value = None

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

    def calculate_sub_interval(self, current_time_step):
        current_time_stamp = self.time_stamps[current_time_step]
        lower_bound = current_time_stamp + self.lower_time_bound
        upper_bound = current_time_stamp + self.upper_time_bound
        #lower_bound_index, upper_bound_index = self.binary_search(self.time_stamps,current_time_step,lower_bound,upper_bound)
        
        lower_bound_index, upper_bound_index = np.searchsorted(self.time_stamps[current_time_step:], [lower_bound, upper_bound])
        lower_bound_index = lower_bound_index + current_time_step
        upper_bound_index = upper_bound_index - 1 + current_time_step

        if lower_bound_index == current_time_step:
            min_robustness = left_subformula_robustness[lower_bound_index]
        else:
            min_robustness = min(left_subformula_robustness[current_time_step:lower_bound_index+1])
        for left_robustness_bounded ,right_robustness_bounded in zip(left_subformula_robustness[lower_bound_index:upper_bound_index+1],right_subformula_robustness[lower_bound_index:upper_bound_index+1]):
            last_robustness = max(last_robustness,min(right_robustness_bounded,min_robustness))
            min_robustness = min(min_robustness,left_robustness_bounded)
        until_robustness.insert(0,last_robustness)
        last_robustness = float('-inf')
    
    
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
                
                lower_bound_index, upper_bound_index = np.searchsorted(time_stamps[current_time_step:], [lower_bound, upper_bound])
                lower_bound_index = lower_bound_index + current_time_step
                upper_bound_index = upper_bound_index - 1 + current_time_step
                

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

