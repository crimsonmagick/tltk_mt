import sys
sys.path.insert(1, '../robustness')
import MTL as MTL 
import numpy as np

mode = 'cpu_threaded'


Ar7 = np.array([[2.5, -0.6],[0.1, -0.08],[-1.9, -0.4],[1.7, 0.8]],dtype=np.float64)
br7 = np.array([5.5,3.8,4.9,7.5],dtype=np.float64)

Ar8 = np.array([[2.5, -0.6],[0.1, -0.08],[-1.9, -0.4],[1.7, 0.8]],dtype=np.float64)
br8 = np.array([4.5,2.8,2.9,2.5],dtype=np.float64)

r7 = MTL.Predicate(['speed','rpm'],Ar7,br7)
r8 = MTL.Predicate(['speed','rpm'],Ar8,br8)

root = MTL.Global(0,float('inf'), MTL.And(r7,MTL.Finally(0,100,r8,mode),mode))
        

