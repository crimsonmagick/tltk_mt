import sys
import MTL as MTL 
import numpy as np

mode = 'cpu_threaded'

Ar1 = np.array([[1, 0],[0, 1]],dtype=np.float64)
br1 = np.array([160, 4500],dtype=np.float64)

r1 = MTL.Predicate(['speed','rpm'],Ar1,br1)

root = MTL.Global(0,100,r1)
        

