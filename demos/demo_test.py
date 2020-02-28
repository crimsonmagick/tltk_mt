import sys

sys.path.insert(1, '../robustness')

import os
import quadprog_polyhedron

import numpy as np

import time

 

list_range = (2**i for i in range(10,31))

mode = 'cpu'

i = 10

g = 1000

#Ap11 = np.array(np.random.rand(g,3),dtype=np.float64)

Ap11 = np.array([[1,0,0],[-1,0,0]], np.float64)

#Ap11 = np.array([[1, 0]],dtype=np.float64)
bp11 = np.array([250,-240],dtype=np.float64)

# bp11 =  np.random.rand(g)

# bp11 = np.ones(g)


Ap12 = np.random.rand(g,3)

bp12 =  np.random.rand(g)


traces = {}

data = [[-500,-500,-500]]*i #np.random.rand(1,3)

traces['data2'] = np.array([[1,1,1]]*i,dtype=np.float64) #np.random.rand(1,3)



time_stamps = np.arange(1, i + 1,dtype=np.float32)
print(Ap11)
print('---------------')
print(np.matmul(Ap11,data[0]))
print('---------------')
       
quadprog_polyhedron.solve_polyhedron_test(Ap11,bp11,data)
t0 = time.time()

del traces

del time_stamps
