import matlab.engine
import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../robustness')
import time
import numpy as np
import pytaliro as pytaliro
import robustness.MTL as MTL

model = 'sldemo_autotrans_mod01'

step = 0.05
inp_range = [0, 100]
simulation_time = 30

opt = ['function', step, inp_range, simulation_time]

#phi = '!(<>_[0,30]speed /\ <>_[0,30]rpm)'

#root = MTL.Finally(0,float('inf'),MTL.Or(MTL.Predicate('speed',[-1,0],-120),MTL.Predicate('rpm',[0,-1],-4500)))

# = [[-1,0],[0,-1]];
#bcomb = [-120,-4500]

#Acomb = [-1,0];
#bcomb = [-120];

#root = MTL.Global(0,float('inf'),MTL.Predicate('comb',Acomb,bcomb))

A1 = -1
b1 = 2

A2 = 1
b2 = 2

interpolation = 'pchip'

mode = 'cpu_threaded'

pred_tags = ['p1', 'p2']

p1_pred = MTL.Predicate(pred_tags[0], A1, b1, mode)
p2_pred = MTL.Predicate(pred_tags[1], A2, b2, mode)

root = MTL.Global(0, float('inf'),MTL.Finally(0, 6.28,(MTL.And(p2_pred, MTL.Finally(0, 3.14, p1_pred, mode))),mode),mode)

cp_samples = np.random.uniform(low=inp_range[0], high=inp_range[1], size=(2,))

predicates = [pred_tags, p1_pred, p2_pred]

time_stamps = np.linspace(-np.pi, np.pi, 1000000)
pred_traces = time_stamps + 0.5*np.sin(2*time_stamps)
traces = {'p1':pred_traces, 'p2':pred_traces}

t0 = time.time()
root.eval_interval(traces, time_stamps)
t1 = time.time()

print('Run time: ', t1 - t0)
print('Robustness: ', root.robustness)
