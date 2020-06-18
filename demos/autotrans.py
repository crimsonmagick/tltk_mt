import matlab.engine
import sys
sys.path.insert(1, '../')

import numpy as np
import tltk
import tltk_mtl as MTL

model = 'data/sldemo_autotrans_mod01'

step = 0.05
inp_range = [0, 100]
simulation_time = 30

opt = ['simulink', step, inp_range, simulation_time]

interpolation = 'pchip'

mode = 'cpu_threaded'
pred_tags = ['speed', 'rpm']
# pred_tags = ['test']
speed_pred = MTL.Predicate('speed',-1,-120)
rpm_pred = MTL.Predicate('rpm',-1,-4500)

predicates = [speed_pred, rpm_pred]
cp_samples = np.random.uniform(low=inp_range[0], high=inp_range[1], size=(2,))
#predicates = [pred_tags, speed_pred, rpm_pred]
#root = MTL.Not((MTL.Finally(0, 30, speed_pred, mode)))
root = MTL.Not(MTL.And(MTL.Finally(0,float('inf'),speed_pred), MTL.Finally(0,float('inf'),rpm_pred)))
results = tltk.falsify(model, interpolation, cp_samples, predicates, root, opt)
