import matlab.engine
import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../robustness')

import numpy as np
import tltk as tltk
import robustness.MTL as MTL

model = []

step = 0.05
inp_range = [0, 100]
simulation_time = 30

opt = ['function', step, inp_range, simulation_time]

interpolation = 'pchip'

mode = 'gpu'

pred_tags = ['test']
test_pred = MTL.Predicate(pred_tags[0],1,150, mode)

cp_samples = np.random.uniform(low=inp_range[0], high=inp_range[1], size=(2,))

predicates = [pred_tags,test_pred]

root = MTL.Not(MTL.Finally(0,100,test_pred))
results = tltk.falsify(model, interpolation, cp_samples, predicates, root, opt)
