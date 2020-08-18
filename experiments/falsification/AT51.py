import matlab.engine
import sys
sys.path.insert(1, '../../')
import numpy as np
import tltk
import tltk_mtl as MTL

model = 'models/sldemo_autotrans_mod01'
model_outputs = ['speed','rpm','gear']

step = 0.05
inp_range = [0, 100]
simulation_time = 30

opt = ['simulink', step, inp_range, simulation_time, model_outputs]

interpolation = 'pconst'

mode = 'cpu_threaded'

gL1 = MTL.Predicate('gear',1,1.5)    
gG1 = MTL.Predicate('gear',-1,-0.5)    

predicates = [gL1,gG1]

# cp_samples = np.random.uniform(low=inp_range[0], high=inp_range[1], size=(7,))
cp_bounds = [(0,100) for i in range(7)]

#root = MTL.Not(MTL.And(MTL.Finally(0,float('inf'),speed_pred), MTL.Finally(0,float('inf'),rpm_pred)))
#root = MTL.Not(MTL.Finally(0, 30, MTL.And(r1,r2)))

root = MTL.Global(0,30,MTL.Or(MTL.Not(MTL.And(MTL.Not(MTL.And(gL1,gG1)),MTL.Next(MTL.And(gL1,gG1)))),MTL.Next(MTL.Global(0,2.5,MTL.And(gL1,gG1)))))

# root = MTL.Global(0,30,MTL.Or(MTL.Not(MTL.And(MTL.Not(MTL.And(gL1,gG1),MTL.Next(MTL.And(gL1,gG1)))),MTL.Next(MTL.Global(0,30,MTL.And(gL1,gG1))))))

results = tltk.falsify(model, interpolation, cp_bounds, predicates, root, opt)
predicates

