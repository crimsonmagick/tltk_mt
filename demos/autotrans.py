import matlab.engine
import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../robustness')

import numpy as np
import pytaliro as pytaliro
import robustness.MTL as MTL

model = 'sldemo_autotrans_mod01'

step = 0.05
inp_range = [0, 100]
simulation_time = 30

opt = ['file', step, inp_range, simulation_time]

#phi = '!(<>_[0,30]speed /\ <>_[0,30]rpm)'

#root = MTL.Finally(0,float('inf'),MTL.Or(MTL.Predicate('speed',[-1,0],-120),MTL.Predicate('rpm',[0,-1],-4500)))

# = [[-1,0],[0,-1]];
#bcomb = [-120,-4500]

#Acomb = [-1,0];
#bcomb = [-120];

#root = MTL.Global(0,float('inf'),MTL.Predicate('comb',Acomb,bcomb))

Aspeed = [1,0]
bspeed = [120]

Arpm = [0,1]
brpm = [4500]

interpolation = 'pchip'

mode = 'false'
#pred_tags = ['speed', 'rpm']
pred_tags = ['test']
test_pred = MTL.Predicate(pred_tags[0],[1,0],[150], mode)
#speed_pred = MTL.Predicate(pred_tags[0], Aspeed, bspeed, mode)
#rpm_pred = MTL.Predicate(pred_tags[1], Arpm, brpm, mode)

#root = MTL.Not(MTL.Or(MTL.Finally(0,float('inf'),speed_pred),MTL.Finally(0,float('inf'),rpm_pred)))
# vvvvvvvvvvvvvvvv THIS ONE WORKED
# root = MTL.Not(MTL.And(MTL.Finally(0,100,speed_pred),MTL.Finally(0,100,rpm_pred)))
#comb_pred =  MTL.Predicate(pred_tags[0], Acomb, bcomb, mode);
#predicates = [pred_tags, speed_pred, rpm_pred]

#predicates = [pred_tags, comb_pred]
cp_samples = np.random.uniform(low=inp_range[0], high=inp_range[1], size=(2,))
#predicates = [pred_tags, speed_pred, rpm_pred]
predicates = [pred_tags,test_pred]
#root = MTL.Not((MTL.Finally(0, 30, speed_pred, mode)))
root = MTL.Not(MTL.Finally(0,100,test_pred))
results = pytaliro.falsify(model, interpolation, cp_samples, predicates, root, opt)
