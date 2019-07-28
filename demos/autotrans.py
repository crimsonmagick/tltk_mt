from falsify import falsify
import robustness.MTL_GPU as MTL

model = 'sldemo_autotrans_mod01'
step = 0.05
inp_range = [0, 100]
simulation_time = 30

phi = '!(<>_[0,30]speed /\ <>_[0,30]rpm)'

Aspeed = 1
bspeed = 120

Arpm = 1
brpm = 4500

interpolation = 'pchip'

mode = 'false'
pred_tags = ['speed', 'rpm']
speed_pred = MTL.Predicate(pred_tags[0], Aspeed, bspeed, mode)
rpm_pred = MTL.Predicate(pred_tags[1], Arpm, brpm, mode)

root = MTL.Not((MTL.Finally(0, 30, speed_pred, mode)))

results = falsify(model, step, inp_range, simulation_time, interpolation, pred_tags, root)
