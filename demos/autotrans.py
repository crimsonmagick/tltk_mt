from falsify import falsify
import robustness.MTL_GPU as MTL

model = 'sldemo_autotrans_mod01'
step = 0.05
inpRange = [0, 100]
simulationTime = 30

phi = '!(<>_[0,30]speed /\ <>_[0,30]rpm)'

Aspeed = 1
bspeed = 120

Arpm = 1
brpm = 4500

mode = 'false'
speed_pred = MTL.Predicate('speed', Aspeed, bspeed, mode)
rpm_pred = MTL.Predicate('rpm', Arpm, brpm, mode)
root = MTL.Not((MTL.Finally(0, 30, speed_pred, mode)))

results = falsify(model, step, inpRange, simulationTime, root)
