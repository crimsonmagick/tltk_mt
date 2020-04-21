import sys
#sys.path.insert(1, '../robustness')
import MTL as MTL 
import numpy as np

mode = 'cpu_threaded'

Ar3 = -1
br3 = -160

Ar4 = 1
br4 = 4500

pred3 = MTL.Predicate('speed',Ar3,br3)
pred4 = MTL.Predicate('rpm',Ar4,br4)

root = MTL.Not(MTL.Finally(0,100,MTL.And(pred3,pred4,mode),mode),mode)
