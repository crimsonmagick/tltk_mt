import sys
sys.path.insert(1, '../robustness')
import MTL as MTL 
import numpy as np

mode = 'cpu_threaded'

Ar3 = -1
br3 = -160

Ar4 = 1
br4 = 4500

pred3 = MTL.Predicate('speed',Ar3,br3)
pred4 = MTL.Predicate('rpm',Ar4,br4)

root = MTL.Not(MTL.And(MTL.Finally(0,float('inf'),pred3,mode),MTL.Global(0,float('inf'),MTL.And(pred4, MTL.Global(0,float('inf'),MTL.Finally(0,float('inf'),MTL.And(pred3,MTL.Until(0,float('inf'),pred3,pred4,mode),mode),mode),mode),mode),mode),mode),mode)
 
