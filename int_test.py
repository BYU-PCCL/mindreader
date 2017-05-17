
from intruder import BasicIntruder
import q
import isovist
from methods_slim import *

isovist = isovist.Isovist( load_polygons() )
X1, Y1, X2, Y2 = polygons_to_segments(load_polygons_here())

model = BasicIntruder( isovist )

pf = q.PF( model=model, cnt=100 )

data = [[ False, 1 ]] * 50

pf.init_particles()

for t in range( len(data) ):
    # data[t] should be a tuple of (seen,heard)
    # seen is boolean; heard is numeric

    # state-level conditioning.  the intruder location
    int_loc = [0.1,0.1]

    pf.step( obs_t=data[t], state_conds=int_loc )










