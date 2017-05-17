
from intruder import BasicIntruder
import q

model = BasicIntruder()

pf = q.PF( model, cnt=100 )

data = load_data()

pf.init_particles()

for t in range( len(data) ):
    # data[t] should be a tuple of (seen,heard)
    # seen is boolean; heard is numeric
    pf.step( obs_t=data[t] )










