
from intruder import BasicIntruder
import q
import isovist
from methods import *
from rrt_smooth import *


# load isovist
isovist = isovist.Isovist( load_polygons() )

# load map/polygons
X1, Y1, X2, Y2 = polygons_to_segments(load_polygons_here())

# load all type paths
types = [ 'alley', 'grid', 'random', 'swirl_in', 'swirl_out' ]
uav_path_types = []
for test_name in types:
    path = load_data( "NaiveAgentPaths/" + test_name + "_paths" )
    path = smooth( path )
    uav_path_types.append( path )

# set number of particles
PART_CNT = 5

# set the model
model = BasicIntruder( isovist )

# instantiate the inference algorithm
pf = q.PF( model=model, cnt=PART_CNT )

# initialize particles with a particular uav path and location on the path
print "Initializing particles..."
pf.init_particles()

# --------------------------------------------------------------

# select uav type, path, and location on the path to try to infer with the particle filter
uav_type = np.random.choice( range(len(types)), p=[0.2,0.2,0.2,0.2,0.2] )
uav_path = uav_path_types[ uav_type ]
uav_loc_on_route = np.random.choice( range(len(uav_path)), p=[1.0/(1.0*len(uav_path))] * len(uav_path) )

print "type=%d, len=%d, start_loc=%d" % ( uav_type, len(uav_path), uav_loc_on_route )

print "Filtering..."

# state-level conditioning.  

# the intruder location
int_loc = [0.1,0.1]

observation_cnt = 1

for t in range( observation_cnt ):

    # get the uav's current location 
    uav_loc = uav_path[ uav_loc_on_route ]

    # use the uav's location to get the sound observation
    heard = noise_level( int_loc, uav_loc )

    # assume that we never see the agent, only hear it
    seen = False

    # tuple observations
    obs = ( seen, heard )

    # particles step 
    pf.step( obs_t=obs, state_conds=int_loc )

    # simulate the uav steping forward
    uav_loc_on_route = np.mod( uav_loc_on_route + 1, len(uav_path) )



type_hist = np.zeros((1,5))
for k in range( PART_CNT ):
    type_hist[ 0, pf.part_gs[k][0] ] += pf.part_score[k]

print type_hist



