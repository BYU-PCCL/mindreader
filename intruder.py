
import numpy as np
import isovist
from methods_slim import load_data, direction, noise_level
from rrt_smooth import *
import cPickle

'''
This is the intruder model.

  knows start and end
  has a belief over UAV locations and type (and UAV's beliefs?), now and in future => heatcube.
  crafts a plan that minimizes detection probability
    -> possibly balancing against path length
    -> planning as inference?

  intruder doesn't know where along the path the UAV is.

  Loops:
    gets an observation (possibly sound or sighting)
    must update beliefs about UAV locations
      -> particle filter?
    replans

  important: there are hidden variables here.
  -> therefore, result is a distribution over possible future locations
  -> if using some sort of optimal planner, then no distribution.

'''

types = [ 'alley', 'grid', 'random', 'swirl_in', 'swirl_out' ]

uav_path_types = []
for test_name in types:
    path = load_data( "NaiveAgentPaths/" + test_name + "_paths" )
    path = smooth( path )
    uav_path_types.append( path )

# ====================================================
#
# Intruder graphical model
#
# - The intruder will use this to do inference over the UAV location
# - and type
#
# ====================================================

class BasicIntruder( object ):
    def __init__( self, isovist=None ):
        self.isovist = isovist

        try:
            self.intersection_cache = cPickle.load( open("./int_cache.cp") )
        except:
            self.precompute_and_save_intersections()

    def precompute_and_save_intersections( self ):
        print "Precomputing intersections..."
        self.intersection_cache = []
        for tind in range(len(types)):
            print "  %s..." % types[tind]
            uav_path = uav_path_types[tind]
            tmp = []
            for uav_loc_on_route in range(len(uav_path)):
                uav_loc = uav_path[ uav_loc_on_route ]
                fv = direction( uav_loc,
                                uav_path[ np.mod(uav_loc_on_route-1,len(uav_path)) ] )
                intersections = self.isovist.GetIsovistIntersections( uav_loc, fv )
                tmp.append( intersections )
            self.intersection_cache.append( tmp )

        cPickle.dump( self.intersection_cache, open("./int_cache.cp","w") )

        print "done!"

    def condition( self, Q, obs_t ):
        Q.condition( name="seen_obs_t", value=obs_t[0] )
        Q.condition( name="heard_obs_t", value=obs_t[1] )

    def global_init( self, Q, glob_conds=None ):
        uav_type = Q.choice( p=np.asarray([1.0]*len(uav_path_types)), name="uav_type" )
        uav_path = uav_path_types[ uav_type ]
        return ( uav_type, uav_path )

    def state_init( self, Q, gs, state_conds=None ):
        uav_type, uav_path = gs
        uav_loc_on_route = Q.choice( p=np.asarray([1.0]*len(uav_path)), name="init_loc" )
        return uav_loc_on_route
    
    def trans( self, Q, gs, ts, glob_conds=None, state_conds=None ):
        # unpack 
        uav_type, uav_path = gs  # globals
        uav_loc_on_route = ts    # per-timestep
        int_loc = state_conds    # conditioned values

        uav_loc = uav_path[ uav_loc_on_route ]

        # sample next state
        new_uav_loc_on_route = np.mod( uav_loc_on_route + 1, len(uav_path) )
        return new_uav_loc_on_route

    def obs( self, Q, gs, ts, glob_conds=None, state_conds=None ):

        # gs is the global state object
        # ts is the current timestep state

        # unpack 
        uav_type, uav_path = gs  # globals
        uav_loc_on_route = ts    # per-timestep
        int_loc = state_conds    # conditioned values
        uav_loc = uav_path[ uav_loc_on_route ]

        # we now have the uav location and our location
        # must implement some sort of observation

        # XXX I think this is backwards - should be from the intruder's perspective!
        # argh... need previous intruder location!
#        fv = direction( uav_path[ uav_loc_on_route ],
#                        uav_path[ np.mod(uav_loc_on_route-1,len(uav_path)) ] )

#        intersections = self.isovist.GetIsovistIntersections( uav_loc, fv )
        intersections = self.intersection_cache[ uav_type ][ uav_loc_on_route ]

        intruder_seen = self.isovist.FindIntruderAtPoint( int_loc, intersections )
        pval = 0.999*intruder_seen + 0.001*(1-intruder_seen)
#        seen = Q.flip( p=pval, name="seen_obs_t" )
        seen = False

        noise = noise_level( int_loc, uav_loc )
        heard = Q.randn( mu=noise, sigma=1.0, name="heard_obs_t" )

        return seen, heard

