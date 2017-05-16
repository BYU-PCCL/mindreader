
import q

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
  -> if using some sort of optimizal planner, then no distribution.

'''

class Int_GS( object ):
    def __init__( self ):
        pass

class Int_TS( object ):
    def __init__( self ):
        pass

# ====================================================

class Intruder( Q_TS ):
    def __init__( self ):
        pass

    def global_init( start, end ):
        return Int_GS()
    
    def step( self, Q, gs, ts ):
        # gs is the global state object
        # ts is the current timestep state

        # 
        
        # must implement some sort of observation
        Q.cond( )

        return 
