


import numpy as np
import isovist
from my_rrt import *
#from methods import *
import planner


class Runner(object):
	def __init__(self, isovist=None):
		self.isovist = isovist
		#TODO: varify these locations
		self.locs = [
            [ 0.100, 1-0.900 ],
            [ 0.566, 1-0.854 ],
            [ 0.761, 1-0.665 ],
            [ 0.523, 1-0.604 ],
            [ 0.241, 1-0.660 ],
            [ 0.425, 1-0.591 ],
            [ 0.303, 1-0.429 ],
            [ 0.815, 1-0.402 ],
            [ 0.675, 1-0.075 ],
            [ 0.432, 1-0.098 ] ]
		rx1,ry1,rx2,ry2 = polygons_to_segments( load_polygons( "./paths.txt" ) )
		self.plan_path = lambda start_loc, goal_loc: planner.run_rrt_opt( start_loc, goal_loc, rx1,ry1,rx2,ry2 )
		self.time_limit = 200
		self.show = True


	def run(self, Q):
		#----------------------------------------------------------
		#				simplified enforcer model 	
		#----------------------------------------------------------


		#enf_start_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="enf_start" )
		#enf_goal_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="enf_goal" )
		enf_start_i = int(np.random.uniform(0, 10))
		enf_goal_i = int(np.random.uniform(0, 10))
		
		print("enf_start_i",enf_start_i)
		print("enf_goal_i",enf_goal_i)

		enf_start = np.atleast_2d( self.locs[enf_start_i] )
		enf_goal = np.atleast_2d( self.locs[enf_goal_i] )

		print("enf_start",enf_start)
		print("enf_goal", enf_goal)

		enf_plan = self.plan_path(enf_start, enf_goal)

		print(len(enf_plan))

		#t = Q.choice( p=1.0/30*np.ones((1,30)), name="t" )
		t = int(np.random.uniform(0, 30))

		enf_loc = np.atleast_2d(enf_plan[t])
		print("enf_loc", enf_loc)	

		#----------------- end of enforcer model ------------------	


		
		#----------------------------------------------------------
		#				runner (intruder) model 	
		#----------------------------------------------------------

		start_i = int(np.random.uniform(0, 10))
		goal_i = int(np.random.uniform(0, 10))

		start = np.atleast_2d( self.locs[start_i] )
		goal = np.atleast_2d( self.locs[goal_i] )

		my_plan = self.plan_path(start, goal)
		my_loc = np.atleast_2d(my_plan[t])

		# TODO: add detection random variable
		seen = 0 #call isovists
		detected = 0 # ~ flip(seen*.999 + (1-seen*.001)


		# uav_loc = uav_path[ uav_loc_on_route ]
  #       fv = direction( uav_loc,
  #                       uav_path[ np.mod(uav_loc_on_route-1,len(uav_path)) ] )
  #       intersections = self.isovist.GetIsovistIntersections( uav_loc, fv )




		#*************** PLOTTING [TESTING] CODE **************
		if self.show:
			fig = plt.figure(1)
			fig.clf()
			ax = fig.add_subplot(1, 1, 1)

			
			scale = 1


		    # plot enf_plan
			path = enf_plan
			for i in range( 0, len(path)-1 ):
				ax.plot( [ path[i][0] * scale, path[i+1][0] * scale ], [ path[i][1] * scale, path[i+1][1] * scale], 'grey' )
				ax.scatter( path[i][0] * scale, path[i][1]  * scale, color="grey", s = 3)

			ax.scatter( enf_start[0,0] * scale, enf_start[0,1]  * scale, color="green")
			ax.scatter( enf_goal[0,0] * scale, enf_goal[0,1] * scale, color = "red")
			ax.scatter( enf_loc[0,0] * scale, enf_loc[0,1] * scale, color = "blue", s = 25)

			# plot intruder_plan
			path = my_plan
			for i in range( 0, len(path)-1 ):
				ax.plot( [ path[i][0] * scale, path[i+1][0] * scale ], [ path[i][1] * scale, path[i+1][1] * scale], 'grey' )
				ax.scatter( path[i][0] * scale, path[i][1]  * scale, color="grey", s = 3)

			ax.scatter( start[0,0] * scale, start[0,1]  * scale, color="green")
			ax.scatter( goal[0,0] * scale, goal[0,1] * scale, color = "red")
			ax.scatter( my_loc[0,0] * scale, my_loc[0,1] * scale, color = "magenta", s = 25)



			# plot all of the destinations
			# for i in xrange(10):
			# 	ax.scatter( np.atleast_2d( self.locs[i] )[0,0] * scale, np.atleast_2d( self.locs[i] )[0,1]  * scale, color="red")

			# plot map
			x1,y1,x2,y2 = polygons_to_segments( load_polygons( "./paths.txt" ) )
			for i in xrange(x1.shape[0]):
				ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'black' )

			plt.ylim((0,scale))
			plt.show()

		#********************************************************




if __name__ == '__main__':
		R = Runner()
		R.run(None)