


import numpy as np
import isovist
from my_rrt import *
from methods import load_data, direction, load_isovist_map, scale_up
import planner


class Runner(object):
	def __init__(self, isovist=None, locs=None, seg_map=[None,None,None,None]):
		self.isovist = isovist
		#TODO: varify these locations
		self.locs = locs
		rx1,ry1,rx2,ry2 = seg_map
		#rx1,ry1,rx2,ry2 = polygons_to_segments( load_polygons( "./paths.txt" ) )
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
		while enf_goal_i == enf_start_i:
			enf_goal_i = int(np.random.uniform(0, 10))
		
		enf_start = np.atleast_2d( self.locs[enf_start_i] )
		enf_goal = np.atleast_2d( self.locs[enf_goal_i] )
		path_noise = .00001
		enf_plan = self.plan_path(enf_start, enf_goal)
		enf_noisy_plan = [enf_plan[0]]
		for i in xrange(1, len(enf_plan)-1):
			# add noise to each location at 't_i'
			loc_t = np.random.multivariate_normal(enf_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			enf_noisy_plan.append(loc_t)
		enf_noisy_plan.append(enf_plan[-1])





		#t = Q.choice( p=1.0/30*np.ones((1,30)), name="t" )
		#t = int(np.random.uniform(0, 30))
		#for display only - to show in the middle-ish
		t = int(np.random.uniform(3, 10))

		enf_loc = np.atleast_2d(enf_plan[t])

		#----------------- end of enforcer model ------------------	


		
		#----------------------------------------------------------
		#				runner (intruder) model 	
		#----------------------------------------------------------

		start_i = int(np.random.uniform(0, 10))
		goal_i = int(np.random.uniform(0, 10))
		while goal_i == start_i:
			goal_i = int(np.random.uniform(0, 10))

		start = np.atleast_2d( self.locs[start_i] )
		goal = np.atleast_2d( self.locs[goal_i] )

		my_plan = self.plan_path(start, goal)
		my_loc = np.atleast_2d(my_plan[t])
		my_noisy_plan = [my_plan[0]]
		for i in xrange(1, len(my_plan)-1):
			# add noise to each location at 't_i'
			loc_t = np.random.multivariate_normal(my_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			my_noisy_plan.append(loc_t)
		my_noisy_plan.append(my_plan[-1])

		# set up the view (vision) of the enforcer
		pre_enf_loc = scale_up(enf_plan[np.mod(t-1, len(enf_plan))])
		cur_enf_loc = scale_up(enf_plan[t])
		fv = direction(cur_enf_loc, pre_enf_loc)
		intersections = self.isovist.GetIsovistIntersections(cur_enf_loc, fv)

		# does the enforcer see me at time 't'
		my_cur_loc = scale_up(my_plan[t])
		was_i_seen = self.isovist.FindIntruderAtPoint( my_cur_loc, intersections )
		detected_prob = 0.999*was_i_seen + 0.001*(1-was_i_seen) # ~ flip(seen*.999 + (1-seen*.001)
		#detected = Q.flip( p=detected_prob, name="was_intruder_detected" )
		print "INTRUDER DETECTED=", detected_prob

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
				ax.scatter( enf_noisy_plan[i][0] * scale, enf_noisy_plan[i][1]  * scale, color="black", s = 3)

			ax.scatter( enf_start[0,0] * scale, enf_start[0,1]  * scale, color="green")
			ax.scatter( enf_goal[0,0] * scale, enf_goal[0,1] * scale, color = "red")
			ax.scatter( enf_loc[0,0] * scale, enf_loc[0,1] * scale, color = "blue", s = 55, marker="v")

			# plot intruder_plan
			path = my_plan
			for i in range( 0, len(path)-1 ):
				ax.plot( [ path[i][0] * scale, path[i+1][0] * scale ], [ path[i][1] * scale, path[i+1][1] * scale], 'grey' )
				ax.scatter( my_noisy_plan[i][0] * scale, my_noisy_plan[i][1]  * scale, color="black", s = 3)

			ax.scatter( start[0,0] * scale, start[0,1]  * scale, color="green")
			ax.scatter( goal[0,0] * scale, goal[0,1] * scale, color = "red")
			ax.scatter( my_loc[0,0] * scale, my_loc[0,1] * scale, color = "magenta", s = 45, marker = "D")



			# plot all of the destinations
			# for i in xrange(10):
			# 	ax.scatter( np.atleast_2d( self.locs[i] )[0,0] * scale, np.atleast_2d( self.locs[i] )[0,1]  * scale, color="red")

			# plot map
			x1,y1,x2,y2 = polygons_to_segments( load_polygons( "./paths.txt" ) )
			for i in xrange(x1.shape[0]):
				ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'black' )

			plt.ylim((0,scale))
			plt.show()






if __name__ == '__main__':
		locs = [
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

		seg_map = polygons_to_segments( load_polygons( "./paths.txt" ) )

		# load isovist
		isovist = isovist.Isovist( load_isovist_map() )
		print(isovist)

		R = Runner(isovist, locs, seg_map)
		R.run(None)









