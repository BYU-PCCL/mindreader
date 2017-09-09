


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
		self.show = False


	def run(self, Q):
		#----------------------------------------------------------
		#				simplified enforcer model 	
		#----------------------------------------------------------

		t = Q.choice( p=1.0/29*np.ones((1,29)), name="t" )

		cnt = len(self.locs)
		enf_start_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="enf_start" )
		enf_goal_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="enf_goal" )
		
		enf_start = np.atleast_2d( self.locs[enf_start_i] )
		enf_goal = np.atleast_2d( self.locs[enf_goal_i] )
		path_noise = .003
		enf_plan = self.plan_path(enf_start, enf_goal)
		enf_noisy_plan = []
		for i in xrange(1, len(enf_plan)-1): #loc_t = np.random.multivariate_normal(enf_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			loc_x = Q.randn( mu=enf_plan[i][0], sigma=path_noise, name="enf_x_"+str(i) )
			loc_y = Q.randn( mu=enf_plan[i][1], sigma=path_noise, name="enf_y_"+str(i) )
			loc_t = [loc_x, loc_y]
			enf_noisy_plan.append(loc_t)
			
		enf_noisy_plan.append(enf_plan[-1])

		
		enf_loc = np.atleast_2d(enf_plan[t])

		#----------------- end of enforcer model ------------------	


		# XXX the runner wants to do goal inference to figure out the next step of the enf
		# TWO desires:
		# 1) high likelihood for a enf path conditioned on past locations
		# 2) high likelihood for his own path conditioned of not being detected for the next step and conditioned
		# 	 on past enf locations !

		#----------------------------------------------------------
		#				runner (intruder) model 	
		#----------------------------------------------------------

		start_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="int_start" )
		goal_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="int_goal" )

		start = np.atleast_2d( self.locs[start_i] )
		goal = np.atleast_2d( self.locs[goal_i] )

		my_plan = self.plan_path(start, goal)
		my_loc = np.atleast_2d(my_plan[t])
		my_noisy_plan = [my_plan[0]]
		for i in xrange(1, len(my_plan)-1):#loc_t = np.random.multivariate_normal(my_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			loc_x = Q.randn( mu=my_plan[i][0], sigma=path_noise, name="int_x_"+str(i) )
			loc_y = Q.randn( mu=my_plan[i][1], sigma=path_noise, name="int_y_"+str(i) )
			loc_t = [loc_x, loc_y]
			my_noisy_plan.append(loc_t)
		my_noisy_plan.append(my_plan[-1])

		# XXX Need to make sure the runner wasn't seen in any of the previous time steps
		i_already_seen = 0
		for i in xrange(t):
			intersections = Q.cache["enf_intersections_t_"+str(i)]# get enforcer's fv for time t
			i_already_seen = self.isovist.FindIntruderAtPoint(my_noisy_plan[i], intersections)
			if (i_already_seen):
				detected_prob = 0.999*i_already_seen + 0.001*(1-i_already_seen) 

		if not i_already_seen:
			# set up the enforcer view (forward vector, fv) for the next step
			cur_enf_loc = scale_up(enf_noisy_plan[t])
			next_enf_loc = scale_up(enf_noisy_plan[t+1])
			fv = direction(next_enf_loc, cur_enf_loc)
			intersections = self.isovist.GetIsovistIntersections(next_enf_loc, fv)

			# does the enforcer see me at time 't'
			my_next_loc = scale_up(my_noisy_plan[t+1])
			will_i_be_seen = self.isovist.FindIntruderAtPoint( my_next_loc, intersections )
			detected_prob = 0.999*will_i_be_seen + 0.001*(1-will_i_be_seen) # ~ flip(seen*.999 + (1-seen*.001)
		

		future_detection = Q.flip( p=detected_prob, name="int_detected" )

		#print "INTRUDER DETECTED=", detected

		# for rendering purposes
		Q.keep("int_plan", my_noisy_plan)
		Q.keep("enf_plan", enf_noisy_plan)

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

			ax.scatter( path[0][0] * scale, path[0][1]  * scale, color="green")
			ax.scatter( path[-1][0] * scale, path[-1][1] * scale, color = "red")
			ax.scatter( path[t][0] * scale, path[t][1] * scale, color = "blue", s = 55, marker="v")

			# plot intruder_plan
			path = my_plan
			for i in range( 0, len(path)-1 ):
				ax.plot( [ path[i][0] * scale, path[i+1][0] * scale ], [ path[i][1] * scale, path[i+1][1] * scale], 'grey' )
				ax.scatter( my_noisy_plan[i][0] * scale, my_noisy_plan[i][1]  * scale, color="black", s = 3)

			ax.scatter( path[0][0] * scale, path[0][1]  * scale, color="green")
			ax.scatter( path[0][0] * scale, path[0][1] * scale, color = "red")
			ax.scatter( path[t][0] * scale, path[t][1] * scale, color = "magenta", s = 45, marker = "D")

			# plot all of the destinations
			# for i in xrange(10):
			# 	ax.scatter( np.atleast_2d( self.locs[i] )[0,0] * scale, np.atleast_2d( self.locs[i] )[0,1]  * scale, color="red")

			# plot map
			x1,y1,x2,y2 = polygons_to_segments( load_polygons( "./paths.txt" ) )
			for i in xrange(x1.shape[0]):
				ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'black' )

			plt.ylim((0,scale))
			plt.show()

		return t, my_plan, future_detection






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

		R = Runner(isovist, locs, seg_map)
		R.run(None)









