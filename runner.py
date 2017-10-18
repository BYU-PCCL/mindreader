


import numpy as np
import isovist
from my_rrt import *
from methods import load_data, direction, load_isovist_map, scale_up
import planner


class Runner(object):
	def __init__(self, isovist=None, locs=None, seg_map=[None,None,None,None]):
		self.isovist = isovist
		self.locs = locs
		rx1,ry1,rx2,ry2 = seg_map
		self.seg_map = seg_map
		#self.plan_path = lambda start_loc, goal_loc: planner.run_rrt_opt( start_loc, goal_loc, rx1,ry1,rx2,ry2 )
		self.time_limit = 200
		self.show = False


	def run(self, Q):
		#----------------------------------------------------------
		#				simplified enforcer model 	
		#----------------------------------------------------------

		t = Q.choice( p=1.0/40*np.ones((1,40)), name="t" )

		cnt = len(self.locs)
		enf_start_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="enf_start" )
		enf_goal_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="enf_goal" )
		
		enf_start = np.atleast_2d( self.locs[enf_start_i] )
		enf_goal = np.atleast_2d( self.locs[enf_goal_i] )
		path_noise = .003
		#enf_plan = self.plan_path(enf_start, enf_goal)
		rx1,ry1,rx2,ry2 = self.seg_map
		enf_plan = planner.run_rrt_opt( np.atleast_2d(enf_start), 
		np.atleast_2d(enf_goal), rx1,ry1,rx2,ry2 )

		enf_noisy_plan = [enf_plan[0]]
		for i in xrange(1, len(enf_plan)-1): #loc_t = np.random.multivariate_normal(enf_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			loc_x = Q.randn( mu=enf_plan[i][0], sigma=path_noise, name="enf_x_"+str(i) )
			loc_y = Q.randn( mu=enf_plan[i][1], sigma=path_noise, name="enf_y_"+str(i) )
			loc_t = [loc_x, loc_y]
			enf_noisy_plan.append(loc_t)

		enf_noisy_plan.append(enf_plan[-1])
		enf_loc = np.atleast_2d(enf_noisy_plan[t])

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

		#my_plan = self.plan_path(start, goal)
		my_plan = planner.run_rrt_opt( np.atleast_2d(start), 
		np.atleast_2d(goal), rx1,ry1,rx2,ry2 )
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

		# for rendering purposes
		Q.keep("int_plan", my_noisy_plan)
		Q.keep("enf_plan", enf_noisy_plan)


# class Runner_Minus(object):
# 	def __init__(self, isovist=None, locs=None, seg_map=[None,None,None,None]):
# 		self.isovist = isovist
# 		self.locs = locs
# 		rx1,ry1,rx2,ry2 = seg_map
# 		self.seg_map = seg_map
# 		self.show = False


# 	def run(self, Q):
		
# 		#----------------------------------------------------------
# 		#				runner (minus) model 	
# 		#----------------------------------------------------------

# 		start_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="int_start" )
# 		goal_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="int_goal" )

# 		start = np.atleast_2d( self.locs[start_i] )
# 		goal = np.atleast_2d( self.locs[goal_i] )

# 		#my_plan = self.plan_path(start, goal)
# 		my_plan = planner.run_rrt_opt( np.atleast_2d(start), 
# 		np.atleast_2d(goal), rx1,ry1,rx2,ry2 )
# 		my_loc = np.atleast_2d(my_plan[t])
# 		my_noisy_plan = [my_plan[0]]
# 		for i in xrange(1, len(my_plan)-1):#loc_t = np.random.multivariate_normal(my_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
# 			loc_x = Q.randn( mu=my_plan[i][0], sigma=path_noise, name="int_x_"+str(i) )
# 			loc_y = Q.randn( mu=my_plan[i][1], sigma=path_noise, name="int_y_"+str(i) )
# 			loc_t = [loc_x, loc_y]
# 			my_noisy_plan.append(loc_t)
# 		my_noisy_plan.append(my_plan[-1])

# 		# for rendering purposes
# 		Q.keep("int_plan", my_noisy_plan)



# class Smart_Runner(object):
# 	def __init__(self, isovist=None, locs=None, seg_map=[None,None,None,None]):
# 		self.isovist = isovist
# 		self.locs = locs
# 		rx1,ry1,rx2,ry2 = seg_map
# 		#self.plan_path = lambda start_loc, goal_loc: planner.run_rrt_opt( start_loc, goal_loc, rx1,ry1,rx2,ry2 )
# 		self.show = False
# 		self.seg_map = seg_map
# 		# initialize model
# 		self.runner_model = create_runner_model(seg_map=seg_map, locs=locs, isovist=isovist)
# 		self.polys, self.epolys = load_segs()
		
# 	def run(self, Q):


# 		#simulates chaser and does inference over chaser's next step
# 		# run inference
# 		post_sample_traces = run_inference(Q, post_samples=3, samples=3)
# 		exp_next_step = expected_next_step_replanning(post_sample_traces, "enf_plan")

# 		if point_in_obstacle(exp_next_step, epolys):
# 			exp_next_step = get_clear_goal(chaser_locs[t], exp_next_step, polys)

# 		Q.keep("post_sample_traces", post_sample_traces)
# 		#---------------------------------------------------------


# 		# he plans to go to his goal
# 		rx1,ry1,rx2,ry2 = self.seg_map
# 		runner_plan = planner.run_rrt_opt( np.atleast_2d(Q.obs("runner_curr")), 
# 		np.atleast_2d(Q,obs("goal")), rx1,ry1,rx2,ry2 )

# 		# check if seen
# 		i_already_seen = 0
# 		# for i in xrange(t):
# 		# 	intersections = Q.cache["enf_intersections_t_"+str(i)]# get enforcer's fv for time t
# 		# 	i_already_seen = self.isovist.FindIntruderAtPoint(my_noisy_plan[i], intersections)
# 		# 	if (i_already_seen):
# 		# 		detected_prob = 0.999*i_already_seen + 0.001*(1-i_already_seen) 

# 		if not i_already_seen:
# 			# set up the enforcer view (forward vector, fv) for the next step
# 			cur_enf_loc = scale_up(enf_noisy_plan[t])
# 			next_enf_loc = scale_up(enf_noisy_plan[t+1])
# 			fv = direction(runner_plan[1], exp_next_step)
# 			intersections = self.isovist.GetIsovistIntersections(exp_next_step, fv)

# 			# does the enforcer see me at time 't'
# 			my_next_loc = scale_up(runner_plan[1])
# 			will_i_be_seen = self.isovist.FindIntruderAtPoint( my_next_loc, intersections )
# 			detected_prob = 0.999*will_i_be_seen + 0.001*(1-will_i_be_seen) # ~ flip(seen*.999 + (1-seen*.001)
		

# 		future_detection = Q.flip( p=detected_prob, name="int_detected" )

# 		# for rendering purposes
# 		Q.keep("int_plan", my_noisy_plan)
# 		Q.keep("enf_plan", enf_noisy_plan)


# 		# was_seen random vairable




