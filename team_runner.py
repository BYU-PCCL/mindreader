import program_trace as program_trace
from methods import load_isovist_map, scale_up, direction, dist, load_segs, point_in_obstacle, get_clear_goal
from methods import plot_middlemost_sample, plot_outermost_sample
from my_rrt import *
import copy
from scipy.misc import logsumexp
import cPickle
from multiprocessing import Pool
from tqdm import tqdm
import planner
import time
from random import randint
from program_trace import ProgramTrace
from inference_alg import importance_sampling, metroplis_hastings, importance_resampling



class BasicRunner(object):
	def __init__(self, isovist=None, locs=None, seg_map=[None,None,None,None]):
		# field of view calculator
		self.isovist = isovist
		# possible start/goal locations
		self.locs = locs
		self.cnt = len(locs)
		# map
		self.seg_map = seg_map
		rx1,ry1,rx2,ry2 = seg_map


	# run the model inside this function
	def run(self, Q):
		self.run_basic(Q)

	def run_basic(self, Q, path_noise=0.003):
		rx1,ry1,rx2,ry2 = self.seg_map
		t = Q.choice( p=1.0/40*np.ones((1,40)), name="t" )

		start_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="run_start" )
		goal_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="run_goal" )
		start = np.atleast_2d( self.locs[start_i] )
		goal = np.atleast_2d( self.locs[goal_i] )

		# plan using the latent variables of start and goal
		my_plan = planner.run_rrt_opt( np.atleast_2d(start), 
		np.atleast_2d(goal), rx1,ry1,rx2,ry2 )
		my_loc = np.atleast_2d(my_plan[t])

		# add noise to the plan
		my_noisy_plan = [my_plan[0]]
		for i in xrange(1, len(my_plan)-1):#loc_t = np.random.multivariate_normal(my_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			loc_x = Q.randn( mu=my_plan[i][0], sigma=path_noise, name="run_x_"+str(i) )
			loc_y = Q.randn( mu=my_plan[i][1], sigma=path_noise, name="run_y_"+str(i) )
			loc_t = [loc_x, loc_y]
			my_noisy_plan.append(loc_t)
		my_noisy_plan.append(my_plan[-1])

		Q.keep("runner_plan", my_noisy_plan)


	#post_sample_traces = run_inference(q, post_samples=6, samples=5)
	# def run_inference(self, program_trace, post_samples, particles):
	# 	post_traces = []
	# 	for i in  tqdm(xrange(post_samples)):
	# 		post_sample_trace = sampling_importance(trace, samples=samples)
	# 		post_traces.append(post_sample_trace)
	# 	return post_traces


class BasicRunnerPOM(object):
	def __init__(self, isovist=None, locs=None, seg_map=[None,None,None,None], mode="collab"):
		# field of view calculator
		self.isovist = isovist
		# possible start/goal locations
		self.locs = locs
		self.cnt = len(locs)
		# map
		self.seg_map = seg_map
		rx1,ry1,rx2,ry2 = seg_map
		self.mode = mode # advers for adversarial , collab for collaborative


	# run the model inside this function
	def run(self, Q):
		self.run_basic_partial(Q)

	def run_basic_partial(self, Q, path_noise=0.000):
		rx1,ry1,rx2,ry2 = self.seg_map
		t = Q.choice( p=1.0/40*np.ones((1,40)), name="t" )

		# #------------- model other agent movements (past and future) -------------- 
		o_start_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="other_run_start" )
		o_goal_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="other_run_goal" )

		o_start = np.atleast_2d( self.locs[o_start_i] )
		o_goal = np.atleast_2d( self.locs[o_goal_i] )

		# plan using the latent variables of start and goal
		# other_plan = planner.run_rrt_opt( np.atleast_2d(o_start), 
		# np.atleast_2d(o_goal), rx1,ry1,rx2,ry2 )

		#print Q.fetch_obs()
		prev_true_loc =  np.atleast_2d([Q.get_obs("other_run_x_"+str(t-1)), Q.get_obs("other_run_y_"+str(t-1))])
		#print "prev_true_loc", prev_true_loc
		other_plan = planner.run_rrt_opt( prev_true_loc, np.atleast_2d(o_goal), rx1,ry1,rx2,ry2 )
		u = 1

		# add noise to the plan
		other_noisy_plan = [self.locs[o_start_i]]
		for i in xrange(1, len(other_plan)-1):#loc_t = np.random.multivariate_normal(my_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			
			if i < t:
				# do not sample if already known to be true:
				loc_t = [Q.get_obs(name="other_run_x_"+str(i)), Q.get_obs(name="other_run_y_"+str(i))]
			else:
				loc_x = Q.randn( mu=other_plan[u][0], sigma=path_noise, name="other_run_x_"+str(i) )
				loc_y = Q.randn( mu=other_plan[u][1], sigma=path_noise, name="other_run_y_"+str(i) )
				loc_t = [loc_x, loc_y]
				u+=1
			other_noisy_plan.append(loc_t)
		other_noisy_plan.append(other_plan[-1])
		

		#------------- model agent's own movements (past and future) --------------
		start_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="run_start" )
		goal_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="run_goal" )

		start = np.atleast_2d( self.locs[start_i] )
		goal = np.atleast_2d( self.locs[goal_i] )

		# plan using the latent variables of start and goal
		my_plan = planner.run_rrt_opt( np.atleast_2d(start), 
		np.atleast_2d(goal), rx1,ry1,rx2,ry2 )
		

		# add noise to the plan
		my_noisy_plan = [my_plan[0]]
		for i in xrange(1, len(my_plan)-1):#loc_t = np.random.multivariate_normal(my_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			loc_x = Q.randn( mu=my_plan[i][0], sigma=path_noise, name="run_x_"+str(i) )
			loc_y = Q.randn( mu=my_plan[i][1], sigma=path_noise, name="run_y_"+str(i) )
			loc_t = [loc_x, loc_y]
			my_noisy_plan.append(loc_t)
		my_noisy_plan.append(my_plan[-1])

		my_loc = my_noisy_plan[t]

		#-------------------------- detections ---------------------
		if self.mode == "collab":
			t_detected = []
			PATH_LIMIT = 30
			for i in xrange(0, PATH_LIMIT):
				cur_loc = scale_up(my_noisy_plan[i])
				intersections = None
				detection_prob = 0.1
				# face the runner if within certain radius
				if dist(my_noisy_plan[i], other_noisy_plan[i]) <= .4: #.35:
					fv = direction(scale_up(other_noisy_plan[i]), cur_loc)
					intersections = self.isovist.GetIsovistIntersections(cur_loc, fv)
				
					# does the agent see other at time 't'
					other_loc = scale_up(other_noisy_plan[i])
					will_other_be_seen = self.isovist.FindIntruderAtPoint( other_loc, intersections )
					if will_other_be_seen:
						detection_prob = 0.9
						t_detected.append(i)

				future_detection = Q.flip( p=detection_prob, name="detected_t_"+str(i) )
				# ^ might want to change this to lflip when you do experiments next
				# you wrote this on 1.19.2018 - iris
				
				Q.keep("intersections-t-"+str(i), intersections)

			same_goal = 0
			if goal_i == o_goal_i:
				same_goal = 1

			same_goal_prob = .999*same_goal + .001*(1-same_goal)

			runners_same_goal = Q.lflip( lp=same_goal_prob, name="same_goal" ) 

			Q.keep("t_detected", t_detected)
			Q.keep("my_plan", my_noisy_plan)
			Q.keep("other_plan", other_noisy_plan)

		if self.mode == "advers":
			t_detected = []
			PATH_LIMIT = 30
			for i in xrange(0, PATH_LIMIT):
				other_loc = scale_up(other_noisy_plan[i])
				intersections = None
				detection_prob = -0.0001
				# face the runner if within certain radius
				if dist(my_noisy_plan[i], other_noisy_plan[i]) <= .4: #.35:
					# -----if other is looking for me
					#print ("fv:", scale_up(my_noisy_plan[i]), other_loc)
					fv = direction(scale_up(my_noisy_plan[i]), other_loc)
					intersections = self.isovist.GetIsovistIntersections(other_loc, fv)
					#print intersections
				
					# does the chaser see other at time 'i'
					curr_loc = scale_up(my_noisy_plan[i])
					#if other is looking for me
					will_I_be_seen = self.isovist.FindIntruderAtPoint( curr_loc, intersections )

					#print will_I_be_seen
					if will_I_be_seen:
						detection_prob = -1.0
						t_detected.append(i)

				
				future_detection = Q.clflip( lp=detection_prob, name="detected_t_"+str(i) )

				#future_detection = Q.flip( p=detection_prob, name="detected_t_"+str(i) )
				
				Q.keep("intersections-t-"+str(i), intersections)

			Q.keep("t_detected", t_detected)
			Q.keep("my_plan", my_noisy_plan)
			Q.keep("other_plan", other_noisy_plan)


	#post_sample_traces = run_inference(q, post_samples=6, samples=5)
	def run_inference(self, program_trace, post_samples, particles):
		post_traces = []
		for i in  tqdm(xrange(post_samples)):
			post_sample_trace = sampling_importance(trace, samples=samples)
			post_traces.append(post_sample_trace)
		return post_traces

		

class TOMRunnerPOM(object):
	def __init__(self, isovist=None, locs=None, seg_map=[None,None,None,None], nested_model=None, inner_samples=1, mode="collab"):
		# field of view calculator
		self.isovist = isovist
		# possible start/goal locations
		self.locs = locs
		self.cnt = len(locs)
		# map
		self.seg_map = seg_map
		rx1,ry1,rx2,ry2 = seg_map
		
		self.nested_model = nested_model # want to add parallel processessing here
		self.L = inner_samples
		self.mode = mode # advers for adversarial , collab for collaborative

		self.directory = None

	def set_dir(self, directory):
		self.directory = directory
		

	# run the model inside this function
	def run(self, Q):
		return self.run_tom_partial(Q)

	def run_tom_partial(self, Q, path_noise=0.000):
		
		rx1,ry1,rx2,ry2 = self.seg_map
		t = Q.choice( p=1.0/40*np.ones((1,40)), name="t" )

		#------------- model agent's movements (past and future) --------------
		start_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="init_run_start" )
		goal_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="init_run_goal" )
		start = np.atleast_2d( self.locs[start_i] )
		goal = np.atleast_2d( self.locs[goal_i] )
		
		# plan using the latent variables of start and goal
		
		prev_true_loc =  np.atleast_2d([Q.get_obs("init_run_x_"+str(t-1)), Q.get_obs("init_run_y_"+str(t-1))])

		my_plan = planner.run_rrt_opt( prev_true_loc, goal, rx1,ry1,rx2,ry2 )
		u = 1
		my_noisy_plan = [self.locs[start_i]]
		
		for i in xrange(1, len(my_plan)-1):#loc_t = np.random.multivariate_normal(my_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			
			if i < t:
				# do not sample if already known to be true
				loc_t = [Q.get_obs("init_run_x_"+str(i)), Q.get_obs("init_run_y_"+str(i))]
			else:
				loc_x = Q.randn( mu=my_plan[u][0], sigma=path_noise, name="init_run_x_"+str(i) )
				loc_y = Q.randn( mu=my_plan[u][1], sigma=path_noise, name="init_run_y_"+str(i) )
				loc_t = [loc_x, loc_y]
				u += 1
			my_noisy_plan.append(loc_t)
		my_noisy_plan.append(my_plan[-1])
		my_loc = my_noisy_plan[t]
		
		#assert (len(my_noisy_plan) == 30)

		#---------------- do inference --------------------------------------------
		other_noisy_plan = None
		other_inferred_trace = None

		if self.mode == "collab":
			post_sample_traces, other_inferred_goal = self.collaborative_nested_inference(Q)
			for trace in post_sample_traces:
				if trace["run_goal"] == other_inferred_goal:
					other_inferred_trace = trace
					other_noisy_plan = trace["my_plan"]
					break

		
		if self.mode == "advers":
			all_t_detected = []
			other_plans = []
			L = self.L
			all_Qls_scores = []
			all_Qls_traces = []
			all_Qls_obs = []
			all_ql_scores = []
			#print "init Q score:", Q.get_score()
			for l in xrange(L):
				q = self.condition_middle_model(Q)
				# print ("Q obs:", Q.fetch_obs())
				# print ("q obs:", q.fetch_obs())
				# raw_input()
				Q_l = Q.get_copy()
				q_score_l, q_l = q.run_model()
				#plot_middlemost_sample(q_l, q_score_l, self.directory)
				other_noisy_plan = q_l["my_plan"]

				#---------------- need to add RV of detection for each time step ----------
				t_detected = []
				PATH_LIMIT = 30
				will_other_be_seen = False
				for i in xrange(0, PATH_LIMIT):
					cur_loc = scale_up(my_noisy_plan[i])
					intersections = None
					detection_prob = -.0001
					# face the runner if within certain radius
					if dist(my_noisy_plan[i], other_noisy_plan[i]) <= .4: #.35:
						fv = direction(scale_up(other_noisy_plan[i]), cur_loc)
						intersections = self.isovist.GetIsovistIntersections(cur_loc, fv)
					
						# does the agent see other at time 't'
						other_loc = scale_up(other_noisy_plan[i])
						will_other_be_seen = self.isovist.FindIntruderAtPoint( other_loc, intersections )
						if will_other_be_seen:
							detection_prob = -1
							#print ("chaser detected runner")
							t_detected.append(i)
					
					#future_detection = Q_l.flip( p=detection_prob, name="detected_t_"+str(i) )
					
					future_detection = Q_l.lflip( lp=detection_prob, name="detected_t_"+str(i))
					#print "REAL Q score after:", Q.get_score()
					#print "Q score before:", Q_l.get_score()

					#raw_input()
					#Q_l.keep("intersections-t-"+str(i), intersections)
				# print ("tested chaser with runner")
				# raw_input()
				
				# add q trace to Q trace and add q's log likelihood to Q's
				#print "q_score_l:", q_score_l
				Q_l.add_trace(name="q_trace", trace=q_l, score=q_score_l)
				all_ql_scores.append(q_score_l)
				#print "list:", all_ql_scores

				# Q_l.keep("t_detected", t_detected)
				all_t_detected.append(t_detected)
				# Q_l.keep("my_plan", my_noisy_plan)
				# Q_l.keep("other_plan", other_noisy_plan)
				other_plans.append(other_noisy_plan)
				
				# Q_l.keep("other_run_start", q_l["run_start"])
				# Q_l.keep("other_run_goal", q_l["run_goal"])
				all_Qls_scores.append(Q_l.get_score())
				all_Qls_traces.append(Q_l.get_trace())
				all_Qls_obs.append(Q_l.get_obs())

				

		#print ("all ql scores:", all_ql_scores)
		#print ("all_Qls_scores:", all_Qls_scores)
		Q.keep("all_ql_scores", np.array(all_ql_scores))
		Q.keep("all_Qls_obs", all_Qls_obs)
		Q.keep("all_Qls_traces", all_Qls_traces)
		Q.keep("all_Qls_scores", np.array(all_Qls_scores))
		Q.keep("mean", np.mean(all_Qls_scores))
		Q.keep("my_plan", my_noisy_plan)
		Q.keep("t_detected", all_t_detected)
		Q.keep("other_plan", other_plans)



		#if (randint(0,10) == 1):
		#plot_outermost_sample(Q.get_trace(), np.mean(all_Qls_scores), self.directory, t, int(time.time()))
		#return 1

	# need to look at how I conditioned the previous model
	def collaborative_nested_inference(self, Q):
		t = Q.fetch("t")
		q = ProgramTrace(self.nested_model)
		q.condition("other_run_start", Q.fetch("init_run_start"))
		q.condition("t", t) 
		q.condition("run_start", Q.get_obs("other_run_start"))
		q.condition("same_goal", True)

		# Assumes that if Agent A detects Agent B, then Agent B detects Agent A
		for i in xrange(24):
			q.condition("detected_t_"+str(i), Q.get_obs("detected_t_"+str(i)))

			#condition on observations from other agent of me
			if (Q.get_obs("detected_t_"+str(i)) == True):
				if (i == (t-1)):
					q.condition("run_x_"+ str(i), Q.get_obs("other_x_"+str(i)))
					q.condition("run_y_"+ str(i), Q.get_obs("other_y_"+str(i)))

		prev_t =(t-1)
		q.condition("other_run_x_"+str(prev_t), Q.fetch("init_run_x_"+str(prev_t)))
		q.condition("other_run_y_"+str(prev_t), Q.fetch("init_run_y_"+str(prev_t)))

		print q.cond_data_db
		trace =  self.get_trace_for_most_probable_goal_location(q)

		return trace

	

	# adversarial NESTED Importance RESAMPLING <-- maybe this is not nested after all?
	def adversarial_importance_resampling(self, Q):
		#print ("----------------------------------inside adversarial_importance_resampling")
		Q, q = self.condition_middle_model(Q)
		q_post_sample_traces, q_log_normalizer = importance_resampling(q, particles=self.SP, _print_inner=False)
		#draw from uniform since all weights are the same (the average weight = log_normalizer)
		chosen_index = np.random.randint(0,self.SP)
		q_trace_vals = q_post_sample_traces[chosen_index]

		#print ("----- log norm of q:", q_log_normalizer)
		#print ("trace val before:", Q.get_score())
		# keep q trace inside Q trace and update Q score based on log_normalizer
		Q.keep(name="chasers_runner_trace", value=q_trace_vals)
		Q.keep("q_log_normalizer", q_log_normalizer)
		#print ("trace val after:", Q.get_score())
		return q_post_sample_traces, q_trace_vals
		
	# adversarial importance sampling (NO INNER INFERENCE) -> sample from posterior porprotional to weight
	def adversarial_importance_sampling(self,Q):
		Q, q = self.condition_middle_model(Q)
		# run the chasers runner model forward
		q_samples, q_scores = q.run_model()
		# add q trace to Q trace
		#Q.add_trace(name="chasers_runner_trace", trace=q_trace_vals, score=q_score)
		return q_samples, q_scores


	def condition_middle_model(self, Q):
		t = Q.fetch("t")
		q = ProgramTrace(self.nested_model)
		q.condition("other_run_start", Q.fetch("init_run_start"))
		#XXX just for testing purposes XXX remove in real simulation
		#q.condition("other_run_goal", Q.fetch("init_run_goal"))
		q.condition("t", Q.fetch("t")) 
		q.condition("run_start", Q.get_obs("other_run_start"))
		#XXX just for testing purposes XXX remove in real simulation
		#q.condition("run_goal", Q.get_obs("other_run_goal"))

		for i in xrange(0, 29):
			# the runner wants all detections to be False
			q.condition("detected_t_"+str(i), False)
			# assumes that the chaser's runner has a perfect knowledge of the chaser
			if i <  t:
				q.set_obs("other_run_x_"+str(i), Q.get_obs("init_run_x_"+str(i)))
				q.set_obs("other_run_y_"+str(i), Q.get_obs("init_run_y_"+str(i)))
				

		return q


	


class TOMCollabRunner(object):
	def __init__(self, isovist=None, locs=None, seg_map=[None,None,None,None], nested_model=None):
		# field of view calculator
		self.isovist = isovist
		# possible start/goal locations
		self.locs = locs
		self.cnt = len(locs)
		# map
		self.seg_map = seg_map
		rx1,ry1,rx2,ry2 = seg_map
		self.nested_model = nested_model


	# run the model inside this function
	def run(self, Q):
		self.tom_run_nested_inference(Q)


	def tom_run_nested_inference(self, Q, path_noise=0.003):
		# represents the other agent
		rx1,ry1,rx2,ry2 = self.seg_map
		t = Q.choice( p=1.0/40*np.ones((1,40)), name="t" )

		start_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="co_run_start" )
		goal_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="co_run_goal" )
		start = np.atleast_2d( self.locs[start_i] )
		goal = np.atleast_2d( self.locs[goal_i] )

		# plan using the latent variables of start and goal
		my_plan = planner.run_rrt_opt( np.atleast_2d(start), 
		np.atleast_2d(goal), rx1,ry1,rx2,ry2 )
		my_loc = np.atleast_2d(my_plan[t])

		# add noise to the plan
		my_noisy_plan = [my_plan[0]]
		for i in xrange(1, len(my_plan)-1):#loc_t = np.random.multivariate_normal(my_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			loc_x = Q.randn( mu=my_plan[i][0], sigma=path_noise, name="co_run_x_"+str(i) )
			loc_y = Q.randn( mu=my_plan[i][1], sigma=path_noise, name="co_run_y_"+str(i) )
			loc_t = [loc_x, loc_y]
			my_noisy_plan.append(loc_t)
		my_noisy_plan.append(my_plan[-1])

		post_sample_traces = self.collaborative_nested_inference(Q)

		# get most probable goal
		partners_goal = self.get_most_probable_goal(post_sample_traces, "run_goal")
		
		same_goal = 0
		if goal_i == partners_goal:
			same_goal = 1
		same_goal_prob = 0.999*same_goal + 0.001*(1-same_goal)

		runners_same_goal = Q.flip( p=same_goal_prob, name="same_goal" ) 

		Q.keep("co_runner_plan", my_noisy_plan)
		Q.keep("partner_goal", partners_goal)
		Q.keep("nested_post_sample_traces", post_sample_traces)


	def collaborative_nested_inference(self, Q):
		t = Q.get_obs("t")
		q = ProgramTrace(self.nested_model)
		q.condition("run_start", Q.get_obs("run_start"))
		q.condition("t", Q.get_obs("t")) 
		# condition on previous time steps
		
		for prev_t in xrange(t):
			q.condition("run_x_"+str(prev_t), Q.get_obs("run_x_"+str(prev_t)))
			q.condition("run_y_"+str(prev_t), Q.get_obs("run_y_"+str(prev_t)))
		
		post_sample_traces = self.run_inference(q, post_samples=10, samples=16)
		return post_sample_traces

	def run_inference(self, trace, post_samples=16, samples=32):
		post_traces = []
		for i in  tqdm(xrange(post_samples)):
			#post_sample_trace = importance_sampling(trace, samples)
			post_sample_trace = importance_sampling(trace, samples)
			post_traces.append(post_sample_trace)
		return post_traces


	def get_most_probable_goal(self, post_sample_traces, goal_rv_name):
		goal_list = []
		# show post sample traces on map
		for trace in post_sample_traces:
			inferred_goal = trace[goal_rv_name]
			goal_list.append(inferred_goal)

		# list with probability for each goal
		goal_probabilities = []
		total_num_inferences = len(goal_list)
		# turn into percents
		for goal in xrange(6):
			goal_cnt = goal_list.count(goal)
			goal_prob = goal_cnt / float(total_num_inferences)
			goal_probabilities.append(goal_prob)

		return goal_probabilities.index(max(goal_probabilities))









		





