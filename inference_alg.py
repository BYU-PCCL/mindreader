from program_trace import ProgramTrace
from random import randint
import random
import numpy as np
import copy
#from scipy.misc import logsumexp
from scipy.special import logsumexp
from tqdm import tqdm
import time
import os
from multiprocessing import Pool



# # importance sampling using the score of trace
# def single_run_model(arg, **kwarg):
# 	return ProgramTrace.run_model(*arg, **kwarg)
	

# def single_run_model_sarg( x ):
# 	if not hasattr( single_run_model_sarg, '__my_init' ):
# 		single_run_model_sarg.__my_init = True # use function attributes as static variables
# 		import os
# 		np.random.seed( os.getpid() )
# 	return single_run_model()


# def mylocal_func( params ):
# 	Q = new Q( params )
# 	 return Q.run_model()

# def importance_sampling(Q, particles):
# 	traces = []
# 	scores = np.arange(particles)

# 	# for i in xrange(particles):
# 		# score, trace_vals = Q.run_model()
# 		# traces.append(copy.deepcopy(trace_vals))
# 		# scores[i] = score

	
# #   p = Pool( 12 )
# 	p = Pool( 5 )
#     # we do all of this because Pool.map pickles its arguments, and you can't pickle a lambda...
# 	params = ((),) * particles
# 	results = p.map( mylocal_func, params )
# 	print results


# 	#print "params:", params
# 	#results = p.map( single_run_model_sarg, zip([Q]*particles, ((),) * particles) )

# 	# for tmp_retval in results:
# 	# 	scores.append( tmp_retval[0] )
# 	# 	traces.append( tmp_retval[1] )

# 	# print "scores:", scores

	
# 	# get weight for score
# 	# weights = np.exp(scores - logsumexp(scores))

# 	# # sample
# 	# chosen_index = np.random.choice([i for i in range(particles)], p=weights)
# 	# return traces[chosen_index]
# 	return 0





# ('previous weights:', array([  2.49886067e-01,   2.49886067e-01,   2.49886067e-01,
#          1.43672550e-19,   2.27866598e-04,   1.31012408e-22,
#          1.43672550e-19,   2.70036699e-31,   2.27866598e-04,
#          2.49886067e-01]))
# ('new weights:', array([  2.49886067e+01,   2.49886067e+01,   2.49886067e+01,
#          1.43672550e-17,   2.27866598e-02,   1.31012408e-20,
#          1.43672550e-17,   2.70036699e-29,   2.27866598e-02,
#          2.49886067e+01]))

def update_conditions(conditions, trace, t):
	# assume runner was not seen before (for next inference step)
	new_conditions = copy.deepcopy(conditions)
	new_conditions["detected_t_"+str(t)] = False
	new_conditions["t"] = conditions["t"]+1
	return new_conditions

def update_observations(observations, trace, t):
	new_observations = copy.deepcopy(observations)
	new_observations["init_run_x_"+str(t)] = trace["init_run_x_"+str(t)]
	new_observations["init_run_y_"+str(t)] = trace["init_run_y_"+str(t)]
	return new_observations


def get_prob_detection_t(trace, K):
	all_t_detected = trace["t_detected"]
	total_detections = 0
	for j in xrange(len(all_t_detected)):
		detections = all_t_detected[j]
		if len(detections) > 0:
			total_detections += 1
	prob = total_detections/float(K)
	return prob

from methods import plot_outermost_sample

def sequential_monte_carlo(T, model, conditions, observations, K):

	file_id = str(int(time.time()))
	directory = "PO_forward_runs/conditioned/SMC_simulations/sim-"+file_id
	os.mkdir(directory)
	model.set_dir(directory)
	detection_probabilities = [0.0]
	Q_T = []
	for t in tqdm(xrange(1, T-1)):
		Q = ProgramTrace(model)

		for name in observations.keys():
			Q.set_obs(name, observations[name])
		for name in conditions.keys():
			Q.condition(name, conditions[name])

		sampled_Q_ks = []
		Q_k_scores = np.arange(K)
		for k in xrange(K):
			score, Q_trace_k = Q.run_model()
			Q_k_scores[k] = Q.fetch("mean")
			#plot_outermost_sample(Q_trace_k, Q_k_scores[k], directory, k)
			sampled_Q_ks.append(Q_trace_k)
		sampled_Q_ks = np.array(sampled_Q_ks)

		log_normalizer = logsumexp(Q_k_scores) - np.log(K) 
		weights = np.exp(Q_k_scores - log_normalizer - np.log(K))
		#print ("outer weights:", weights, "log norm:", log_normalizer)
		
		resampled_indexes = np.random.choice([i for i in range(K)], K, replace=True, p=weights) 

		resampled_Qs = sampled_Q_ks[resampled_indexes]

		sampled_Q_trace = resampled_Qs[np.random.randint(0,K)]
		#plot_outermost_sample(sampled_Q_trace, log_normalizer, directory, t)
		
		Q_T.append(sampled_Q_trace)

		conditions = update_conditions(conditions, sampled_Q_trace, t)
		observations = update_observations(observations, sampled_Q_trace, t)
		detection_probabilities.append(get_prob_detection_t(sampled_Q_trace, K))
		print ("detection_probabilities", detection_probabilities)

	return Q_T, detection_probabilities


def single_run_model_sarg(x):
	if not hasattr( single_run_model_sarg, '__my_init' ):
		single_run_model_sarg.__my_init = True # use function attributes as static variables
		import os
		np.random.seed( os.getpid() )
	return single_run_model( x[0], x[1], x[2])

def single_run_model(model, observations, conditions):
	Q = ProgramTrace(model)
	
	for name in observations.keys():
		Q.set_obs(name, observations[name])
	for name in conditions.keys():
		Q.condition(name, conditions[name])
	score, trace = Q.run_model()
	mean = Q.fetch("mean")
	Q = None
	return (mean, trace)

import sys
import pickle
from my_rrt import *

# We use K different chasers and continue them through time
#params = ((model, observations, conditions),)*K # K different params
def sequential_monte_carlo_par(params, K, T=30):

	file_id = str(int(time.time()))
	directory = "PO_forward_runs/conditioned/SMC_simulations/sim-"+file_id
	os.mkdir(directory)
	#model.set_dir(directory)
	for p in params:
		p[0].set_dir(directory)

	# KQ_T = []
	# KQ_T_scores = []
	for t in tqdm(xrange(1, T-1)):
		
		sampled_Q_ks = None
		Q_k_scores = None

		p = Pool(9)
		results = p.map(single_run_model_sarg, params)
		p.close()
		p.join() 
		Q_k_scores = np.array((zip(*results))[0])
		sampled_Q_ks = np.array((zip(*results))[1])

		# print("K scores:", KQ_T_scores)

		# KQ_T.append(sampled_Q_ks)
		# KQ_T_scores.append(Q_k_scores)
		
		pickle.dump( [K, t, KQ_T, KQ_T_scores], open( directory+"/"+file_id+"_t-"+str(t)+".p", "wb" ))

		updated_params = []
		i = 0
		for p in params:
			conditions = update_observations(p[1], sampled_Q_ks[i], t)
			observations = update_conditions(p[2], sampled_Q_ks[i], t)
			updated_params.append((p[0],conditions, observations))
			#plot_outermost_sample(sampled_Q_ks[i], Q_k_scores[i], directory, t, i)
			i+=1
		
		params = tuple(updated_params)

	#pickle.dump( [K, KQ_T, KQ_T_scores], open( directory+"/"+file_id+".p", "wb" ))
	# return KQ_T, KQ_T_scores #detection_probabilities

# this is SMC where K is a single param and we resample and choose one
# chaser sample to continue in time
def special_sequential_monte_carlo_par(T, model, conditions, observations, K):

	file_id = str(int(time.time()))
	directory = "PO_forward_runs/conditioned/SMC_simulations/sim-"+file_id
	os.mkdir(directory)
	model.set_dir(directory)
	detection_probabilities = [0.0]
	Q_T = []
	for t in tqdm(xrange(1, T-1)):
		
		sampled_Q_ks = []

		p = Pool(10)
		params = ((model, observations, conditions),)*K # K different params
		results = p.map(single_run_model_sarg, params)
		
		Q_k_scores = np.array((zip(*results))[0])
		sampled_Q_ks = np.array((zip(*results))[1])

		log_normalizer = logsumexp(Q_k_scores) - np.log(K) 
		weights = np.exp(Q_k_scores - log_normalizer - np.log(K))
		print ("outer weights:", weights, "log norm:", log_normalizer)
		
		resampled_indexes = np.random.choice([i for i in range(K)], K, replace=True, p=weights) 

		resampled_Qs = sampled_Q_ks[resampled_indexes]

		sampled_Q_trace = resampled_Qs[np.random.randint(0,K)] # compute K params and continue (ea obs and cond)
		plot_outermost_sample(sampled_Q_trace, log_normalizer, directory, t)
		
		Q_T.append(sampled_Q_trace)


		conditions = update_conditions(conditions, sampled_Q_trace, t)
		observations = update_observations(observations, sampled_Q_trace, t)
		detection_probabilities.append(get_prob_detection_t(sampled_Q_trace, K))
		print ("detection_probabilities", detection_probabilities)

	# plot probability
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(list(xrange(T-1)), detection_probabilities, 'bo')
	ax.plot(list(xrange(T-1)), detection_probabilities, 'b--')
	ax.set_xlabel("time step")
	ax.set_ylabel("probability of detection")
	fig.savefig(directory+"/detection-probs.eps", bbox_inches='tight')

	return Q_T, detection_probabilities


def importance_resampling(Q, particles, _print_inner=False, _print_outer=False):
	traces = []
	scores = np.arange(particles)
	inner_log_norms = []
	outer_detect_totals = 0
	for i in tqdm(xrange(particles)):
		score, trace_vals = Q.run_model()
		traces.append(copy.deepcopy(trace_vals))
		scores[i] = score

		if _print_outer:
			inner_log_norms.append(trace_vals["q_log_normalizer"])

		if False:
			from show_models import setup_plot, load_polygons, polygons_to_segments, close_plot
			from matplotlib import pyplot as plt
			import time
			

			PS = 1
			SP = particles
			trace = trace_vals
			if len(trace["t_detected"]) > 0:
				outer_detect_totals +=1
			inner_log_norms.append(trace["q_log_normalizer"])
			if len(trace["t_detected"])==0:
				print ("q_log_norm", trace["q_log_normalizer"])
				print ("score:", score)
				print ("particle:", i, "detection count:", len(trace["t_detected"]))

			if True: #len(trace["t_detected"])>0:

				print ("---------------------------------------------------")
				print ("****************  GOOD ONE! ***********************")
				print ("---------------------------------------------------")
				print ("q_log_norm", trace["q_log_normalizer"])
				print ("score:", score)
				print ("particle:", i, "detection count:", len(trace["t_detected"]))
				
				print ("---------------------------------------------------")
				print ("***************************************************")
				print ("---------------------------------------------------")
				locs = [[ 0.100, 1-0.900 ],[ 0.566, 1-0.854 ],[ 0.761, 1-0.665 ],
				[ 0.523, 1-0.604 ],[ 0.241, 1-0.660 ],[ 0.425, 1-0.591 ],
				[ 0.303, 1-0.429 ],[ 0.815, 1-0.402 ],[ 0.675, 1-0.075 ],
				[ 0.432, 1-0.098 ] ]
				poly_map = polygons_to_segments( load_polygons( "./paths.txt" ) )
				fig, ax = setup_plot(poly_map, locs)
			
				path = trace["my_plan"]
				t = trace["t"]
				for i in range(0, 39):
					ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
						'orange', linestyle=":", linewidth=1, label="Agent's Plan")
				
					
				# mark the runner at time t on its plan
				ax.scatter( path[t-1][0],  path[t-1][1] , s = 95, facecolors='none', edgecolors='orange')

				path = trace["other_plan"]

				# mark the runner at time t on its plan
				ax.scatter( path[t-1][0],  path[t-1][1] , s = 95, facecolors='none', edgecolors='blue')

				for i in range(0, 39):
					ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
						'blue', linestyle="--", linewidth=1, label="Other's Plan")
					if i in trace["t_detected"]:
						ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')
					
				

				plt.figtext(0.92, 0.85, "Values", horizontalalignment='left', weight="bold") 
				plt.figtext(0.92, 0.80, "R Start: " +str(trace["init_run_start"]), horizontalalignment='left') 
				#plt.figtext(0.92, 0.75, "A Goal: " +str(trace["init_run_goal"]), horizontalalignment='left') 
				plt.figtext(0.92, 0.75, "C Start: " +str(trace["other_run_start"]), horizontalalignment='left')
				#plt.figtext(0.92, 0.65, "B Goal: " +str(trace["other_run_goal"]), horizontalalignment='left') 
				plt.figtext(0.92, 0.70, "time step: " +str(trace["t"]), horizontalalignment='left') 
				plt.figtext(0.92, 0.65, "C detected R count: " +str(len(trace["t_detected"])), horizontalalignment='left') 
				plt.figtext(0.92, 0.60, "score:" + str(score), horizontalalignment='left')
				#close_plot(fig, ax, plot_name="PO_forward_runs/unconditioned/single_samples/tom/tom_run_and_find-"+str(int(time.time()))+".eps")
				close_plot(fig, ax, 
					plot_name="PO_forward_runs/conditioned/single_samples/tom/outer-most_model-P"+
					str(PS)+"-S"+str(SP)+"-"+str(int(time.time()))+".eps")



		# XXX THE CODE BELOW is a way to visualize what is happening
		# in a trace and its respective score (when conditioned)
		# specifically this is meant for the middle model when
		# it's the chaser's runner model. 

		if _print_inner:
				#---------plt the weights of different forward runs--------------------------
			from show_models import setup_plot, load_polygons, polygons_to_segments, close_plot
			from matplotlib import pyplot as plt
			import time
			locs = [[ 0.100, 1-0.900 ],[ 0.566, 1-0.854 ],[ 0.761, 1-0.665 ],
			[ 0.523, 1-0.604 ],[ 0.241, 1-0.660 ],[ 0.425, 1-0.591 ],
			[ 0.303, 1-0.429 ],[ 0.815, 1-0.402 ],[ 0.675, 1-0.075 ],
			[ 0.432, 1-0.098 ] ]
			poly_map = polygons_to_segments( load_polygons( "./paths.txt" ) )

			PS = 1
			SP = particles
			
			fig, ax = setup_plot(poly_map, locs)
			trace = trace_vals
			
			path = trace["my_plan"]
			t = trace["t"]
			for i in range(0, t-1):
				ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
					'blue', linestyle=":", linewidth=1, label="Agent's Plan")
				if i in trace["t_detected"]:
					ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')
			for i in range(t-1, 39):
				ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
					'blue', linestyle=":", linewidth=1)
				if i in trace["t_detected"]:
					ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')

			# mark the runner at time t on its plan
			ax.scatter( path[t-1][0],  path[t-1][1] , s = 95, facecolors='none', edgecolors='blue')

			path = trace["other_plan"]

			for i in range(0, t-1):
				ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
					'orange', linestyle="--", linewidth=1, label="Other's Plan")
				# else:
				# 	ax.scatter( path[i][0],  path[i][1] , s = 30, facecolors='none', edgecolors='grey')
			# mark the runner at time t on its plan
			ax.scatter( path[t-1][0],  path[t-1][1] , s = 95, facecolors='none', edgecolors='orange')

			for i in range(t, 39):
				ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
					'orange', linestyle="--", linewidth=1, label="Other's Plan")
				
			

			plt.figtext(0.92, 0.85, "Values", horizontalalignment='left', weight="bold") 
			plt.figtext(0.92, 0.80, "R Start: " +str(trace["run_start"]), horizontalalignment='left') 
			#plt.figtext(0.92, 0.75, "A Goal: " +str(trace["init_run_goal"]), horizontalalignment='left') 
			plt.figtext(0.92, 0.75, "C Start: " +str(trace["other_run_start"]), horizontalalignment='left')
			#plt.figtext(0.92, 0.65, "B Goal: " +str(trace["other_run_goal"]), horizontalalignment='left') 
			plt.figtext(0.92, 0.70, "time step: " +str(trace["t"]), horizontalalignment='left') 
			plt.figtext(0.92, 0.65, "C detected R count: " +str(len(trace["t_detected"])), horizontalalignment='left') 
			plt.figtext(0.92, 0.60, "score:" + str(score), horizontalalignment='left')
			#close_plot(fig, ax, plot_name="PO_forward_runs/unconditioned/single_samples/tom/tom_run_and_find-"+str(int(time.time()))+".eps")
			close_plot(fig, ax, 
				plot_name="PO_forward_runs/conditioned/single_samples/tom/middle-most_model-P"+
				str(PS)+"-S"+str(SP)+"-"+str(int(time.time()))+".eps")

			

			#----------------------------------------------------------------------------


	# prev_weights = np.exp(scores - logsumexp(scores))
	# print ("previous weights:", prev_weights)

	
	
	log_normalizer = logsumexp(scores) - np.log(particles) # log avg weight
	weights = np.exp(scores - log_normalizer - np.log(particles))
	
	chosen_index = np.random.choice([i for i in range(particles)], particles, replace=True, p=weights) 
	#print "chosen_index", chosen_index
	traces = np.array(traces)

	return traces[chosen_index], log_normalizer


# # importance sampling using the score of trace
def importance_sampling(Q, particles):

	#pool = Pool(4) 
	#op = p.map(getattr_proxy(Q, "run_model"), ((),) * particles) 
	#print pool.map(Q.run_model, ((),) * particles)  
	#print(op) 
	from show_models import setup_plot, load_polygons, polygons_to_segments, close_plot
	from matplotlib import pyplot as plt
	import time
	

	traces = []
	scores = np.arange(particles)

	for i in xrange(particles):
		score, trace_vals = Q.run_model()
		traces.append(copy.deepcopy(trace_vals))
		scores[i] = score
		print ("SCORES:", scores)

		PS = 1
		SP = particles
		trace = trace_vals
		if len(trace["t_detected"])==0:
			print ("score:", score)
			print ("particle:", i, "detection count:", len(trace["t_detected"]))

		if True: #len(trace["t_detected"])>0:

			print ("---------------------------------------------------")
			print ("***************************************************")
			print ("---------------------------------------------------")
			print ("score:", score)
			print ("particle:", i, "detection count:", len(trace["t_detected"]))
			
			print ("---------------------------------------------------")
			print ("***************************************************")
			print ("---------------------------------------------------")
			locs = [[ 0.100, 1-0.900 ],[ 0.566, 1-0.854 ],[ 0.761, 1-0.665 ],
			[ 0.523, 1-0.604 ],[ 0.241, 1-0.660 ],[ 0.425, 1-0.591 ],
			[ 0.303, 1-0.429 ],[ 0.815, 1-0.402 ],[ 0.675, 1-0.075 ],
			[ 0.432, 1-0.098 ] ]
			poly_map = polygons_to_segments( load_polygons( "./paths.txt" ) )
			fig, ax = setup_plot(poly_map, locs)
		
			path = trace["my_plan"]
			t = trace["t"]
			for i in range(0, 39):
				ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
					'orange', linestyle=":", linewidth=1, label="Agent's Plan")
			
				
			# mark the runner at time t on its plan
			ax.scatter( path[t-1][0],  path[t-1][1] , s = 95, facecolors='none', edgecolors='orange')

			path = trace["other_plan"]

			# mark the runner at time t on its plan
			ax.scatter( path[t-1][0],  path[t-1][1] , s = 95, facecolors='none', edgecolors='blue')

			for i in range(0, 39):
				ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
					'blue', linestyle="--", linewidth=1, label="Other's Plan")
				if i in trace["t_detected"]:
					ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')
				
			

			plt.figtext(0.92, 0.85, "Values", horizontalalignment='left', weight="bold") 
			plt.figtext(0.92, 0.80, "R Start: " +str(trace["init_run_start"]), horizontalalignment='left') 
			#plt.figtext(0.92, 0.75, "A Goal: " +str(trace["init_run_goal"]), horizontalalignment='left') 
			plt.figtext(0.92, 0.75, "C Start: " +str(trace["other_run_start"]), horizontalalignment='left')
			#plt.figtext(0.92, 0.65, "B Goal: " +str(trace["other_run_goal"]), horizontalalignment='left') 
			plt.figtext(0.92, 0.70, "time step: " +str(trace["t"]), horizontalalignment='left') 
			plt.figtext(0.92, 0.65, "C detected R count: " +str(len(trace["t_detected"])), horizontalalignment='left') 
			plt.figtext(0.92, 0.60, "score:" + str(score), horizontalalignment='left')
			#close_plot(fig, ax, plot_name="PO_forward_runs/unconditioned/single_samples/tom/tom_run_and_find-"+str(int(time.time()))+".eps")
			close_plot(fig, ax, 
				plot_name="PO_forward_runs/conditioned/single_samples/tom/outer-most_model-P"+
				str(PS)+"-S"+str(SP)+"-"+str(int(time.time()))+".eps")


	#log_normalizer = logsumexp(scores) - np.log(particles) # log avg weight
	#weights = np.exp(scores - log_normalizer - np.log(particles))

	# unnormalized since we are going to add it to the outer log likelihood
	return traces, scores 


def metroplis_hastings(Q, particles):
	traces = []
	scores = np.arange(particles)

	current_score, current_trace = Q.run_model()
	for i in xrange(particles):

		# sample from proposal distribution
		proposed_score, proposed_trace = Q.run_model()

		# accept new sample
		if np.log(random.uniform(0,1)) < proposed_score - current_score:
			current_trace = copy.deepcopy(proposed_trace)
			current_score = proposed_score

	return current_trace





