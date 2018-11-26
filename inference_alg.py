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



def update_conditions(conditions, trace, t):
	# assume runner was not seen before (for next inference step)
	#new_conditions = copy.deepcopy(conditions)
	#new_conditions["detected_t_"+str(t)] = False
	new_conditions = {}
	new_conditions["t"] = conditions["t"]+1
	new_conditions["init_run_start"] = 4
	new_conditions["other_run_start"] = 8
	T=30
	for i in xrange(t, T-1):
		new_conditions["detected_t_"+str(i)] = True
	return new_conditions

def update_observations(observations, trace, t):
	new_observations = copy.deepcopy(observations)
	new_observations["init_run_x_"+str(t)] = trace["init_run_x_"+str(t)]
	new_observations["init_run_y_"+str(t)] = trace["init_run_y_"+str(t)]
	return new_observations

# XXX SOOOO WRONG - old attempt
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
	#print trace
	return (mean, trace)

import sys
import pickle
from my_rrt import *

# We use K different chasers and continue them through time
#params = ((model, observations, conditions),)*K # K different params
def sequential_monte_carlo_par(params, K, T=30, t_start=1, get_data=False):

	if not get_data:
		file_id = str(int(time.time()))
		directory = "PO_forward_runs/conditioned/SMC_simulations/resamp-sim-"+file_id
		os.mkdir(directory)
		print ("Directory:", directory)
		#model.set_dir(directory)
		for p in params:
			p[0].set_dir(directory)
			p[0].set_id(file_id)

	# KQ_T = []
	# KQ_T_scores = []
	for t in xrange(t_start, T-1):
		
		sampled_Q_ks = None
		Q_k_scores = None
		resampled_Q_ks = None

		p = Pool(10)

		results = p.map(single_run_model_sarg, params)
		p.close()
		p.join() 

		Q_k_scores = np.array((zip(*results))[0])
		sampled_Q_ks = np.array((zip(*results))[1])

		#raw_input()

		log_normalizer = logsumexp(Q_k_scores) - np.log(K) 
		weights = np.exp(Q_k_scores - log_normalizer - np.log(K))

		resampled_indexes = np.random.choice([i for i in range(K)], K, replace=True, p=weights) 

		resampled_Q_ks = sampled_Q_ks[resampled_indexes]

		params = tuple(np.array(params)[resampled_indexes])

		KQ_save = {}
		KQ_save["log_normalizer"] = log_normalizer
		KQ_save["norm_weights"] = weights
		KQ_save["t"] = t
		KQ_save["K"] = K
		KQ_save["orig_scores"] = Q_k_scores
		k_ = 1
		# For each K trace
		for Q_trace in sampled_Q_ks:
			small_Qk = {}
			# Get L trace information
			small_Qk["all_ql_scores"] = Q_trace["all_ql_scores"]
			small_Qk["all_Qls_scores"] = Q_trace["all_Qls_scores"]

			# Get L imaginary chaser information
			combined_single_K_and_L_traces = Q_trace["all_Qls_traces"]
			L_imaginary_chaser_plans = []
			for Ql_trace in combined_single_K_and_L_traces:
				runner_trace = Ql_trace["q_trace"]
				imaginary_chaser_plan = runner_trace["other_plan"]
				L_imaginary_chaser_plans.append(imaginary_chaser_plan)

			small_Qk["L_imaginary_chaser_plans"] = L_imaginary_chaser_plans


			# details of kth trace
			small_Qk["mean"] = Q_trace["mean"]
			small_Qk["my_plan"] = Q_trace["my_plan"]
			small_Qk["t_detected_list"] = Q_trace["t_detected"]
			small_Qk["other_plan"] = Q_trace["other_plan"]
			
			KQ_save["orig-"+str(k_)] = small_Qk
			k_ +=1

		k_ = 1
		for Q_trace in resampled_Q_ks:
			small_Qk = {}
			small_Qk["all_ql_scores"] = Q_trace["all_ql_scores"]
			small_Qk["all_Qls_scores"] = Q_trace["all_Qls_scores"]

			# Get L imaginary chaser information
			combined_single_K_and_L_traces = Q_trace["all_Qls_traces"]
			L_imaginary_chaser_plans = []
			for Ql_trace in combined_single_K_and_L_traces:
				runner_trace = Ql_trace["q_trace"]
				imaginary_chaser_plan = runner_trace["other_plan"]
				L_imaginary_chaser_plans.append(imaginary_chaser_plan)

			small_Qk["L_imaginary_chaser_plans"] = L_imaginary_chaser_plans


			small_Qk["mean"] = Q_trace["mean"]
			small_Qk["my_plan"] = Q_trace["my_plan"]
			small_Qk["t_detected_list"] = Q_trace["t_detected"]
			small_Qk["other_plan"] = Q_trace["other_plan"]
			
			KQ_save["resamp-"+str(k_)] = small_Qk
			k_ +=1
		
		if not get_data:
			pickle.dump( KQ_save, open( directory+"/"+file_id+"_t-"+str(t)+".p", "wb" ))
		#pickle.dump( [Q_k_scores,sampled_Q_ks,resampled_indexes], open( directory+"/"+file_id+"_t-"+str(t)+"-LARGE.p", "wb" ))

		# resample Ks

		updated_params = []
		i = 0
		for p in params:
			conditions = update_observations(p[1], resampled_Q_ks[i], t)
			observations = update_conditions(p[2], resampled_Q_ks[i], t)
			updated_params.append((p[0],conditions, observations))
			#plot_outermost_sample(sampled_Q_ks[i], Q_k_scores[i], directory, t, i)
			i+=1
		
		params = tuple(updated_params)

	if get_data:
		return Q_k_scores, sampled_Q_ks


	#pickle.dump( [K, KQ_T, KQ_T_scores], open( directory+"/"+file_id+".p", "wb" ))
	# return KQ_T, KQ_T_scores #detection_probabilities

# this is SMC where K is a single param and we resample and choose one
# chaser sample to continue in time
# def special_sequential_monte_carlo_par(T, model, conditions, observations, K):

# 	file_id = str(int(time.time()))
# 	directory = "PO_forward_runs/conditioned/SMC_simulations/sim-"+file_id
# 	os.mkdir(directory)
# 	model.set_dir(directory)
# 	detection_probabilities = [0.0]
# 	Q_T = []
# 	for t in tqdm(xrange(1, T-1)):
		
# 		sampled_Q_ks = []

# 		p = Pool(10)
# 		params = ((model, observations, conditions),)*K # K different params
# 		results = p.map(single_run_model_sarg, params)
		
# 		Q_k_scores = np.array((zip(*results))[0])
# 		sampled_Q_ks = np.array((zip(*results))[1])

# 		log_normalizer = logsumexp(Q_k_scores) - np.log(K) 
# 		weights = np.exp(Q_k_scores - log_normalizer - np.log(K))
# 		print ("outer weights:", weights, "log norm:", log_normalizer)
		
# 		resampled_indexes = np.random.choice([i for i in range(K)], K, replace=True, p=weights) 

# 		resampled_Qs = sampled_Q_ks[resampled_indexes]

# 		sampled_Q_trace = resampled_Qs[np.random.randint(0,K)] # compute K params and continue (ea obs and cond)
# 		plot_outermost_sample(sampled_Q_trace, log_normalizer, directory, t)
		
# 		Q_T.append(sampled_Q_trace)


# 		conditions = update_conditions(conditions, sampled_Q_trace, t)
# 		observations = update_observations(observations, sampled_Q_trace, t)
# 		detection_probabilities.append(get_prob_detection_t(sampled_Q_trace, K))
# 		print ("detection_probabilities", detection_probabilities)

# 	# plot probability
# 	fig = plt.figure(1)
# 	fig.clf()
# 	ax = fig.add_subplot(1, 1, 1)
# 	ax.plot(list(xrange(T-1)), detection_probabilities, 'bo')
# 	ax.plot(list(xrange(T-1)), detection_probabilities, 'b--')
# 	ax.set_xlabel("time step")
# 	ax.set_ylabel("probability of detection")
# 	fig.savefig(directory+"/detection-probs.eps", bbox_inches='tight')

# 	return Q_T, detection_probabilities




# def sequential_monte_carlo(T, model, conditions, observations, K):

# 	file_id = str(int(time.time()))
# 	directory = "PO_forward_runs/conditioned/SMC_simulations/sim-"+file_id
# 	os.mkdir(directory)
# 	model.set_dir(directory)
# 	detection_probabilities = [0.0]
# 	Q_T = []
# 	for t in tqdm(xrange(1, T-1)):
# 		Q = ProgramTrace(model)

# 		for name in observations.keys():
# 			Q.set_obs(name, observations[name])
# 		for name in conditions.keys():
# 			Q.condition(name, conditions[name])

# 		sampled_Q_ks = []
# 		Q_k_scores = np.arange(K)
# 		for k in xrange(K):
# 			score, Q_trace_k = Q.run_model()
# 			Q_k_scores[k] = Q.fetch("mean")
# 			#plot_outermost_sample(Q_trace_k, Q_k_scores[k], directory, k)
# 			sampled_Q_ks.append(Q_trace_k)
# 		sampled_Q_ks = np.array(sampled_Q_ks)

# 		#raw_input()

# 		log_normalizer = logsumexp(Q_k_scores) - np.log(K) 
# 		weights = np.exp(Q_k_scores - log_normalizer - np.log(K))
# 		#print ("outer weights:", weights, "log norm:", log_normalizer)
		
# 		resampled_indexes = np.random.choice([i for i in range(K)], K, replace=True, p=weights) 

# 		resampled_Qs = sampled_Q_ks[resampled_indexes]

# 		sampled_Q_trace = resampled_Qs[np.random.randint(0,K)]
# 		#plot_outermost_sample(sampled_Q_trace, log_normalizer, directory, t)
		
# 		Q_T.append(sampled_Q_trace)

# 		conditions = update_conditions(conditions, sampled_Q_trace, t)
# 		observations = update_observations(observations, sampled_Q_trace, t)
# 		detection_probabilities.append(get_prob_detection_t(sampled_Q_trace, K))
# 		#print ("detection_probabilities", detection_probabilities)

# 	return Q_T, detection_probabilities