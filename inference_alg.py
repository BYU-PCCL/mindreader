from program_trace import ProgramTrace
from random import randint
import random
import numpy as np
import copy
#from scipy.misc import logsumexp
from scipy.special import logsumexp
from tqdm import tqdm
#from multiprocessing import Pool



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

def add_conditions(conditions, Q, t):
	# assume runner was not seen before (for next inference step)
	conditions["detected_t_"+str(t)] = False
	conditions["init_run_x_"+str(t)] = Q.fetch("init_run_x_"+str(t))
	conditions["init_run_y_"+str(t)] = Q.fetch("init_run_y_"+str(t))
	return conditions



def sequential_monte_carlo(T, model, conditions, K):

	Q_T = np.arange(T)
	for t in xrange(T):
		Q = ProgramTrace(model)

		for name in conditions.keys:
			Q.condition(name, conditions[name])

		sampled_Q_ks = np.arange(K)
		Q_k_scores = np.arange(K)
		for k in xrange(K):
			Q_k, score = Q.run_model()
			Q_k_scores[k] = score
			sampled_Q_ks[k] = Q_k

		log_normalizer = logsumexp(Q_k_l_scores) - np.log(total_particles) 
		weights = np.exp(Q_k_l_scores - log_normalizer - np.log(total_particles))
		
		resampled_index = np.random.choice([i for i in range(total_particles)], total_particles, replace=True, p=weights) 

		sampled_Q = sampled_Q_k_ls[resampled_index]

		conditions = add_conditions(conditions, sampled_Q, t)
		Q_T[t] = sampled_Q

	return Q_T













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





