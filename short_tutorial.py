
from show_models import *
from adversarial_experiments import combine_all_into_heatmap

#**********************************************
# all of the models are located in team_runner.py
# The ones that we care about now are:
# 1. BasicRunnerPOM
# 2. TOMRunnerPOM
#


# FYI: other models were used for my collaborative experiments in the Fall 2017
# from team_runner import BasicRunner
# from team_runner import TOMCollabRunner


# ----BasicRunnerPOM : Middle most model with a built in innermost model
# ----TOMRunnerPOM: Outermost model, uses BasicRunnerPOM as its nested model


#**********************************************

def MIDDLEMOST_MODLE_EXAMPLE(locs, poly_map, isovist, PS=10, SP=32, inf_type="IS"):
	# True Chaser Observations
	#chaser_true = [[0.815, 0.598], [0.7759947201243943, 0.5791223433762703], [0.7369894402487884, 0.5602446867525407], [0.6979841603731827, 0.5413670301288112], [0.6589788804975769, 0.5224893735050815], [0.6199736006219713, 0.5036117168813519], [0.590715204108021, 0.48188403611330455], [0.6061149294313255, 0.44543870211582487], [0.5788500910385285, 0.41175775252718594], [0.5515852526457314, 0.37807680293854706], [0.5243204142529344, 0.3443958533499081]]
	
	# Create the Middlemost model
	runner_model = BasicRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist, mode="advers")
	Q = ProgramTrace(runner_model)
	
	# Since this is the middlemost model - it's in the perspective of the runner
	# this means that "other_run_start" and "other_run_goal" are the start and goal
	# locations for the chaser
	# and "run_start" and "run_goal" are the start and goal locations for the chaser

	Q.condition("run_start", 6) #  Runner's Start Loc
	#Q.condition("run_goal", 9) 
	Q.condition("other_run_start", 7) # Chaser's Start Loc
	#Q.condition("other_run_goal", 9)
	t = 11 # time step in the simulation
	Q.condition("t", t)
	

	for i in xrange(t):
		# this helps the model know that the runner hasn't been detection (before timestep 't')
		Q.condition("detected_t_"+str(i), False)

		# this is if you wanted to condition the chaser's true locations
		# Q.condition("other_run_x_"+str(i), chaser_true[i][0])
		# Q.condition("other_run_y_"+str(i), chaser_true[i][1])

	# this conditions future timesteps for detection = "False"
	# since the runner does not want to be detected
	for i in xrange(t, 26):
		Q.condition("detected_t_"+str(i), False)

	#To run inference, you can use Importance Sampling or a simple MH
	if inf_type == "IS":
		post_sample_traces = run_inference(Q, post_samples=PS, samples=SP) # samples = particle count
	if inf_type == "MH":
		post_sample_traces = run_inference_MH(Q, post_samples=PS, samples=SP)


	# to go through the posterior samples (of traces)

	paths = []
	other_paths = []
	for trace in post_sample_traces:
		path = trace["my_plan"]
		other_path = trace["other_plan"]

		# remove hovering points on path
		no_hover_path = []
		for pt in path:
			if abs(pt[0] - path[-2][0]) > .01:
				if abs(pt[1] - path[-2][1]) > 0.01:
					no_hover_path.append(pt)
		no_hover_path.append(path[-2])
		paths.append(no_hover_path)

		# remove hovering points on other path
		no_hover_other_path = []
		for pt in other_path:
			if abs(pt[0] - other_path[-2][0]) > .01:
				if abs(pt[1] - other_path[-2][1]) > 0.01:
					no_hover_other_path.append(pt)
		no_hover_other_path.append(other_path[-2])
		other_paths.append(no_hover_other_path)

	return paths, other_paths


	


def EX_1_SIMPLE_FULL_NESTED_MODEL(locs, poly_map, isovist, PS=10, SP=32, inf_type="IS"):
	#-----------run TOM partially observable model ------
	runner_model = BasicRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist, mode="advers")
	# pass in middle most model in tom_runner_model
	tom_runner_model = TOMRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist, 
		nested_model=runner_model, ps=1, sp=1, mode="advers") # adversarial model

	Q = ProgramTrace(tom_runner_model)
	
	# IMPORTANT: since the middle most model needs to be conditioned... 
	# because we assume that the chaser knows the starting location of the Runner
	# we set it as an observation first.
	# then inside the TOMRunnerPOM model, it conditiones the model (CAN BE REMOVED)
	Q.set_obs("other_run_start", 8) # "other" is the Runner, therefor this is Runner's Start Loc rv plot_name
	
	# then you can run the model
	tom_runner_model.run(Q)

def EX_2_INFERENCE_FULL_NESTED_MODEL(locs, poly_map, isovist, PS=10, SP=32, nested_PS=10, nested_SP=32, inf_type="IS"):
	#-----------run TOM partially observable model ------
	runner_model = BasicRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist, mode="advers")
	# pass in middle most model in tom_runner_model
	tom_runner_model = TOMRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist, 
		nested_model=runner_model, ps=nested_PS, sp=nested_SP, mode="advers") # adversarial model

	Q = ProgramTrace(tom_runner_model)
	
	# # since the middle most model needs to be conditioned... 
	# # because we assume that the chaser knows the starting location of the Runner
	# # we set it as an observation first.
	# # then inside the TOMRunnerPOM model, it conditiones the model 
	Q.set_obs("other_run_start", 8) # "other" is the Runner, therefor this is Runner's Start Loc rv name

	# conditioning the chaser's starting location
	Q.condition("init_run_start", 1)

	# conditioning the timestep	
	t = 8
	Q.condition("t", t)

	post_sample_traces = run_inference(Q, post_samples=PS, samples=SP)

	# for this example we just grab the first trace
	trace = post_sample_traces[0]

	fig, ax = setup_plot(poly_map, locs)

	print ""
	print "TIMESTEPS RUNNER WAS DETECTED:", trace["t_detected"]

	for i in trace["t_detected"]:
		intersections = trace["intersections-t-"+str(i)]
		# show last isovist
		if not intersections is None:
			intersections = np.asarray(intersections)
			intersections /= 500.0
			if not intersections.shape[0] == 0:
				patches = [ Polygon(intersections, True)]
				p = PatchCollection(patches, cmap=matplotlib.cm.Set2, alpha=0.2)
				colors = 100*np.random.rand(len(patches))
				p.set_array(np.array(colors))
				ax.add_collection(p)

	nested_post_samples = trace["nested_post_samples"]
	for nested_trace in nested_post_samples:
		#print "HERE"
		path = nested_trace["my_plan"]
		for i in range(t-1, 39):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'red', linestyle="-", linewidth=1, label="Other's Plan", alpha=0.2)
			if i in trace["t_detected"]:
				ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')

	path = trace["my_plan"]
	t = trace["t"]
	for i in range(0, t-1):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'orange', linestyle=":", linewidth=1, label="Agent's Plan")
	for i in range(t-1, 39):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'grey', linestyle=":", linewidth=1)


	# mark the runner at time t on its plan
	ax.scatter( path[t-1][0],  path[t-1][1] , s = 95, facecolors='none', edgecolors='orange')

	path = trace["other_plan"]

	for i in range(0, t-1):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'blue', linestyle="--", linewidth=1, label="Other's Plan")
		if i in trace["t_detected"]:
			ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='blue')
		# else:
		# 	ax.scatter( path[i][0],  path[i][1] , s = 30, facecolors='none', edgecolors='grey')
	# mark the runner at time t on its plan
	ax.scatter( path[t-1][0],  path[t-1][1] , s = 95, facecolors='none', edgecolors='blue')

	for i in range(t, 39):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'blue', linestyle="--", linewidth=1, label="Other's Plan")
		if i in trace["t_detected"]:
			ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')

	

	plt.figtext(0.92, 0.85, "Values", horizontalalignment='left', weight="bold") 
	plt.figtext(0.92, 0.80, "A Start: " +str(trace["init_run_start"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.75, "A Goal: " +str(trace["init_run_goal"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.70, "B Start: " +str(trace["other_run_start"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.65, "B Goal: " +str(trace["other_run_goal"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.60, "time step: " +str(trace["t"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.55, "A detected B count: " +str(len(trace["t_detected"])), horizontalalignment='left') 
	#close_plot(fig, ax, plot_name="PO_forward_runs/unconditioned/single_samples/tom/tom_run_and_find-"+str(int(time.time()))+".eps")
	plot_full_path="PO_forward_runs/conditioned/single_samples/tom/IS_tom_run_and_find-P"+str(PS)+"-S"+str(SP)+"-"+str(int(time.time()))+".eps"
	close_plot(fig, ax, plot_name=plot_full_path)
	
	#print "OUTPUT CAN BE SEEN AT:", plot_full_path



if __name__ == '__main__':
	#plot("test.eps")
	
	locs = None
	poly_map = None
	isovist = None

	# ------------- setup for map "paths" large bremen map ---------------
	
	# start and goal locations
	locs = [[ 0.100, 1-0.900 ],[ 0.566, 1-0.854 ],[ 0.761, 1-0.665 ],
		[ 0.523, 1-0.604 ],[ 0.241, 1-0.660 ],[ 0.425, 1-0.591 ],
		[ 0.303, 1-0.429 ],[ 0.815, 1-0.402 ],[ 0.675, 1-0.075 ],
		[ 0.432, 1-0.098 ] ]
	poly_map = polygons_to_segments( load_polygons( "./paths.txt" ) )
	isovist = i.Isovist( load_isovist_map() )



	#plots the map and the locations if said so in the function
	#plot(poly_map, plot_name="testing_advers.eps", locs=locs)



	####################################################################################
	# EXAMPLE 0: middlemost model - show on heatmap
	####################################################################################
	print "RUNNING MIDDLEMOST MODEL EXAMPLE...DISPLAYS 2 HEATMAPS - ONE FOR CHASER AND ANOTHER FOR RUNNER"
	path, other_path = MIDDLEMOST_MODLE_EXAMPLE(locs, poly_map, isovist, PS=1, SP=1, inf_type="IS")
	combine_all_into_heatmap(path, other_path, poly_map=poly_map, locs=locs, PS=1, SP=1)
	print "DONE."

	# PS tells the nested inference how many samples from the posterior you want
	# and SP tells the nested inference how many particles to use
	
	####################################################################################
	# EXAMPLE 1: we simply run the model - no display
	####################################################################################
	print "RUNNING SIMPLE FULL NESTED MODEL - DOES NOT DISPLAY ANYTHING..."
	EX_1_SIMPLE_FULL_NESTED_MODEL(locs, poly_map, isovist, PS=1, SP=1, inf_type="IS")
	print "DONE."

	####################################################################################
	# EXAMPLE 2: we run inference over the model - show display on plots
	####################################################################################
	# XXX: CHANGE PS AND SP for outer inference, and nested_PS, nested_SP for inner inference
	# XXX: PS: samples from the posterior
	# XXX: SP: particle count for the inference algorithms
	print "RUNNING FULL NESTED MODEL EXAMPLE - DISPLAYS SINGLE PLOT OF OUTCOME..."
	EX_2_INFERENCE_FULL_NESTED_MODEL(locs, poly_map, isovist, PS=1, SP=1, nested_PS=1, nested_SP=1, inf_type="IS")
	print "DONE."
	# you can look a lot at Adversarial_experiments.py
	# ALSO at show_models.py (however it's messy because it invovles collaborative experiments.)



