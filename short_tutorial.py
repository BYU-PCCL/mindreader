
from show_models import *

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
	chaser_true = [[0.815, 0.598], [0.7759947201243943, 0.5791223433762703], [0.7369894402487884, 0.5602446867525407], [0.6979841603731827, 0.5413670301288112], [0.6589788804975769, 0.5224893735050815], [0.6199736006219713, 0.5036117168813519], [0.590715204108021, 0.48188403611330455], [0.6061149294313255, 0.44543870211582487], [0.5788500910385285, 0.41175775252718594], [0.5515852526457314, 0.37807680293854706], [0.5243204142529344, 0.3443958533499081]]
	
	# Create the Middlemost model
	runner_model = BasicRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist, mode=mode)
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
	runner_goals = []
	chaser_goals = []

	for trace in post_sample_traces:
		path = trace["my_plan"] # the runner's plan
		other_path = trace["other_plan"] # the chaser's plan
		runner_goals.append(trace["run_goal"])
		chaser_goals.append(trace["other_run_goal"]) 



def OUTERMOST_MODEL_EXAMPLE(locs, poly_map, isovist, PS=10, SP=32, inf_type="IS"):
	#-----------run TOM partially observable model ------
	runner_model = BasicRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist)
	# pass in middle most model in tom_runner_model
	tom_runner_model = TOMRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist, 
		nested_model=runner_model, ps=5, sp=32, model="advers") # adversarial model



	Q = ProgramTrace(tom_runner_model)

	Q.condition("init_run_start", 1) # Chaser Start Loc

	# since the middle most model needs to be conditioned... 
	# because we assume that the chaser knows the starting location of the Runner
	# we set it as an observation first.
	# then inside the TOMRunnerPOM model, it conditiones the model 
	Q.set_obs("other_run_start", 8) # other is the Runner, Runner Start Loc
	t = 8
	Q.condition("t", t)

	my_plan = [[0.56599999999999995, 0.14600000000000002], [0.55556809280738129, 0.18721437976543853], [0.5445341110551255, 0.23050610975954094], [0.53610663650986889, 0.27570707994484339], [0.51838817809125204, 0.31364677053850654], [0.51518717782727763, 0.35497127647386223], [0.55530733907318341, 0.38068397671159959], [0.56603426039645688, 0.420586401074451], [0.57876353334580199, 0.46649009155864785], [0.5894309668616603, 0.50360276514056979]]
	other_plan = [[0.67500000000000004, 0.92500000000000004], [0.65595794760796733, 0.88364953074871666], [0.63593639118326906, 0.84460049053946695], [0.632454561514043, 0.80177855079821769], [0.62322367768018871, 0.76592206121988504], [0.61070805265174255, 0.72506794767846527], [0.58606327906428324, 0.68850497091510821], [0.56369666918811434, 0.65184792821656079], [0.53867971062239195, 0.61822274529712795], [0.51418992415138365, 0.57765462476393603]]
	detections = [False, False, False, False, False, False, False, True]

	post_sample_traces = run_inference(Q, post_samples=PS, samples=SP)

	trace = post_sample_traces[0]



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



	MIDDLEMOST_MODLE_EXAMPLE(locs, poly_map, isovist, PS=10, SP=32, inf_type="IS")

	# you can look a lot at Adversarial_experiments.py
	# ALSO at show_models.py (however it's messy because it invovles collaborative experiments.)



