from enforcer import *
from methods import load_isovist_map, scale_up, direction, dist, detect, load_segs, get_clear_goal,point_in_obstacle
from my_rrt import *
import isovist as i
from random import randint
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def example_conditions(trace):
	t = 9
	enf_locs = [[0.10000000000000001, 0.099999999999999978], [0.14263744, 0.12136674], [0.18804032, 0.14804696], [0.22508292, 0.18175865], [0.2630608, 0.20759783], 
	[0.27751185, 0.25381817], [0.28647708, 0.3064716], [0.31561698691448764, 0.34934704574305842], 
	[0.35098850256306374, 0.38377195776999806], [0.38263032945387726, 0.42667173804548703]]
	trace.set_obs("enf_start", 0)
	#trace.condition("enf_goal", 7)
	trace.condition("t", t)
	for i in xrange(0, t+1):
		trace.set_obs("enf_x_"+str(i), enf_locs[i][0])
		trace.set_obs("enf_y_"+str(i), enf_locs[i][1])
	#trace.condition("int_start", 1)
	#trace.condition("int_goal", 8) # used to be 8 
	trace.condition("int_detected", True)
	try:
		intersection_cache = cPickle.load( open("./enf_int_cache.cp") )
	except:
		intersection_cache = precompute_and_save_intersections(t, enf_locs, trace)

	for i in xrange(t+1):
		trace.cache["enf_intersections_t_"+str(i)] = intersection_cache[i]
	return trace

def create_chaser_model(seg_map=None, locs=None, isovist=None):
	if locs is None:
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

	if seg_map is None:
		seg_map = polygons_to_segments( load_polygons( "./paths.txt" ) )
	if isovist is None:
		isovist = i.Isovist( load_isovist_map() )
	model = Chaser(isovist=isovist, locs=locs, seg_map=seg_map)
	return model

def precompute_and_save_intersections(t, enf_locs, trace):
		print "Precomputing intersections..."
		intersection_cache = []
		for i in xrange(t+1):
			next_enf_loc = scale_up(enf_locs[i])
			cur_enf_loc = scale_up(enf_locs[i-1])
			fv = direction(next_enf_loc, cur_enf_loc)
			intersections = trace.model.isovist.GetIsovistIntersections(cur_enf_loc, fv)
			intersection_cache.append(intersections)

		cPickle.dump( intersection_cache, open("./enf_int_cache.cp","w") )
		print "done!, Interction Count:", len(intersection_cache)
		return intersection_cache

def test_chaser():
	# a single time step test
	model = create_chaser_model()
	# create empty trace using model
	Q = p.ProgramTrace(model)
	# set testing/example conditions in trace
	Q = example_conditions(Q)
	# run inference
	post_sample_traces = run_inference(Q, post_samples=1, samples=2) 

	# XXX needs to plan a path to the expected next step of the runner
	# (in this case we would need the optimzed path to get a short distance to the goal)

	# we don't want to do goal inference on ourself. we don't need a path that matches
	# the past steps, we simply need to replan from where the agent currently is. 
	# the enforcer keeps on replanning until the intruder gets to its goal 
	# or detects him

	save_chaser_post_traces(post_sample_traces)


def save_chaser_post_traces(chaser_post_sample_traces, plot_name=None, runner_true_locs=None, intersections=None):
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	scale = 1

	# plot map
	x1,y1,x2,y2 = polygons_to_segments( load_polygons( "./paths.txt" ) )
	for i in xrange(x1.shape[0]):
		ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'black' )

	for sample_i, trace in enumerate(chaser_post_sample_traces):
		# get time
		t = trace["t"]
		# plot enf_plan
		path = trace["enf_plan"]
		for i in range( 0, len(path)-1):
			ax.plot( [ path[i][0] * scale, path[i+1][0] * scale ], [ path[i][1] * scale, path[i+1][1] * scale], 'black', linestyle=":", linewidth=2)
			# if i <= t+1:
			# 	ax.scatter( path[i][0] * scale, path[i][1]  * scale, color="darkslategray", s = 3)
			# else:
			# 	break
		
		
		ax.scatter( path[0][0] * scale, path[0][1]  * scale, color="green") #Start location
		#ax.scatter( path[-1][0] * scale, path[-1][1] * scale, color = "red") 
		ax.scatter( path[0][0] * scale, path[0][1] * scale, color = "darkslategray", s = 55, marker="v") #Enforcer
		enf_true_next_x = path[1][0]
		enf_true_next_y = path[1][1]
		#chaser_exp_next_step = expected_next_step(chaser_post_sample_traces,"enf_plan")
		#ax.scatter( chaser_exp_next_step[0] * scale, chaser_exp_next_step[1] * scale, color = "darkcyan", s = 50, marker = "o")

		#*****************************************************************************************************
		# ploting inference information over the chaser's model of the intruder's model of the chaser's model
		#*****************************************************************************************************
		run_post_sample_traces = trace["runner_post_sample_traces"]
		for sample_i, r_trace in enumerate(run_post_sample_traces):
			# plot enf_plan
			path = r_trace["enf_plan"]
			for i in range( 0, len(path)-1 ):
				ax.plot( [ path[i][0] * scale, path[i+1][0] * scale ], [ path[i][1] * scale, path[i+1][1] * scale], 'blue' ,linestyle="--", alpha=0.5)
				if i <= t+1:
					ax.scatter( path[i][0] * scale, path[i][1]  * scale, color="navy", s = 3)
				else:
					break
			
			ax.scatter( path[t+1][0] * scale, path[t+1][1]  * scale, color="blue", s = 40, marker="x", linewidths=1) #enforcer (t+1)
			
			#ax.scatter( path[-1][0] * scale, path[-1][1] * scale, color = "red") #Intruder Inferred Goals of the Enforcer
			ax.scatter( path[t][0] * scale, path[t][1] * scale, color = "blue", s = 55, marker="v") #Enforcer
			ax.scatter( path[0][0] * scale, path[0][1]  * scale, color="green") #Start location

			# plot intruder_plan
			path = r_trace["int_plan"]
			for i in range( 0, len(path)-1 ):
				if t == 0:
					break
				ax.plot( [ path[i][0] * scale, path[i+1][0] * scale ], [ path[i][1] * scale, path[i+1][1] * scale], 'grey', alpha=0.6, linestyle="-")
				if i+1 == t:
					break		
			ax.scatter( path[t+1][0] * scale, path[t+1][1]  * scale, color="magenta", s = 40, marker="x",linewidths=1) #Runner (t+1)

			
			#ax.scatter( path[0][0] * scale, path[0][1] * scale, color = "red")
			ax.scatter( path[t][0] * scale, path[t][1] * scale, color = "magenta", s = 45, marker = "D") #Runner

			exp_next_step = expected_next_step(run_post_sample_traces, "int_plan")
			ax.scatter( exp_next_step[0] * scale, exp_next_step[1] * scale, color = "darkorchid", s = 50, marker = "o")
			ax.scatter( path[0][0] * scale, path[0][1]  * scale, color="green")

		ax.scatter( enf_true_next_x * scale, enf_true_next_y  * scale, color="darkturquoise", s = 80, marker="v") #enforcer (t+1)
		#*******************************************************

	if not runner_true_locs is None:
		for i in range( 0, len(runner_true_locs)-1):
			ax.plot( [ runner_true_locs[i][0] * scale, runner_true_locs[i+1][0] * scale ], [ runner_true_locs[i][1] * scale, runner_true_locs[i+1][1] * scale], 'orange', linestyle="-.", linewidth=2)
	runnert_last = runner_true_locs[-1]
	ax.scatter( runnert_last[0] * scale, runnert_last[1]  * scale, color="orange", s = 80, marker="D")
	
	# show last isovist
	if not intersections is None:
		patches = [ Polygon(intersections, True)]
		p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
		colors = 100*np.random.rand(len(patches))
		p.set_array(np.array(colors))
		ax.add_collection(p)
	# plot all of the destinations
	# for i in xrange(10):
	# 	ax.scatter( np.atleast_2d( self.locs[i] )[0,0] * scale, np.atleast_2d( self.locs[i] )[0,1]  * scale, color="red")


	plt.ylim((0,scale))
	chartBox = ax.get_position()
	ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*1, chartBox.height])

	enforcer_legend = plt.Line2D([0,0],[0,1], color='blue', marker='v', linestyle='')
	runner_legend = plt.Line2D([0,0],[0,1], color='magenta', marker='D', linestyle='')
	next_step_runner_legend = plt.Line2D([0,0],[0,1], color='magenta', marker='x', linestyle='')
	next_step_enforcer_legend = plt.Line2D([0,0],[0,1], color='blue', marker='x', linestyle='')
	starting_legend = plt.Line2D([0,0],[0,1], color='green', marker='o', linestyle='')
	runner_exp_next_legend = plt.Line2D([0,0],[0,1], color='darkorchid', marker='o', linestyle='')
	enforcer_plan_legend = plt.Line2D([0,0],[0,1], color='black', linestyle=':', linewidth=2)
	enforcer_next_legend = plt.Line2D([0,0],[0,1], color='darkturquoise', marker='v', linestyle='')
	true_runner_legend = plt.Line2D([0,0],[0,1], color='orange', marker='D', linestyle='')
	true_runner_path_legend = plt.Line2D([0,0],[0,1], color='orange', linestyle='-.')

	# create legend from custom artist/label lists
	lgd = ax.legend([enforcer_legend,runner_legend,next_step_runner_legend, 
		next_step_enforcer_legend, starting_legend, 
		runner_exp_next_legend, enforcer_plan_legend, enforcer_next_legend,
		true_runner_legend, true_runner_path_legend], 
		["C ", "C's Infer R Loc", "C's Infer R's Next", "C's Infer R's Infer C's Next", 
		"Starting Points", "C's Infer R's Exp Next", "C's Plan to Infer R's Exp Next", 
		"C's Next", "True R", "True R Path"], 
		loc='upper center', 
		bbox_to_anchor=(1.15, 1), shadow=True, ncol=1, scatterpoints = 1)

	if plot_name is None:
		plot_name = str(int(time.time()))+".png"
	fig.savefig(plot_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
	#plt.show()

def run_simulation(locs, seg_map, isovist, polys, epolys):
	# simulation id
	sim_id = str(int(time.time()))
	# unpack 
	rx1,ry1,rx2,ry2 = seg_map
	
	intersection_cache = []
	# get random start location
	start_index = randint(0, len(locs)-1)
	chaser_start = locs[start_index]
	# begin tracking history of movements
	chaser_locs = [chaser_start]
	# create model of chaser
	chaser_model = create_chaser_model(seg_map=seg_map, locs=locs, isovist=isovist)

	# create model of runner
	# TODO ^
	# get runner's random start location (far enough from chaser])
	# TODO ^
	runner_start = locs[1]
	# get runner's random goal location (far enough from its start)
	runner_goal = locs[8]
	# store history of runner's locations
	runner_true_locs = [runner_start]

	#TEMP FOR TESTING (NAIVE RUNNER)
	runner_path = planner.run_rrt_opt( np.atleast_2d(runner_start), 
		np.atleast_2d(runner_goal), rx1,ry1,rx2,ry2 )
	runner_locs = runner_path

	# begin timer
	#for t in xrange(TIME_LIMIT):
	for t in xrange(0,11):

		# get true runner's location at time t + 1
		runner_true_loc = runner_locs[t+1]
		runner_true_locs.append(runner_true_loc)
		# TODO: replace with call to smart runner in another test case

		# check if the runner has reached its goal 
		if runner_true_loc == runner_goal:
			print "Failed: Runner Reached Goal"
			return False
			break

		# create empty trace
		Q = p.ProgramTrace(chaser_model)
		# condition the trace
		Q.condition("t", t)
		Q.set_obs("enf_start", start_index)
		Q.condition("int_detected", True)
		for pre_t in xrange(t+1):
			Q.set_obs("enf_x_"+str(pre_t), chaser_locs[pre_t][0])
			Q.set_obs("enf_y_"+str(pre_t), chaser_locs[pre_t][1])
			if pre_t < t:
				Q.cache["enf_intersections_t_"+str(pre_t)] = intersection_cache[pre_t]

		# run inference
		post_sample_traces = run_inference(Q, post_samples=3, samples=2)
		exp_next_step = expected_next_step_replanning(post_sample_traces, "enf_plan")

		if point_in_obstacle(exp_next_step, epolys):
			exp_next_step = get_clear_goal(chaser_locs[t], exp_next_step, polys)

		# replan to its expected next step
		enf_plan = planner.run_rrt_opt( np.atleast_2d(chaser_locs[t]), 
			np.atleast_2d(exp_next_step), rx1,ry1,rx2,ry2, just_need_step=True)

		if enf_plan is None:
			print "trying again"
			t -= 1
			continue
		enf_next_step = enf_plan[1]
		# store step made
		chaser_locs.append(enf_next_step)

		inters = None
		# check to see if the runner was detected
		runner_detected = False
		# only if he is close
		if dist(enf_next_step, runner_true_loc) <= .4:
			# add a bit of hearing by facing the chaser towards the runner
			fv = direction(scale_up(runner_true_loc), scale_up(enf_next_step))
			# check if runner can be detected
			runner_true_next_loc = scale_up(runner_true_loc)
			intersections = isovist.GetIsovistIntersections(scale_up(enf_next_step), fv)
			runner_detected = isovist.FindIntruderAtPoint( scale_up(runner_true_loc), intersections)
			inters = np.asarray(intersections)
			inters /= 500.0

		# store intersections at time 't' facing in the direction it stepped
		fv = direction(scale_up(chaser_locs[t+1]), scale_up(chaser_locs[t]))
		intersections = isovist.GetIsovistIntersections(scale_up(chaser_locs[t+1]), fv)
		intersection_cache.append(intersections)
		if inters is None:
			inters = np.asarray(intersections)
			inters /= 500.0


		# plot behavior
		plot_name = None
		if runner_detected:
			print "Success: Runner Detected Before Goal"
			plot_name = str(sim_id)+"_t-"+str(t)+"_s-"+"T"+".png"
			save_chaser_post_traces(post_sample_traces, plot_name=plot_name, runner_true_locs=runner_true_locs, intersections=inters)
			return True
			break
		else:
			plot_name = str(sim_id)+"_t-"+str(t)+"_s-"+"F"+".png"
			print "searching..."
		
		save_chaser_post_traces(post_sample_traces, plot_name=plot_name, runner_true_locs=runner_true_locs, intersections=inters)


if __name__ == '__main__':

	# XXX This is testing the runner model. We can view samples from the prior
	# conditioned [on the variable list below]

	#test_chaser()

	# setup
	locs = [[ 0.100, 1-0.900 ],[ 0.566, 1-0.854 ],[ 0.761, 1-0.665 ],
		[ 0.523, 1-0.604 ],[ 0.241, 1-0.660 ],[ 0.425, 1-0.591 ],
		[ 0.303, 1-0.429 ],[ 0.815, 1-0.402 ],[ 0.675, 1-0.075 ],
		[ 0.432, 1-0.098 ] ]
	seg_map = polygons_to_segments( load_polygons( "./paths.txt" ) )
	isovist = i.Isovist( load_isovist_map() )
	polys, epolys = load_segs()

	dets =[]
	# TODO: for x simulations
	for x in xrange(10):
		detection = run_simulation(locs, seg_map, isovist, polys, epolys)
		dets.append(detection)

	print dets 














