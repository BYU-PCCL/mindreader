import isovist as i
import program_trace as p
import runner as r
from methods import load_isovist_map, scale_up, direction
from my_rrt import *
import copy
from scipy.misc import logsumexp
import cPickle
from multiprocessing import Pool
from tqdm import tqdm
import planner
import time
from random import randint
#import seaborn
#seaborn.set_style(style='white')

# XXX in order for the entruder to do theory of mind
	# to intercept the runner agent, the entruder must do 
	# goal inference.

class Chaser(object):
	def __init__(self, isovist=None, locs=None, seg_map=[None,None,None,None]):
		self.isovist = isovist
		self.locs = locs
		rx1,ry1,rx2,ry2 = seg_map
		self.plan_path = lambda start_loc, goal_loc: planner.run_rrt_opt( start_loc, goal_loc, rx1,ry1,rx2,ry2 )
		#self.time_limit = 200
		self.show = False
		# initialize model
		self.runner_model = create_runner_model(seg_map=seg_map, locs=locs, isovist=isovist)
		

	def run(self, Q):
		t = Q.choice( p=1.0/29*np.ones((1,29)), name="t" )

		# current location of chaser (at time 't')
		curr_loc = [Q.get_obs("enf_x_"+str(t)), Q.get_obs("enf_y_"+str(t))]

		# randomly choose a start and goal for runner
		cnt = len(self.locs)
		int_start_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="int_start" )
		int_goal_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="int_goal" )

		# create empty trace using model
		q = p.ProgramTrace(self.runner_model)
		# condition q on Q 
		q = self.condition_q(Q, q)

		# run inference to get intruder's expected next step
		post_sample_traces = run_inference(q, post_samples=6, samples=5) # 10, 5
		runner_exp_next_step = expected_next_step(post_sample_traces, "int_plan")

		Q.keep("q", q)
		Q.keep("runner_post_sample_traces", post_sample_traces)
		Q.keep("runner_exp_next_step", runner_exp_next_step)

		# plan path toward anticipated location (t+1) for runner
		enf_plan = self.plan_path(np.atleast_2d( curr_loc), np.atleast_2d( runner_exp_next_step))
		Q.keep("enf_plan", enf_plan)

		# set up the enforcer view (forward vector, fv) for the next step
		cur_enf_loc = scale_up(curr_loc)
		next_enf_loc = scale_up(enf_plan[t+1])
		fv = direction(next_enf_loc, cur_enf_loc)

		intersections = self.isovist.GetIsovistIntersections(next_enf_loc, fv)

		# does the enforcer see me at time 't'
		runner_next_loc = scale_up(runner_exp_next_step)
		will_runner_be_seen = self.isovist.FindIntruderAtPoint( runner_next_loc, intersections )
		detected_prob = 0.999*will_runner_be_seen + 0.001*(1-will_runner_be_seen) # ~ flip(seen*.999 + (1-seen*.001)
		
		# XXX should consider seeing if the enforcer will see the intruder 
		# before reaching its simulated goal location, not just in the next step
		# down side: it would need more isovist calculations at each step
		runner_detected = Q.flip( p=detected_prob, name="int_detected" )


	def condition_q(self, Q, q):
		t = Q.fetch("t")
		q.condition("t", t)
		q.condition("enf_start", Q.get_obs("enf_start"))
		for i in xrange(1,t+1):
			q.condition("enf_x_"+str(i), Q.get_obs("enf_x_"+str(i)))
			q.condition("enf_y_"+str(i), Q.get_obs("enf_y_"+str(i)))
		q.condition("int_start", Q.fetch("int_start"))
		q.condition("int_goal", Q.fetch("int_goal"))
		q.condition("int_detected", False)

		for i in xrange(t):
			q.cache["enf_intersections_t_"+str(i)] = Q.cache["enf_intersections_t_"+str(i)] 

		return q


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


def create_runner_model(seg_map=None, locs=None, isovist=None):
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
	model = r.Runner(isovist=isovist, locs=locs, seg_map=seg_map)
	return model

def example_conditions(trace):
	#enf_locs = [[0.14263744, 0.12136674], [0.18804032, 0.14804696], [0.22508292, 0.18175865],
	#			[0.2630608, 0.20759783], [0.27751185, 0.25381817], [0.28647708, 0.3064716]]

	enf_locs = [[0.10000000000000001, 0.099999999999999978], [0.14263744, 0.12136674], [0.18804032, 0.14804696], [0.22508292, 0.18175865], [0.2630608, 0.20759783], 
	[0.27751185, 0.25381817], [0.28647708, 0.3064716], [0.31561698691448764, 0.34934704574305842], 
	[0.35098850256306374, 0.38377195776999806], [0.38263032945387726, 0.42667173804548703]]

	trace.set_obs("enf_start", 0)
	#trace.condition("enf_goal", 7)
	t = 9
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

def par_params(cnt, samples):
	params = ()
	for i in xrange(cnt):
		trace = example_conditions(p.ProgramTrace(create_runner_model()))
		params += ((trace, samples),)
	return params

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

def sampling_importance(trace, samples=3):
	traces = []
	scores = np.arange(samples)

	for i in xrange(samples):
		#deep copy is not working
		score, trace_vals = trace.run_model()
		traces.append(copy.deepcopy(trace_vals))
		scores[i] = score

	# get weight for score
	weights = np.exp(scores - logsumexp(scores))

	# normalize between 0 and 1
	weights = weights / sum(weights)

	# sample
	chosen_index = np.random.choice([i for i in range(samples)], p=weights)
	return traces[chosen_index]


def run_inference(trace, post_samples=5, samples=5):
	post_traces = []
	for i in  tqdm(xrange(post_samples)):
		post_sample_trace = sampling_importance(trace, samples=samples)
		post_traces.append(post_sample_trace)
	return post_traces


def run_inference_par(post_samples=5, samples=5):
	p = Pool( 6 )
	params = par_params(post_samples, samples)
	print params
	results = p.map( sampling_importance, params )
	return results

def expected_next_step(post_sample_traces, name):
	t = post_sample_traces[0]["t"]
	next_steps = []
	for sample_i, trace in enumerate(post_sample_traces):
		next_steps.append(trace[name][t+1])
	expected_step = list(np.mean(next_steps, axis=0))
	return expected_step

def expected_next_step_replanning(post_sample_traces, name):
	next_steps = []
	for sample_i, trace in enumerate(post_sample_traces):
		next_steps.append(trace[name][1])
	expected_step = list(np.mean(next_steps, axis=0))
	return expected_step

def test_chaser():

	# a single time step test
	model = create_chaser_model()
	intersection_cache = []

	# create empty trace using model
	Q = p.ProgramTrace(model)
	# set testing/example conditions in trace
	Q = example_conditions(Q)
	# run inference
	post_sample_traces = run_inference(Q, post_samples=1, samples=2) # 5, 3

	# XXX needs to plan a path to the expected next step of the runner
	# (in this case we would need the optimzed path to get a short distance to the goal)

	# we don't want to do goal inference on ourself. we don't need a path that matches
	# the past steps, we simply need to replan from where the agent currently is. 
	# the enforcer keeps on replanning until the intruder gets to its goal 
	# or detects him

	save_chaser_post_traces(post_sample_traces)


def save_chaser_post_traces(chaser_post_sample_traces, plot_name=None, runner_true_locs=None):
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

	# if not runner_true_locs is None:
	# 	for i in range( 0, len(runner_true_locs)-1):
	# 		ax.plot( [ path[i][0] * scale, path[i+1][0] * scale ], [ path[i][1] * scale, path[i+1][1] * scale], 'black', linestyle=":", linewidth=2)
		
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

	# create legend from custom artist/label lists
	lgd = ax.legend([enforcer_legend,runner_legend,next_step_runner_legend, 
		next_step_enforcer_legend, starting_legend, 
		runner_exp_next_legend, enforcer_plan_legend, enforcer_next_legend], 
		["C ", "C's Infer R Loc", "C's Infer R's Next", "C's Infer R's Infer C's Next", 
		"Starting Points", "C's Infer R's Exp Next", "C's Plan to Infer R's Exp Next", "C's Next"], 
		loc='upper center', 
		bbox_to_anchor=(1.15, 1), shadow=True, ncol=1, scatterpoints = 1)

	if plot_name is None:
		plot_name = str(int(time.time()))+".png"
	fig.savefig(plot_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
	#plt.show()

def run_simulation(locs, seg_map, isovist):
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
	for t in xrange(0,2):

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
		post_sample_traces = run_inference(Q, post_samples=1, samples=2)
		exp_next_step = expected_next_step_replanning(post_sample_traces, "enf_plan")
		# replan to its expected next step
		enf_plan = planner.run_rrt_opt( np.atleast_2d(chaser_locs[t]), 
			np.atleast_2d(exp_next_step), rx1,ry1,rx2,ry2 )
		enf_next_step = enf_plan[1]
		# store step made
		chaser_locs.append(enf_next_step)

		# check to see if the runner was detected
		fv = direction(scale_up(chaser_locs[t+1]), scale_up(chaser_locs[t]))
		intersections = isovist.GetIsovistIntersections(scale_up(chaser_locs[t+1]), fv)
		runner_true_next_loc = scale_up(runner_true_loc)
		runner_detected = isovist.FindIntruderAtPoint( runner_true_next_loc, intersections )

		# store intersections at time 't'
		intersection_cache.append(intersections)

		# plot behavior
		plot_name = None
		if runner_detected:
			print "Success: Runner Detected Before Goal"
			plot_name = str(sim_id)+"_t-"+str(t)+"_s-"+"T"+".png"
			return True
			break
		else:
			plot_name = str(sim_id)+"_t-"+str(t)+"_s-"+"F"+".png"
			print "searching..."
		
		save_chaser_post_traces(post_sample_traces, plot_name=plot_name, runner_true_locs=runner_true_locs)





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


	# TODO: for x simulations
	detection = run_simulation(locs, seg_map, isovist)



	# # initialize model
	# model = create_runner_model()
	# # create empty trace using model
	# trace = p.ProgramTrace(model)
	# # set testing/example conditions in trace
	# trace = example_conditions(trace)
	# # run inference
	# post_sample_traces = run_inference(trace, post_samples=10, samples=5)
	# # pickel post sample traces
	# cPickle.dump( post_sample_traces, open("./9-09-p10-s5-4.cp","w") )
	# #shot posterior samples
	# show_runner_post_traces(post_sample_traces)
	# print "DONE"






