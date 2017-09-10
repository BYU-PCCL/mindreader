import isovist as i
import program_trace as p
import runner as r
from methods import load_isovist_map, scale_up, direction
from my_rrt import *
import copy
from scipy.misc import logsumexp
import cPickle
from multiprocessing import Pool
#import seaborn
#seaborn.set_style(style='white')

# XXX in order for the entruder to do theory of mind
	# to intercept the runner agent, the entruder must do 
	# goal inference.

class Chaser(object):
	def __init__(self, isovist=None, locs=None, seg_map=[None,None,None,None]):
		self.isovist = isovist
		#TODO: varify these locations
		self.locs = locs
		rx1,ry1,rx2,ry2 = seg_map
		#rx1,ry1,rx2,ry2 = polygons_to_segments( load_polygons( "./paths.txt" ) )
		self.plan_path = lambda start_loc, goal_loc: planner.run_rrt_opt( start_loc, goal_loc, rx1,ry1,rx2,ry2 )
		self.time_limit = 200
		self.show = False
		# initialize model
		self.intruder_model = create_intruder_model()
		

	def run(self, Q):
		t = Q.choice( p=1.0/29*np.ones((1,29)), name="t" )

		cnt = len(self.locs)
		enf_start_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="enf_start" )
		enf_goal_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="enf_goal" )
		
		enf_start = np.atleast_2d( self.locs[enf_start_i] )
		enf_goal = np.atleast_2d( self.locs[enf_goal_i] )
		path_noise = .003
		enf_plan = self.plan_path(enf_start, enf_goal)
		enf_noisy_plan = []
		for i in xrange(1, len(enf_plan)-1): 
			loc_x = Q.randn( mu=enf_plan[i][0], sigma=path_noise, name="enf_x_"+str(i) )
			loc_y = Q.randn( mu=enf_plan[i][1], sigma=path_noise, name="enf_y_"+str(i) )
			loc_t = [loc_x, loc_y]
			enf_noisy_plan.append(loc_t)
			
		enf_noisy_plan.append(enf_plan[-1])
		enf_loc = np.atleast_2d(enf_plan[t])

		# create empty trace using model
		q = p.ProgramTrace(self.intruder_model)
		# condition q on Q 
		q = condition_q(Q, q)
		# run inference to get intruder's expected next step
		post_sample_traces = run_inference(q, post_samples=10, samples =5)
		exp_next_step = intruder_expected_next_step(post_sample_traces)

		# XXX Need to make sure the runner wasn't seen in any of the previous time steps
		# already_detected = 0
		# for i in xrange(t):
		# 	intersections = Q.cache["enf_intersections_t_"+str(i)]# get enforcer's fv for time t
		# 	i_already_seen = self.isovist.FindIntruderAtPoint(my_noisy_plan[i], intersections)
		# 	if (i_already_seen):
		# 		detected_prob = 0.999*i_already_seen + 0.001*(1-i_already_seen) 

		# if not i_already_seen:
		# 	# set up the enforcer view (forward vector, fv) for the next step
		# 	cur_enf_loc = scale_up(enf_noisy_plan[t])
		# 	next_enf_loc = scale_up(enf_noisy_plan[t+1])
		# 	fv = direction(next_enf_loc, cur_enf_loc)
		# 	intersections = self.isovist.GetIsovistIntersections(next_enf_loc, fv)

		# 	# does the enforcer see me at time 't'
		# 	my_next_loc = scale_up(my_noisy_plan[t+1])
		# 	will_i_be_seen = self.isovist.FindIntruderAtPoint( my_next_loc, intersections )
		# 	detected_prob = 0.999*will_i_be_seen + 0.001*(1-will_i_be_seen) # ~ flip(seen*.999 + (1-seen*.001)
		

		# future_detection = Q.flip( p=detected_prob, name="int_detected" )


	def condition_q(Q, q):
		t = Q["t"]
		q.condition("t", t)
		trace.condition("enf_start", Q["enf_start"])
		for i in xrange(t+1):
			q.condition("enf_x_"+str(i), Q["enf_x_"+str(i)])
		q.condition("int_start", Q["int_start"])
		q.condition("int_goal", Q["int_goal"])
		q.condition("int_detected", False)

		for i in xrange(t+1):
			q.cache["enf_intersections_t_"+str(i)] = Q.cache["enf_intersections_t_"] 

		return q


def create_chaser_model():
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
	isovist = i.Isovist( load_isovist_map() )
	model = r.Chaser(isovist, locs, seg_map)
	return model


def create_runner_model():
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
	isovist = i.Isovist( load_isovist_map() )
	model = r.Runner(isovist, locs, seg_map)
	return model

def example_conditions(trace):
	#enf_locs = [[0.14263744, 0.12136674], [0.18804032, 0.14804696], [0.22508292, 0.18175865],
	#			[0.2630608, 0.20759783], [0.27751185, 0.25381817], [0.28647708, 0.3064716]]

	enf_locs = [[0.10000000000000001, 0.099999999999999978], [0.14263744, 0.12136674], [0.18804032, 0.14804696], [0.22508292, 0.18175865], [0.2630608, 0.20759783], 
	[0.27751185, 0.25381817], [0.28647708, 0.3064716], [0.31561698691448764, 0.34934704574305842], 
	[0.35098850256306374, 0.38377195776999806], [0.38263032945387726, 0.42667173804548703]]

	trace.condition("enf_start", 0)
	#trace.condition("enf_goal", 7)
	t = 9
	trace.condition("t", t)
	for i in xrange(0, t+1):
		trace.condition("enf_x_"+str(i), enf_locs[i][0])
		trace.condition("enf_y_"+str(i), enf_locs[i][1])

	trace.condition("int_start", 1)
	trace.condition("int_goal", 8)
	trace.condition("int_detected", False)

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
	for i in xrange(post_samples):
		post_sample_trace = sampling_importance(trace, samples=samples)
		post_traces.append(post_sample_trace)
	return post_traces


def run_inference_par(post_samples=5, samples=5):
	p = Pool( 6 )
	params = par_params(post_samples, samples)
	print params
	results = p.map( sampling_importance, params )
	return results

def intruder_expected_next_step(post_sample_traces):
	t = post_sample_traces[0]["t"]
	next_steps = []
	for sample_i, trace in enumerate(post_sample_traces):
		next_steps.append(trace["int_plan"][t+1])
	expected_step = list(np.mean(next_steps, axis=0))
	return expected_step

def show_post_traces(post_sample_traces):
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	#ax = fig.add_axes([0.1, 0.1, 0.6, 0.6])
	scale = 1

	for sample_i, trace in enumerate(post_sample_traces):
		# get time
		t = trace["t"]
		# plot enf_plan
		path = trace["enf_plan"]
		for i in range( 0, len(path)-1 ):
			#ax.plot( [ path[i][0] * scale, path[i+1][0] * scale ], [ path[i][1] * scale, path[i+1][1] * scale], 'blue' )
			if i <= t+1:
				ax.scatter( path[i][0] * scale, path[i][1]  * scale, color="navy", s = 3)
			else:
				break
		
		ax.scatter( path[t+1][0] * scale, path[t+1][1]  * scale, color="blue", s = 40, marker="x") #enforcer (t+1)
		ax.scatter( path[0][0] * scale, path[0][1]  * scale, color="green") #Start location
		ax.scatter( path[-1][0] * scale, path[-1][1] * scale, color = "red") #Intruder Inferred Goals
		ax.scatter( path[t][0] * scale, path[t][1] * scale, color = "blue", s = 55, marker="v") #Enforcer

		# plot intruder_plan
		path = trace["int_plan"]
		for i in range( 0, len(path)-1 ):
			ax.plot( [ path[i][0] * scale, path[i+1][0] * scale ], [ path[i][1] * scale, path[i+1][1] * scale], 'grey', alpha=0.7 )
			if i+1 == t:
				break		
		ax.scatter( path[t+1][0] * scale, path[t+1][1]  * scale, color="magenta", s = 40, marker="x") #Runner (t+1)

		ax.scatter( path[0][0] * scale, path[0][1]  * scale, color="green")
		#ax.scatter( path[0][0] * scale, path[0][1] * scale, color = "red")
		ax.scatter( path[t][0] * scale, path[t][1] * scale, color = "magenta", s = 45, marker = "D") #Runner

		exp_next_step = intruder_expected_next_step(post_sample_traces)
		ax.scatter( exp_next_step[0] * scale, exp_next_step[1] * scale, color = "gold", s = 50, marker = "o")

	# plot all of the destinations
	# for i in xrange(10):
	# 	ax.scatter( np.atleast_2d( self.locs[i] )[0,0] * scale, np.atleast_2d( self.locs[i] )[0,1]  * scale, color="red")

	# plot map
	x1,y1,x2,y2 = polygons_to_segments( load_polygons( "./paths.txt" ) )
	for i in xrange(x1.shape[0]):
		ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'black' )

	plt.ylim((0,scale))
	#savefig('p5-s5-ex1.png')
	chartBox = ax.get_position()
	ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*1, chartBox.height])

	enforcer_legend = plt.Line2D([0,0],[0,1], color='blue', marker='v', linestyle='')
	runner_legend = plt.Line2D([0,0],[0,1], color='magenta', marker='D', linestyle='')
	next_step_runner_legend = plt.Line2D([0,0],[0,1], color='magenta', marker='x', linestyle='')
	next_step_enforcer_legend = plt.Line2D([0,0],[0,1], color='blue', marker='x', linestyle='')
	starting_legend = plt.Line2D([0,0],[0,1], color='green', marker='o', linestyle='')
	inference_legend = plt.Line2D([0,0],[0,1], color='red', marker='o', linestyle='')

	#Create legend from custom artist/label lists
	lgd = ax.legend([enforcer_legend,runner_legend,next_step_runner_legend, next_step_enforcer_legend, starting_legend, inference_legend], ["Enforcer", "Runner", "Runner Next Step", "Enforcer Next Step", "Starting Points", "Infered Goal"], loc='upper center', bbox_to_anchor=(1.15, 1), shadow=True, ncol=1, scatterpoints = 1)
	# lgd = ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1), shadow=True, ncol=1, scatterpoints = 1)

	
	fig.savefig('plot.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	# plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1, bbox_inches="tight")
	#plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
	plt.show()

def start_count_down():
	#currently always begins the same location
	enf_locs = [[ 0.100, 1-0.900 ]]
	model = create_chaser_model()

	Q = p.ProgramTrace(model)
	Q.condition("enf_start", 0)
	trace.condition("t", 0)

	for t in xrange(30):
		#condition past chaser location
		for j in xrange(0, t+1):
			trace.condition("enf_x_"+str(j+1), enf_locs[j][0])
			trace.condition("enf_y_"+str(j+1), enf_locs[j][1])


	# find foward vectors and cache the intersects 
	next_enf_loc = scale_up(enf_locs[i])
	cur_enf_loc = scale_up(enf_locs[i-1])
	fv = direction(next_enf_loc, cur_enf_loc)
	intersections = trace.model.isovist.GetIsovistIntersections(cur_enf_loc, fv)
	intersection_cache.append(intersections)

if __name__ == '__main__':

	# XXX This is testing the runner model. We can view samples from the prior
	# conditioned [on the variable list below]

	# initialize model
	model = create_runner_model()
	# create empty trace using model
	trace = p.ProgramTrace(model)
	# set testing/example conditions in trace
	trace = example_conditions(trace)
	# run inference
	#post_sample_traces = run_inference(trace, post_samples=10, samples =5)
	post_sample_traces = run_inference(trace, post_samples=10, samples =5)
	#post_sample_traces = run_inference_par(post_samples=5, samples=5)
	# pickel post sample traces
	cPickle.dump( post_sample_traces, open("./9-09-p10-s5-4.cp","w") )
	#shot posterior samples
	show_post_traces(post_sample_traces)
	print "DONE"






