
import program_trace as p
import runner as r
from methods import load_isovist_map, scale_up, direction, load_segs, point_in_obstacle, get_clear_goal
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

class Chaser(object):
	def __init__(self, isovist=None, locs=None, seg_map=[None,None,None,None]):
		self.isovist = isovist
		self.locs = locs
		rx1,ry1,rx2,ry2 = seg_map
		#self.plan_path = lambda start_loc, goal_loc: planner.run_rrt_opt( start_loc, goal_loc, rx1,ry1,rx2,ry2 )
		self.show = False
		self.seg_map = seg_map
		# initialize model
		self.runner_model = create_runner_model(seg_map=seg_map, locs=locs, isovist=isovist)
		self.polys, self.epolys = load_segs()
	
	#run_naive
	def run_naive(self, Q):
		t = Q.choice( p=1.0/41*np.ones((1,41)), name="t" )

		# current location of chaser (at time 't')
		curr_loc = [Q.get_obs("enf_x_"+str(t)), Q.get_obs("enf_y_"+str(t))]

		# randomly choose a start and goal for runner
		cnt = len(self.locs)
		int_start_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="int_start" )
		int_goal_i = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="int_goal" )

		rx1,ry1,rx2,ry2 = self.seg_map
		int_plan = planner.run_rrt_opt( np.atleast_2d(self.locs[int_start_i]), 
			np.atleast_2d( self.locs[int_goal_i]), rx1,ry1,rx2,ry2, slow=True)
		
		# plan path toward anticipated location (t+1) for runner
		#enf_plan = self.plan_path(np.atleast_2d( curr_loc), np.atleast_2d( runner_exp_next_step))
		t_ = min(t+12, len(int_plan)-1)
		enf_plan = planner.run_rrt_opt( np.atleast_2d( curr_loc), np.atleast_2d( int_plan[t_]), rx1,ry1,rx2,ry2)
		Q.keep("enf_plan", enf_plan)
		Q.keep("int_plan", int_plan)

		# set up the enforcer view (forward vector, fv) for the next step
		cur_enf_loc = scale_up(curr_loc)
		next_enf_loc = scale_up(enf_plan[1])
		fv = direction(next_enf_loc, cur_enf_loc)

		intersections = self.isovist.GetIsovistIntersections(next_enf_loc, fv)

		# does the enforcer see me at time 't'
		runner_next_loc = scale_up(int_plan[t+1])
		will_runner_be_seen = self.isovist.FindIntruderAtPoint( int_plan[t+1], intersections )
		detected_prob = 0.999*will_runner_be_seen + 0.001*(1-will_runner_be_seen) # ~ flip(seen*.999 + (1-seen*.001)
		
		# XXX should consider seeing if the enforcer will see the intruder 
		# before reaching its simulated goal location, not just in the next step
		# down side: it would need more isovist calculations at each step
		runner_detected = Q.flip( p=detected_prob, name="int_detected" )


	def run(self, Q):
		t = Q.choice( p=1.0/41*np.ones((1,41)), name="t" )

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
		runner_exp_next_step, runner_exp_path = rand_expected_future_step(post_sample_traces, "int_plan")

		#if point_in_obstacle(runner_exp_next_step, self.epolys):
		#	runner_exp_next_step = get_clear_goal(curr_loc, runner_exp_next_step, self.polys)

		Q.keep("q", q.trace)
		Q.keep("runner_post_sample_traces", post_sample_traces)
		Q.keep("runner_exp_next_step", runner_exp_next_step)

		# plan path toward anticipated location (t+1) for runner
		#enf_plan = self.plan_path(np.atleast_2d( curr_loc), np.atleast_2d( runner_exp_next_step))
		rx1,ry1,rx2,ry2 = self.seg_map
		enf_plan = planner.run_rrt_opt( np.atleast_2d( curr_loc), np.atleast_2d( runner_exp_next_step), rx1,ry1,rx2,ry2)
		Q.keep("enf_plan", enf_plan)

		# set up the enforcer view (forward vector, fv) for the next step
		cur_enf_loc = scale_up(curr_loc)
		next_enf_loc = scale_up(enf_plan[1])
		next_int_loc = scale_up(runner_exp_path[t+1])

		fv = direction(next_int_loc, cur_enf_loc)

		intersections = self.isovist.GetIsovistIntersections(next_enf_loc, fv)

		# does the enforcer see me at time 't'
		#runner_next_loc = scale_up(runner_exp_next_step)
		will_runner_be_seen = self.isovist.FindIntruderAtPoint( next_int_loc, intersections )
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
	#model = r.Runner_Minus(isovist=isovist, locs=locs, seg_map=seg_map)
	model = r.Runner(isovist=isovist, locs=locs, seg_map=seg_map)
	return model



# def par_params(cnt, samples):
# 	params = ()
# 	for i in xrange(cnt):
# 		trace = example_conditions(p.ProgramTrace(create_runner_model()))
# 		params += ((trace, samples),)
# 	return params

# def run_inference_par(post_samples=5, samples=5):
# 	p = Pool( 6 )
# 	params = par_params(post_samples, samples)
# 	print params
# 	results = p.map( sampling_importance, params )
# 	return results


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

	# sample
	chosen_index = np.random.choice([i for i in range(samples)], p=weights)
	return traces[chosen_index]


def run_inference(trace, post_samples=1, samples=1):
	post_traces = []
	for i in  tqdm(xrange(post_samples)):
		post_sample_trace = sampling_importance(trace, samples=samples)
		post_traces.append(post_sample_trace)
	return post_traces


def rand_expected_future_step(post_sample_traces, name):
	t = post_sample_traces[0]["t"]
	rand_path = randint(0, len(post_sample_traces)-1)
	paths = []
	next_steps = []
	for sample_i, trace in enumerate(post_sample_traces):
		t_fut = min(t+7, len(trace[name]))
		next_steps.append(trace[name][t_fut])
		paths.append(trace[name])
	# print "rand_path", rand_path
	# print "next_steps", next_steps
	# print "len(paths):", len(paths)
	return next_steps[rand_path], paths[rand_path]

def expected_next_step(post_sample_traces, name):
	t = post_sample_traces[0]["t"]
	next_steps = []
	for sample_i, trace in enumerate(post_sample_traces):
		next_steps.append(trace[name][t+1])
	expected_step = list(np.mean(next_steps, axis=0))
	return expected_step

def expected_int_future_step(post_sample_traces, name="int_plan"):
	t = post_sample_traces[0]["t"]
	next_steps = []
	for sample_i, trace in enumerate(post_sample_traces):
		t_fut = min(t+12, len(trace[name])-1)
		next_steps.append(trace[name][t_fut])
	expected_step = list(np.mean(next_steps, axis=0))
	return expected_step

def expected_next_step_replanning(post_sample_traces, name):
	next_steps = []
	for sample_i, trace in enumerate(post_sample_traces):
		next_steps.append(trace[name][1])
	expected_step = list(np.mean(next_steps, axis=0))
	return expected_step





