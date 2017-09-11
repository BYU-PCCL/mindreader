
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

#import seaborn
#seaborn.set_style(style='white')

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





