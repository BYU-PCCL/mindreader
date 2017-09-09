import isovist as i
import program_trace as p
import runner as r
from methods import load_isovist_map
from my_rrt import *
import copy
from scipy.misc import logsumexp

# XXX in order for the entruder to do theory of mind
	# to intercept the runner agent, the entruder must do 
	# goal inference.

def create_model():
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
	enf_locs = [[0.14263744, 0.12136674], [0.18804032, 0.14804696], [0.22508292, 0.18175865],
				[0.2630608, 0.20759783], [0.27751185, 0.25381817], [0.28647708, 0.3064716]]

	trace.condition("enf_start", 0)
	trace.condition("enf_goal", 7)
	trace.condition("t", 5)
	for i in xrange(0, 6):
		trace.condition("enf_x_"+str(i+1), enf_locs[i][0])
		trace.condition("enf_y_"+str(i+1), enf_locs[i][1])

	trace.condition("int_start", 1)
	trace.condition("int_goal", 9)
	trace.condition("int_detected", False)
	return trace


def sampling_importance(trace, samples=10):
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


def run_inference(trace, post_samples=10):
	post_traces = []
	for i in xrange(post_samples):
		post_sample_trace = sampling_importance(trace, samples=10)
		post_traces.append(post_traces)
	return post_traces


if __name__ == '__main__':

	# XXX This is testing the runner model. We can view samples from the prior
	# conditioned [on the variable list below]

	# initialize model
	model = create_model()
	# create empty trace using model
	trace = p.ProgramTrace(model)
	# set testing/example conditions in trace
	trace = example_conditions(trace)
	# run inference
	post_sample_traces = run_inference(trace, post_samples=10)







