import isovist as i
import program_trace as p
import runner as r
from methods import load_isovist_map
from my_rrt import *
import copy as c

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


def sampling_importance(trace, samples=1):
	traces = []
	scores = np.arange(samples)

	for i in xrange(samples):
		#deep copy is not working
		t = c.deepcopy(trace)
		score, trace_vals = t.run_model()
		traces.append(trace_vals)
		scores[i] = score
	print traces
	print scores


	#weights = np.exp()

 # 	  weights = exp.(scores - logsumexp(scores))
 #    weights = weights / sum(weights)

 #    # pick a trace in propotion to its relative weight and return it
 #    chosen = rand(Categorical(weights))
 #    return traces[chosen]


if __name__ == '__main__':

	# XXX This is testing the runner model. We can view samples from the prior
	# conditioned [on the variable list below]

	# initialize model
	model = create_model()
	# create empty trace using model
	trace = p.ProgramTrace(model)
	# set testing/example conditions in trace
	trace = example_conditions(trace)

	#sampling_importance(trace, samples=1)


	# generate 
	# something = trace.run_model()

	# print something





