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
		trace.condition("enf_x_"+str(i+1), enf_locs[i][0])
		trace.condition("enf_y_"+str(i+1), enf_locs[i][1])

	trace.condition("int_start", 1)
	trace.condition("int_goal", 9)
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
		trace = example_conditions(p.ProgramTrace(create_model()))
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
		
		ax.scatter( path[t+1][0] * scale, path[t+1][1]  * scale, color="blue", s = 40, marker="x", label="Enforcer (t+1)")
		ax.scatter( path[0][0] * scale, path[0][1]  * scale, color="green", label="Start location")
		ax.scatter( path[-1][0] * scale, path[-1][1] * scale, color = "red", label="Entruder Inferred Goals")
		ax.scatter( path[t][0] * scale, path[t][1] * scale, color = "blue", s = 55, marker="v", label="Enforcer")

		# plot intruder_plan
		path = trace["int_plan"]
		for i in range( 0, len(path)-1 ):
			ax.plot( [ path[i][0] * scale, path[i+1][0] * scale ], [ path[i][1] * scale, path[i+1][1] * scale], 'grey', alpha=0.7 )
			if i+1 == t:
				break		
		ax.scatter( path[t+1][0] * scale, path[t+1][1]  * scale, color="magenta", s = 40, marker="x", label="Runner (t+1)")

		ax.scatter( path[0][0] * scale, path[0][1]  * scale, color="green")
		#ax.scatter( path[0][0] * scale, path[0][1] * scale, color = "red")
		ax.scatter( path[t][0] * scale, path[t][1] * scale, color = "magenta", s = 45, marker = "D", label="Runner")

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
	#ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1), shadow=True, ncol=1)
	
	fig.savefig('plot.png')
	#plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1, bbox_inches="tight")
	#plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
	plt.show()

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
	#post_sample_traces = run_inference(trace, post_samples=10, samples =5)
	post_sample_traces = run_inference(trace, post_samples=10, samples =5)
	#post_sample_traces = run_inference_par(post_samples=5, samples=5)
	# pickel post sample traces
	cPickle.dump( post_sample_traces, open("./9-09-p10-s5-1.cp","w") )
	#shot posterior samples
	show_post_traces(post_sample_traces)
	print "DONE"






