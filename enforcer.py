import isovist
import program_trace as p
import runner as r
from methods import load_isovist_map
from my_rrt import *



if __name__ == '__main__':
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

	# load isovist
	isovist = isovist.Isovist( load_isovist_map() )

	model = r.Runner(isovist, locs, seg_map)
	#model.run(None)

	# XXX in order for the entruder to do theory of mind
	# to intercept the runner agent, the entruder must know
	# the start and goal locations of the runner. 
	# So in this case, we're not trying to infer the goal explicitly
	# of the agent, we are trying to infer the path the runner
	# would take to his goal in order to remain undetected

# (1, [array([[ 0.14263744]]), array([[ 0.12136674]])])
# (2, [array([[ 0.18804032]]), array([[ 0.14804696]])])
# (3, [array([[ 0.22508292]]), array([[ 0.18175865]])])
# (4, [array([[ 0.2630608]]), array([[ 0.20759783]])])
# (5, [array([[ 0.27751185]]), array([[ 0.25381817]])])
# (6, [array([[ 0.28647708]]), array([[ 0.3064716]])])

	enf_locs = [[0.14263744, 0.12136674], [0.18804032, 0.14804696], [0.22508292, 0.18175865],
				[0.2630608, 0.20759783], [0.27751185, 0.25381817], [0.28647708, 0.3064716]]

	trace = p.ProgramTrace(model)
	trace.condition("enf_start", 0)
	trace.condition("enf_goal", 7)
	trace.condition("t", 5)
	for i in xrange(0, 6):
		trace.condition("enf_x_"+str(i+1), enf_locs[i][0])
		trace.condition("enf_y_"+str(i+1), enf_locs[i][1])

	something = trace.run_model()
	print something





