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



	trace = p.ProgramTrace(model)
	something = trace.run_model()
	print something





