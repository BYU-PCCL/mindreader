
from show_models import *
import planner
import pickle

# This experiment runs the chaser's runner model to see, given the chaser's true previous
# locations, where the runner thinks the naive chaser is going to go and then 
# where he (the runner)will go given that information
# 

def exp_1(poly_map, isovist, locs):
	
	T = 30
	t = 4
	conditions = {}
	observations = {}
	chaser_plan = pickle.load( open( "chaser_plan_2.p", "rb" ) )

	# Set Conditions
	conditions["init_run_start"] = 4
	conditions["other_run_start"] = 8
	conditions["t"] = t
	for step in xrange(t, T-1):
		conditions["detected_t_"+str(step)] = True

	# Set Observations
	observations["other_run_start"] = 8
	observations["init_run_x_0"] = locs[4][0]
	observations["init_run_y_0"] = locs[4][1]
	for step in xrange(1, t):
		observations["init_run_x_"+str(step)] = chaser_plan[step][0]
		observations["init_run_y_"+str(step)] = chaser_plan[step][1]

	K = 128
	L = 16

	runner_model = BasicRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist, mode="advers")
	tom_runner_model = TOMRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist, 
	 	nested_model=runner_model, inner_samples=L, mode="advers") #inf_type="IR")
	model = tom_runner_model


	params = ((model, observations, conditions),)*K # K different params
	
	sequential_monte_carlo_par(params, K, T=t+2, t_start=t)




def create_RRT(poly_map, isovist, locs):
	rx1,ry1,rx2,ry2 = poly_map
	path = planner.run_rrt_opt( np.atleast_2d(locs[4]), np.atleast_2d(locs[8]), rx1,ry1,rx2,ry2 )
	print path

	fig, ax = setup_plot(poly_map, locs)

	for i in xrange(28):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'orange', linestyle=":", linewidth=1, label="Agent's Plan")

	t = 5
	ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='orange')

	close_plot(fig, ax, plot_name="chaser_plan.eps")

	pickle.dump( path, open( "chaser_plan.p", "wb" ))





if __name__ == '__main__':
	
	# setup 
	locs = [[ 0.100, 1-0.900 ],[ 0.566, 1-0.854 ],[ 0.761, 1-0.665 ],
		[ 0.523, 1-0.604 ],[ 0.241, 1-0.660 ],[ 0.425, 1-0.591 ],
		[ 0.303, 1-0.429 ],[ 0.815, 1-0.402 ],[ 0.675, 1-0.075 ],
		[ 0.432, 1-0.098 ] ]

	poly_map = polygons_to_segments( load_polygons( "./paths.txt" ) )
	isovist = i.Isovist( load_isovist_map() )

	# create random chaser plan
	#create_RRT(poly_map, isovist, locs)

	# run experiment 1
	exp_1(poly_map, isovist, locs)


