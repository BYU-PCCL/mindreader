
from show_models import *
import planner
import pickle
from termcolor import colored
from smartest_chaser import SmartestChaser
from methods import plot_detection_scenario

 #python -W ignore SMC-experiments.py

# smart chaser TODO: need to rewrite model in team_runner.py





	#chaser.run()

###########################################################################
# detection experiments
###########################################################################
# ------ naive runner plan 
# runner_paths/plan1542294334.p
# naive_runner takes 17 steps to reach goal. start=1, goal=8

# ---- smarter runner plan
# runner_paths/plan1542294356.p
# smarter runner takes 18 steps to reach goal. start=1, goal=8 
# hides half the time from the center of the city

 #another example:
 # runner_paths/plan1542294349.p
 # smarter runner takes 22 steps to reach goal. start=1, goal=8
 # hides by wrapping around the city, avoiding the center of the city. 
###########################################################################
def exp_detection(path_filename="runner_paths/plan1542294334.p", chaser=None, verbose=False):
	
	# naive_runner takes 17 steps to reach goal. start=1, goal=8
	runners_plan = no_hover([pickle.load( open( path_filename, "rb" ))])[0]
	print len(runners_plan)

	detected = []
	TOTAL_SIMULAIONS = 40
	LIMITED_TIME_STEPS = np.min([15, len(runners_plan)])
	for sim in xrange(TOTAL_SIMULAIONS):
		chaser.restart(chaser.my_start, chaser.runner_start, new_scenario=True)
		chaser_plan = [chaser.get_start_location()]
		# run simulation for as long as it takes for the runner to get to the goal location
		for t in tqdm(xrange(1,LIMITED_TIME_STEPS)):#len(runners_plan)):
			# runner take step
			runner_loc = runners_plan[t]

			# chaser take step
			chaser_loc = chaser.step()
			# add chaser step to its path
			chaser_plan.append(chaser_loc)
			#chaser_loc = [0.5,0.6]


			#forward vector of chaser
			if dist(chaser_loc, runner_loc) <= 0.4:
				# forward direction toward the runner
				fv = direction(scale_up(runner_loc), scale_up(chaser_loc))
			else:
				# forward direction based on its direction of step
				fv = direction(scale_up(chaser_plan[t-1]), scale_up(chaser_loc))

			# calculate field of view
			intersections = isovist.GetIsovistIntersections(scale_up(chaser_loc), fv)
			# is the runner detected
			is_runner_visible = isovist.FindIntruderAtPoint( scale_up(runner_loc), intersections)
			
			if verbose:
			 	plot_detection_scenario(chaser_plan[:t+1], runners_plan[:t+1])
				chaser.plot_detection_scenario(chaser_plan[:t+1], runners_plan[:t+1], t)
				if is_runner_visible:
					print "t:", t, "\tdetected:", colored(is_runner_visible, 'green')
					
					break
				else:
					print "t:", t, "\tdetected:", colored(is_runner_visible, 'red')



			
			chaser.update(chaser_plan, t+1)

		detected.append(is_runner_visible)
		

	print np.sum(detected)/float(TOTAL_SIMULAIONS) * 100, "% detected"
	#plot_detection_scenario(chaser_plan, runners_plan)

def exp_1(poly_map, isovist, locs):
	
	T = 30
	t = 1
	conditions = {}
	observations = {}
	chaser_plan = pickle.load( open( "chaser_plan_3.p", "rb" ) )

	# Set Conditions
	conditions["init_run_start"] = 3
	conditions["other_run_start"] = 8
	conditions["t"] = t
	for step in xrange(t, T-1):
		conditions["detected_t_"+str(step)] = True

	# Set Observations
	observations["other_run_start"] = 8
	observations["init_run_x_0"] = locs[3][0]
	observations["init_run_y_0"] = locs[3][1]
	for step in xrange(1, t):
		observations["init_run_x_"+str(step)] = chaser_plan[step][0]
		observations["init_run_y_"+str(step)] = chaser_plan[step][1]

	K = 1
	L = 1


	runner_model = BasicRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist, mode="advers")
	tom_runner_model = TOMRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist, 
	 	nested_model=runner_model, inner_samples=L, mode="advers") #inf_type="IR")
	model = tom_runner_model


	params = ((model, observations, conditions),)*K # K different params
	
	sequential_monte_carlo_par(params, K, T=t+2, t_start=t)

	store_dir = "K"+str(K)+"L"+str(L)+"-"+str(int(time.time()))
	directory = model.get_dir()+"/"+model.get_id()+"_t-"+str(t)+".p"
	show_exp_1_results(directory, poly_map, isovist, locs, store_dir)

def exp_2(poly_map, isovist, locs):
	
	T = 30
	t = 1
	conditions = {}
	observations = {}

	# Set Conditions
	conditions["init_run_start"] = 3
	conditions["init_run_goal"] = 9
	conditions["other_run_start"] = 7

	conditions["t"] = t
	for step in xrange(t, T-1):
		conditions["detected_t_"+str(step)] = True

	# Set Observations
	observations["other_run_start"] = 7
	observations["other_run_goal"] = 9
	observations["init_run_x_0"] = locs[3][0]
	observations["init_run_y_0"] = locs[3][1]

	K = 32
	L = 64


	runner_model = BasicRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist, mode="advers")
	tom_runner_model = TOMRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist, 
	 	nested_model=runner_model, inner_samples=L, mode="advers") #inf_type="IR")
	model = tom_runner_model


	params = ((model, observations, conditions),)*K # K different params
	
	sequential_monte_carlo_par(params, K, T=t+2, t_start=t)

	store_dir = "K"+str(K)+"L"+str(L)+"-"+str(int(time.time()))
	directory = model.get_dir()+"/"+model.get_id()+"_t-"+str(t)+".p"
	show_exp_2_results(directory, poly_map, isovist, locs, store_dir)



def show_exp_2_results(directory, poly_map, isovist, locs, store_dir=""):
	#fig, ax = setup_plot(poly_map, locs)

	#close_plot(fig, ax, plot_name="exp_1_results.eps")

	os.mkdir("exp-2-heatmaps/"+store_dir)
	KQ_info = pickle.load( open( directory, "rb" ) )
	K = KQ_info["K"]


	#sample = np.where(mean_q_scores ==  np.max(mean_q_scores))[0][0]
	real_chaser_plans = []
	for sample in tqdm(xrange(1, K+1)):
		trace_k = KQ_info["resamp-"+str(sample)]

		#print np.array(trace_k["all_ql_scores"]).mean()

		runners_plan = trace_k["other_plan"] #(16, 30, 2)
		imaginary_chaser_plans = trace_k["L_imaginary_chaser_plans"]#(16, 30, 2)
		score = trace_k["mean"]
		print "trace mean", score

		runner_scores = np.array(trace_k["all_ql_scores"])

		map_style = "jet"
		#map_style = "cubehelix"

		dir_name = "exp-2-heatmaps/"+store_dir+"/K"+str(sample)
		os.mkdir(dir_name)

		hmap = path_to_heatmap(no_hover(runners_plan, remove_start= 1))
		hmap = np.mean( hmap, axis=2 )
		fig, ax = setup_plot(poly_map, locs, scale = 500)
		plt.xticks([])
		plt.yticks([])
		cax = ax.imshow( hmap, interpolation='nearest', cmap=map_style, origin='lower')
		cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
		cbar.ax.set_yticklabels(['0', '', ''])
		plt.savefig(dir_name + "/runners_plan_hmap.eps", bbox_inches='tight')


		hmap = path_to_heatmap(no_hover(imaginary_chaser_plans, remove_start=1))
		hmap = np.mean( hmap, axis=2 )
		fig, ax = setup_plot(poly_map, locs, scale = 500)
		plt.xticks([])
		plt.yticks([])
		cax = ax.imshow( hmap, interpolation='nearest', cmap=map_style, origin='lower')
		cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
		cbar.ax.set_yticklabels(['0', '', ''])
		plt.savefig(dir_name + "/im_chaser_plan_hmap.eps", bbox_inches='tight')

		real_chaser_plans.append(trace_k["my_plan"])


	#print trace_k["my_plan"]
	hmap = path_to_heatmap(no_hover(real_chaser_plans, remove_start=1))
	hmap = np.mean( hmap, axis=2 )
	fig, ax = setup_plot(poly_map, locs, scale = 500)
	plt.xticks([])
	plt.yticks([])
	cax = ax.imshow( hmap, interpolation='nearest', cmap=map_style, origin='lower')
	cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
	cbar.ax.set_yticklabels(['0', '', ''])
	plt.savefig(dir_name + "/real_chaser_plan_hmap.eps", bbox_inches='tight')


def no_hover(plans, remove_start=0):
	paths = []
	for path in plans:
		# remove hovering points on path
		no_hover_path = []
		for pt in path:
			if abs(pt[0] - path[-2][0]) > .01:
				if abs(pt[1] - path[-2][1]) > 0.01:
					no_hover_path.append(pt)
		no_hover_path.append(path[-2])
		if len(no_hover_path[remove_start:]) > 0:
			paths.append(no_hover_path[remove_start:])
	return paths



def show_exp_1_results(directory, poly_map, isovist, locs, store_dir=""):
	#fig, ax = setup_plot(poly_map, locs)

	#close_plot(fig, ax, plot_name="exp_1_results.eps")

	os.mkdir("exp-1-heatmaps/"+store_dir)
	KQ_info = pickle.load( open( directory, "rb" ) )
	K = KQ_info["K"]

	mean_q_scores = []
	q_base = 8.295
	for k in xrange(1, K+1):
		trace_k = KQ_info["orig-"+str(k)]
		mean_q_scores.append((np.array(trace_k["all_ql_scores"]) + q_base).mean())
	#print np.mean(mean_q_scores)
	#print np.max(mean_q_scores), np.where(mean_q_scores ==  np.max(mean_q_scores))



	#sample = np.where(mean_q_scores ==  np.max(mean_q_scores))[0][0]

	for sample in tqdm(xrange(1, K+1)):
		trace_k = KQ_info["resamp-"+str(sample)]

		#print np.array(trace_k["all_ql_scores"]).mean()

		runners_plan = trace_k["other_plan"] #(16, 30, 2)
		imaginary_chaser_plans = trace_k["L_imaginary_chaser_plans"]#(16, 30, 2)
		score = trace_k["mean"]
		runner_scores = np.array(trace_k["all_ql_scores"])

		map_style = "jet"
		#map_style = "cubehelix"

		dir_name = "exp-1-heatmaps/"+store_dir+"/K"+str(sample)
		os.mkdir(dir_name)

		hmap = path_to_heatmap(no_hover(runners_plan, remove_start= 2))
		hmap = np.mean( hmap, axis=2 )
		fig, ax = setup_plot(poly_map, locs, scale = 500)
		plt.xticks([])
		plt.yticks([])
		cax = ax.imshow( hmap, interpolation='nearest', cmap=map_style, origin='lower')
		cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
		cbar.ax.set_yticklabels(['0', '', ''])
		plt.savefig(dir_name + "/runners_plan_hmap.eps", bbox_inches='tight')




		hmap = path_to_heatmap(no_hover(imaginary_chaser_plans, remove_start=2))
		hmap = np.mean( hmap, axis=2 )
		fig, ax = setup_plot(poly_map, locs, scale = 500)
		plt.xticks([])
		plt.yticks([])
		cax = ax.imshow( hmap, interpolation='nearest', cmap=map_style, origin='lower')
		cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
		cbar.ax.set_yticklabels(['0', '', ''])
		plt.savefig(dir_name + "/chaser_plan_hmap.eps", bbox_inches='tight')


def show_single_exp_1_results(directory, poly_map, isovist, locs, store_dir=""):

	os.mkdir("exp-1-heatmaps/"+store_dir)
	KQ_info = pickle.load( open( directory, "rb" ) )
	K = KQ_info["K"]

	# mean_q_scores = []
	# q_base = 8.295
	# for k in xrange(1, K+1):
	# 	trace_k = KQ_info["orig-"+str(k)]
	# 	mean_q_scores.append((np.array(trace_k["all_ql_scores"]) + q_base).mean())
	
	# print np.mean(mean_q_scores)

	#print np.max(mean_q_scores), np.where(mean_q_scores ==  np.max(mean_q_scores))
	#sample = np.where(mean_q_scores ==  np.max(mean_q_scores))[0][0]
	sample = 30

	trace_k = KQ_info["resamp-"+str(sample)]

	#print np.array(trace_k["all_ql_scores"]).mean()

	runners_plan = trace_k["other_plan"] #(16, 30, 2)
	imaginary_chaser_plans = trace_k["L_imaginary_chaser_plans"]#(16, 30, 2)
	score = trace_k["mean"]
	runner_scores = np.array(trace_k["all_ql_scores"])

	map_style = "jet"
	#map_style = "cubehelix"

	dir_name = "exp-1-heatmaps/"+store_dir+"/K"+str(sample)
	os.mkdir(dir_name)

	hmap = path_to_heatmap(no_hover(runners_plan, remove_start= 2))
	hmap = np.mean( hmap, axis=2 )
	fig, ax = setup_plot(poly_map, locs, scale = 500)
	plt.xticks([])
	plt.yticks([])
	cax = ax.imshow( hmap, interpolation='nearest', cmap=map_style, origin='lower')
	cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
	cbar.ax.set_yticklabels(['0', '', ''])
	plt.savefig(dir_name + "/runners_plan_hmap.eps", bbox_inches='tight')




	hmap = path_to_heatmap(no_hover(imaginary_chaser_plans, remove_start=2))
	hmap = np.mean( hmap, axis=2 )
	fig, ax = setup_plot(poly_map, locs, scale = 500)
	plt.xticks([])
	plt.yticks([])
	cax = ax.imshow( hmap, interpolation='nearest', cmap=map_style, origin='lower')
	cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
	cbar.ax.set_yticklabels(['0', '', ''])
	plt.savefig(dir_name + "/chaser_plan_hmap.eps", bbox_inches='tight')



def create_RRT(poly_map, isovist, locs, name="runner_plan_naive"):
	rx1,ry1,rx2,ry2 = poly_map
	path = planner.run_rrt_opt( np.atleast_2d(locs[4]), np.atleast_2d(locs[8]), rx1,ry1,rx2,ry2 )
	print path

	fig, ax = setup_plot(poly_map, locs)

	for i in xrange(28):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'orange', linestyle=":", linewidth=1, label="Agent's Plan")

	t = 5
	ax.scatter( path[t][0],  path[t][1], s = 95, facecolors='none', edgecolors='orange')

	rand_id = str(int(time.time()))
	close_plot(fig, ax, plot_name="runner_paths/"+name+rand_id+".eps")

	pickle.dump( path, open( "runner_paths/"+name+rand_id+".p", "wb" ))





if __name__ == '__main__':
	
	# setup 
	locs = [[ 0.100, 1-0.900 ],[ 0.566, 1-0.854 ],[ 0.761, 1-0.665 ],
		[ 0.523, 1-0.604 ],[ 0.241, 1-0.660 ],[ 0.425, 1-0.591 ],
		[ 0.303, 1-0.429 ],[ 0.815, 1-0.402 ],[ 0.675, 1-0.075 ],
		[ 0.432, 1-0.098 ] ]

	poly_map = polygons_to_segments( load_polygons( "./paths.txt" ) )
	isovist = i.Isovist( load_isovist_map() )

	# create random chaser plan
	# for i in xrange(100):
	# 	create_RRT(poly_map, isovist, locs, "plan2")

	# run experiment 1
	#exp_1(poly_map, isovist, locs)
	#exp_2(poly_map, isovist, locs)


	#directory = "PO_forward_runs/conditioned/SMC_simulations/resamp-sim-1542127053/1542127053_t-4.p"
	#1542151325
	#directory = "PO_forward_runs/conditioned/SMC_simulations/resamp-sim-1542151325/1542151325_t-4.p"
	#1542153321
	#directory = "PO_forward_runs/conditioned/SMC_simulations/resamp-sim-1542153321/1542153321_t-4.p"
	#1542155003
	#directory = "PO_forward_runs/conditioned/SMC_simulations/resamp-sim-1542155003/1542155003_t-1.p"
	#1542156717
	#directory = "PO_forward_runs/conditioned/SMC_simulations/resamp-sim-1542156717/1542156717_t-5.p"
	#1542209339
	#run_id = "1542213761"
	#t = 1
	#directory = "PO_forward_runs/conditioned/SMC_simulations/resamp-sim-"+run_id+"/"+run_id+"_t-"+str(t)+".p"
	

	#show_single_exp_1_results(directory, poly_map, isovist, locs, store_dir="k=30-2")
	#show_exp_1_results(directory, poly_map, isovist, locs, store_dir="k=30")


	# detection experiments
	#exp_detection(poly_map, isovist, locs, path_filename="runner_paths/plan1542294334.p", verbose=True)
	#plan1542294356.p
	
	# start 4 tp 8
	naive = "runner_paths/plan21542303398.p" # 100% smart chaser out of ten
	smarter = "runner_paths/plan21542303411.p" # 30% smart chaser out of ten 
	smart_chaser = SmartestChaser(seg_map=poly_map, locs=locs, isovist=isovist, K=1, L=1, my_start=8, runner_start=4)
	exp_detection( path_filename=smarter, chaser =smart_chaser,  verbose=True)

	# naive = "runner_paths/plan21542303398.p" # 100 % out of ten
	# smarter = "runner_paths/plan21542303411.p" # 
	# tom_chaser = SmartestChaser(seg_map=poly_map, locs=locs, isovist=isovist, K=128, L=16, my_start=8, runner_start=4)
	# exp_detection( path_filename=smarter, chaser =tom_chaser,  verbose=True)