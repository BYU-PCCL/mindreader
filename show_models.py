#from enforcer import *
from methods import load_isovist_map, scale_up, direction, dist, detect, load_segs, get_clear_goal,point_in_obstacle
from my_rrt import *
import isovist as i
from random import randint
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import cPickle
from inference_alg import importance_sampling, metroplis_hastings
from program_trace import ProgramTrace
from planner import * 
from tqdm import tqdm
import os

# team runner model types
from team_runner import BasicRunner
from team_runner import TOMCollabRunner
from team_runner import BasicRunnerPOM
from team_runner import TOMRunnerPOM


#import seaborn

def plot(poly_map, plot_name=None, locs=None):
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	scale = 1

	# plot map
	x1,y1,x2,y2 = poly_map
	#x1,y1,x2,y2 = polygons_to_segments( load_polygons( "./map_2.txt" ) )
	for i in xrange(x1.shape[0]):
		ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'black', linewidth=1 )


	add_locations = True
	if add_locations:
		for i in xrange(len(locs)):
			ax.scatter( locs[i][0],  locs[i][1] , color="Green", s = 70, marker='+', linestyle='-')
			ax.scatter( locs[i][0],  locs[i][1] , s = 95, facecolors='none', edgecolors='g')
			if i == 5:
				ax.scatter( locs[i][0],  locs[i][1] , color="Red", s = 70, marker='+', linestyle='-')
				ax.scatter( locs[i][0],  locs[i][1] , s = 95, facecolors='none', edgecolors='red')

	
	if plot_name is None:
		plot_name = str(int(time.time()))+".eps"

	ax.set_ylim(ymax = 1, ymin = 0)
	ax.set_xlim(xmax = 1, xmin = 0)

	#plt.show()
	fig.savefig(plot_name, bbox_inches='tight')

def setup_plot(poly_map, locs=None):
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	scale = 1

	# plot map
	x1,y1,x2,y2 = poly_map
	for i in xrange(x1.shape[0]):
		ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'black', linewidth=1  )

	# show possible start and goal locations
	add_locations = True
	if add_locations:
		for i in xrange(len(locs)):
			ax.scatter( locs[i][0],  locs[i][1] , color="Green", s = 50, marker='+', linestyle='-')
			ax.scatter( locs[i][0],  locs[i][1] , s = 75, facecolors='none', edgecolors='g')
	return fig, ax

def close_plot(fig, ax, plot_name=None):
	if plot_name is None:
		plot_name = str(int(time.time()))+".eps"
	print "plot_name:", plot_name

	ax.set_ylim(ymax = 1, ymin = 0)
	ax.set_xlim(xmax = 1, xmin = 0)

	#plt.show()
	fig.savefig(plot_name, bbox_inches='tight')

def run_basic_forward(runner_model, poly_map, locs):

	fig, ax = setup_plot(poly_map, locs)

	for samples in xrange(100):
		Q = ProgramTrace(runner_model)
		Q.condition("run_start", 2)
		Q.condition("run_goal", 4)
		Q.condition("t", 8)
		score, trace = Q.run_model()

		path = trace["runner_plan"]

		t = trace["t"]
		for i in range( 0, len(path)-1):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'black', linestyle=":", linewidth=1)
			#ax.scatter( path[i][0],  path[i][1] , s = 70, facecolors='none', edgecolors='grey')

		ax.scatter( path[t][0],  path[t][1] , s = 70, facecolors='none', edgecolors='red')

	plt.figtext(0.92, 0.85, "Values", horizontalalignment='left', weight="bold") 
	plt.figtext(0.92, 0.80, "Start: " +str(trace["run_start"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.75, "Goal: " +str(trace["run_goal"]), horizontalalignment='left')
	plt.figtext(0.92, 0.70, "time step: " +str(trace["t"]), horizontalalignment='left') 


	close_plot(fig, ax, plot_name=str(int(time.time()))+"-basic.eps")


def plot_runner(poly_map, trace, locs=None):
	fig, ax = setup_plot(poly_map, locs)

	# get time
	t = trace["t"]

	# plot runner's plan
	path = trace["runner_plan"]
	for i in range( 0, len(path)-1):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'black', linestyle=":", linewidth=1)
		# if you want to show each time step in blue
	close_plot(fig, ax, plot_name=str(int(time.time()))+"-1.eps")

	for i in range( 0, len(path)):
		ax.scatter( path[i][0],  path[i][1] , s = 35, facecolors='none', edgecolors='blue')

	# mark the runner at time t on its plan
	#ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='b')

	# # plot partner's plan
	# path = trace["partner_plan"]
	# for i in range( 0, len(path)-1):
	# 	ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
	# 		'grey', linestyle="--", linewidth=1)
	# # mark the parter on its plan on time t
	# ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='orange')

	close_plot(fig, ax, plot_name=str(int(time.time()))+"-2.eps")


def simulate_running_goal_inference(runner_model, poly_map, locs):
	x1,y1,x2,y2 = poly_map
	# plan for the runner
	start = 4
	goal = 0
	path = run_rrt_opt( np.atleast_2d(locs[start]), 
		np.atleast_2d(locs[goal]), x1,y1,x2,y2 )

	fig, ax = setup_plot(poly_map, locs)

	for i in range( 0, len(path)-1):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'lightgrey', linestyle="-", linewidth=1)

	# ax.scatter( path[0][0],  path[0][1] , s = 120, facecolors='none', edgecolors='g')
	# ax.scatter( path[-1][0],  path[-1][1] , s = 120, facecolors='none', edgecolors='r')

	close_plot(fig, ax, "true_path.eps")
		

	#for t in xrange(15):

	t = 10
	Q = ProgramTrace(runner_model)

	Q.condition("run_start", 4)
	Q.condition("t", t) 
	# condition on previous time steps
	for prev_t in xrange(t):
		Q.condition("run_x_"+str(prev_t), path[prev_t][0])
		Q.condition("run_y_"+str(prev_t), path[prev_t][1])
		ax.scatter( path[prev_t][0],  path[prev_t][1] , s = 70, facecolors='none', edgecolors='b')
	ax.scatter( path[t][0],  path[t][1] , s = 80, facecolors='none', edgecolors='r')

	post_sample_traces = run_inference(Q, post_samples=32, samples=64)

	# show post sample traces on map
	for trace in post_sample_traces:

		path = trace["runner_plan"]
		for i in range( 0, len(path)-1):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'red', linestyle="--", linewidth=1, alpha = 0.2)

	close_plot(fig, ax, "infered_goal.eps")

# This simulates two agents performing goal inference on one another after each move
def goal_inference_while_moving(runner_model, poly_map, locs):
	sim_id = str(int(time.time()))
	x1,y1,x2,y2 = poly_map
	# plan for the runner
	start = 4
	goal = 1
	path = run_rrt_opt( np.atleast_2d(locs[start]), 
		np.atleast_2d(locs[goal]), x1,y1,x2,y2 )

	fig, ax = setup_plot(poly_map, locs)

	for i in range( 0, len(path)-1):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'lightgrey', linestyle="-", linewidth=1)

	ax.scatter( path[0][0],  path[0][1] , s = 120, facecolors='none', edgecolors='b')
	ax.scatter( path[-1][0],  path[-1][1] , s = 120, facecolors='none', edgecolors='r')

	close_plot(fig, ax, "time/" + sim_id + "_true_path.eps")

	#inferrred_goals = { 0:0, 1:0, 2:0, 3:0, 4:0, 5:0 }
	inferrred_goals = []
	for t in xrange(0, min(25, len(path))):
		fig, ax = setup_plot(poly_map, locs)
		Q = ProgramTrace(runner_model)

		Q.condition("run_start", start)
		Q.condition("t", t) 
		# condition on previous time steps
		for prev_t in xrange(t):
			Q.condition("run_x_"+str(prev_t), path[prev_t][0])
			Q.condition("run_y_"+str(prev_t), path[prev_t][1])
			ax.scatter( path[prev_t][0],  path[prev_t][1] , s = 70, facecolors='none', edgecolors='b')
		ax.scatter( path[t][0],  path[t][1] , s = 80, facecolors='none', edgecolors='r')

		post_sample_traces = run_inference(Q, post_samples=10, samples=16)

		goal_list = []
		# show post sample traces on map
		for trace in post_sample_traces:
			inferred_goal = trace["run_goal"]
			goal_list.append(inferred_goal)
			#print goal_list
			inff_path = trace["runner_plan"]
			for i in range( 0, len(inff_path)-1):
				ax.plot( [inff_path[i][0], inff_path[i+1][0] ], [ inff_path[i][1], inff_path[i+1][1]], 
					'red', linestyle="--", linewidth=1, alpha = 0.2)

		inferrred_goals.append(goal_list)
		print "goal list:", goal_list
		close_plot(fig, ax, "time/" + sim_id + "-post-samples-t-"+str(t)+".eps")

	print "inferrred_goals:", inferrred_goals
	return inferrred_goals, sim_id


def get_most_probable_goal_location(runner_model, poly_map, locs, sim_id, path, start, t, character):
	fig, ax = setup_plot(poly_map, locs)
	Q = ProgramTrace(runner_model)

	Q.condition("run_start", start)
	Q.condition("t", t) 
	# condition on previous time steps
	for prev_t in xrange(t):
		Q.condition("run_x_"+str(prev_t), path[prev_t][0])
		Q.condition("run_y_"+str(prev_t), path[prev_t][1])
		ax.scatter( path[prev_t][0],  path[prev_t][1] , s = 70, facecolors='none', edgecolors='b')
	ax.scatter( path[t][0],  path[t][1] , s = 80, facecolors='none', edgecolors='r')

	post_sample_traces = run_inference(Q, post_samples=10, samples=16)

	goal_list = []
	# show post sample traces on map
	for trace in post_sample_traces:
		inferred_goal = trace["run_goal"]
		goal_list.append(inferred_goal)
		#print goal_list
		inff_path = trace["runner_plan"]
		for i in range( 0, len(inff_path)-1):
			ax.plot( [inff_path[i][0], inff_path[i+1][0] ], [ inff_path[i][1], inff_path[i+1][1]], 
				'red', linestyle="--", linewidth=1, alpha = 0.2)

	close_plot(fig, ax, "collab/" + sim_id + character + "-post-samples-t-"+str(t)+".eps")

	# list with probability for each goal
	goal_probabilities = []
	total_num_inferences = len(goal_list)
	# turn into percents
	for goal in xrange(10):
		goal_cnt = goal_list.count(goal)
		goal_prob = goal_cnt / float(total_num_inferences)
		goal_probabilities.append(goal_prob)

	return goal_probabilities.index(max(goal_probabilities))


def add_Obs(Q, start, t, path):
	Q.set_obs("run_start", start)

	Q.set_obs("t", t)
	# condition on previous time steps
	for prev_t in xrange(t):
		Q.set_obs("run_x_"+str(prev_t), path[prev_t][0])
		Q.set_obs("run_y_"+str(prev_t), path[prev_t][1])
	return Q


# This simulates two agents performing nested goal inference using "TOM"
def two_agent_nested_goal_inference_while_moving(runner_model, poly_map, locs):
	sim_id = str(int(time.time()))
	x1,y1,x2,y2 = poly_map

	#Alice will start at some location
	alice_start = 4
	alice_path = [locs[alice_start]]

	#Bob will start st some other location
	bob_start = 5
	bob_path = [locs[bob_start]]

	alices_inferrred_goals_for_bob = []
	bobs_inferrred_goals_for_alice = []
	# for each time step
	for t in xrange(0, 25):
		#Alice will conduct goal inference on observations of bob's location
		Q = add_Obs(ProgramTrace(runner_model), alice_start, t, alice_path)
		inferred_bob_goal, bobs_goal_list = nested_most_probable_goal_location(Q, poly_map, locs, sim_id, 
			bob_path, bob_start, t, "B")
		
		#Bob will conduct goal inference on observations of alice's location
		Q = add_Obs(ProgramTrace(runner_model), bob_start, t, bob_path)
		inferred_alice_goal, alices_goal_list = nested_most_probable_goal_location(Q, poly_map, locs, sim_id, 
			alice_path, alice_start, t,"A")

		alices_inferrred_goals_for_bob.append(bobs_goal_list)
		bobs_inferrred_goals_for_alice.append(alices_goal_list)

		#Alice will move toward Bob's goal after planning
		alice_plan = run_rrt_opt( np.atleast_2d(alice_path[-1]), 
		np.atleast_2d(locs[inferred_bob_goal]), x1,y1,x2,y2 )
		alice_path.append(alice_plan[1])

		#Bob will move toward Alice's goal after planning
		bob_plan = run_rrt_opt( np.atleast_2d(bob_path[-1]), 
		np.atleast_2d(locs[inferred_alice_goal]), x1,y1,x2,y2 )
		bob_path.append(bob_plan[1])

		plot_movements(alice_path, bob_path, sim_id, poly_map, locs, t, code="nested")

	line_plotting(alices_inferrred_goals_for_bob, sim_id, code="A-in-B", directory="tom-collab")
	line_plotting(bobs_inferrred_goals_for_alice, sim_id, code="B-in-A", directory="tom-collab")

def nested_most_probable_goal_location(Q, poly_map, locs, sim_id, path, start, t, character):
	fig, ax = setup_plot(poly_map, locs)

	Q.condition("co_run_start", start)
	Q.condition("t", t) 
	# condition on previous time steps
	for prev_t in xrange(t):
		Q.condition("co_run_x_"+str(prev_t), path[prev_t][0])
		Q.condition("co_run_y_"+str(prev_t), path[prev_t][1])
		ax.scatter( path[prev_t][0],  path[prev_t][1] , s = 70, facecolors='none', edgecolors='b')
	ax.scatter( path[t][0],  path[t][1] , s = 80, facecolors='none', edgecolors='r')

	Q.condition("same_goal", True)

	post_sample_traces = run_inference(Q, post_samples=10, samples=16)

	goal_list = []
	# show post sample traces on map
	for trace in post_sample_traces:
		inferred_goal = trace["co_run_goal"]
		goal_list.append(inferred_goal)
		#print goal_list
		inff_path = trace["co_runner_plan"]
		for i in range( 0, len(inff_path)-1):
			ax.plot( [inff_path[i][0], inff_path[i+1][0] ], [ inff_path[i][1], inff_path[i+1][1]], 
				'red', linestyle="--", linewidth=1, alpha = 0.2)

	close_plot(fig, ax, "tom-collab/" + sim_id + character + "-post-samples-t-"+str(t)+".eps")

	# list with probability for each goal
	goal_probabilities = []
	total_num_inferences = len(goal_list)
	# turn into percents
	for goal in xrange(6):
		goal_cnt = goal_list.count(goal)
		goal_prob = goal_cnt / float(total_num_inferences)
		goal_probabilities.append(goal_prob)

	return goal_probabilities.index(max(goal_probabilities)), goal_list



def plot_movements(a_path, b_path, sim_id, poly_map, locs, t, code="", directory="collab"):
	fig, ax = setup_plot(poly_map, locs)
	# PLOTTING MOVEMENTS
	path = a_path
	for i in range(0, t):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'black', linestyle=":", linewidth=2, label="Alice")
	# mark the runner at time t on its plan
	ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='orange')

	path = b_path
	for i in range(0, t):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'black', linestyle="--", linewidth=2, label="Bob")
	# mark the runner at time t on its plan
	ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='blue')
	close_plot(fig, ax, directory + "/" + sim_id + code + "-" +str(t)+".eps")



def follow_the_leader_goal_inference(runner_model, poly_map, locs):
	sim_id = str(int(time.time()))
	x1,y1,x2,y2 = poly_map

	#Alice will start at some location
	alice_start = 4
	goal = 3
	alice_path = run_rrt_opt( np.atleast_2d(locs[alice_start]), 
		np.atleast_2d(locs[goal]), x1,y1,x2,y2 )

	#Bob will start st some other location
	bob_start = 5
	#bob_path = [locs[bob_start]]
	bob_path = [[0.2, 0.15]]

	# for each time step
	for t in xrange(0, 25):
		
		#Bob will conduct goal inference on observations of alice's location
		inferred_alice_goal = get_most_probable_goal_location(runner_model, poly_map, locs, sim_id, 
			alice_path, alice_start, t, "FL-A")

		#Bob will move toward Alice's goal after planning
		bob_plan = run_rrt_opt( np.atleast_2d(bob_path[-1]), 
		np.atleast_2d(locs[inferred_alice_goal]), x1,y1,x2,y2 )
		bob_path.append(bob_plan[1])

		plot_movements(alice_path, bob_path, sim_id, poly_map, locs, t, code="FL-t")


def two_agent_goal_inference_while_moving(runner_model, poly_map, locs):
	sim_id = str(int(time.time()))
	x1,y1,x2,y2 = poly_map

	#Alice will start at some location
	alice_start = 4
	alice_path = [locs[alice_start]]

	#Bob will start st some other location
	bob_start = 5
	bob_path = [locs[bob_start]]

	# for each time step
	for t in xrange(0, 25):
		#Alice will conduct goal inference on observations of bob's location
		inferred_bob_goal = get_most_probable_goal_location(runner_model, poly_map, locs, sim_id, 
			bob_path, bob_start, t, "B")
		
		#Bob will conduct goal inference on observations of alice's location
		inferred_alice_goal = get_most_probable_goal_location(runner_model, poly_map, locs, sim_id, 
			alice_path, alice_start, t,"A")

		#Alice will move toward Bob's goal after planning
		alice_plan = run_rrt_opt( np.atleast_2d(alice_path[-1]), 
		np.atleast_2d(locs[inferred_bob_goal]), x1,y1,x2,y2 )
		alice_path.append(alice_plan[1])

		#Bob will move toward Alice's goal after planning
		bob_plan = run_rrt_opt( np.atleast_2d(bob_path[-1]), 
		np.atleast_2d(locs[inferred_alice_goal]), x1,y1,x2,y2 )
		bob_path.append(bob_plan[1])

		plot_movements(alice_path, bob_path, sim_id, poly_map, locs, t, code="double-goal")



def run_inference(trace, post_samples=16, samples=32):
	post_traces = []
	for i in  tqdm(xrange(post_samples)):
		#post_sample_trace = importance_sampling(trace, samples)
		post_sample_trace = importance_sampling(trace, samples)
		post_traces.append(post_sample_trace)
	return post_traces

def run_inference_MH(trace, post_samples=16, samples=32):
	post_traces = []
	for i in  tqdm(xrange(post_samples)):
		#post_sample_trace = importance_sampling(trace, samples)
		post_sample_trace = metroplis_hastings(trace, samples)
		post_traces.append(post_sample_trace)
	return post_traces


def line_plotting(inferrred_goals, sim_id, code="", directory="time"):

	#inferrred_goals =  [[0,1,2,3,4,5], [0,0,1,2,3,4], [0,0,0,1,2], [0,0,0,0,1], [0,0,0,0,0]]

	goal_probabilities = [[], [], [], [], [], []]
	for t in xrange(len(inferrred_goals)):
		inf_goals_at_t = inferrred_goals[t]
		total_num_inferences = len(inf_goals_at_t)
		# turn into percents
		for goal in xrange(6):
			goal_cnt = inf_goals_at_t.count(goal)
			goal_prob = goal_cnt / float(total_num_inferences)
			goal_probabilities[goal].append(goal_prob)

	print "goal_probabilities", goal_probabilities

	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i in xrange(len(goal_probabilities)):
		probs = goal_probabilities[i]
		ax.plot(probs, label="Goal " + str(i))
	ax.legend(loc='upper left')
	ax.set_ylabel('probability of goal')
	ax.set_xlabel('time step')
	fig.savefig( directory +'/' + sim_id + code + '_infering_goals.eps')

def run_inference_PO(locs, poly_map, isovist):
	runner_model = BasicRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist)
	Q = ProgramTrace(runner_model)
	Q.condition("run_start", 2)
	#Q.condition("run_goal", 6)
	Q.condition("other_run_start", 5)
	Q.condition("same_goal", True)
	# Q.condition("other_run_goal", 8)
	t = 8
	Q.condition("t", t)
	other_agent_path = [[0.42499999999999999, 0.40900000000000003], [0.42637252682699389, 0.45015681250166928], [0.42341590262456913, 0.4990041011765457], [0.4379044695823372, 0.53904127684124392], [0.45788032502031573, 0.55940630406732961], [0.50799614070063526, 0.55464417160785029], [0.54654058973382436, 0.5808784272257409], [0.55952256198212424, 0.61299850534228906], [0.583496600005313, 0.65140274446045765], [0.6005654644648678, 0.69540383781443371], [0.61589998114275823, 0.72963695272460394], [0.63340952989938293, 0.77334914393651799], [0.63415333146793929, 0.81788817596701702], [0.63380864206140508, 0.85068496120846326], [0.6574056904956691, 0.89840736011907418], [0.67365232860101742, 0.92626170086940862], [0.67272997642459853, 0.92476136486233518], [0.670977615078586, 0.9256403675603595], [0.67335036075493437, 0.92761813409888549], [0.67196251574528953, 0.92641384269803029], [0.67098919832850146, 0.92568243002410999], [0.67525871466320719, 0.9229149307167801], [0.67844855732075227, 0.92569329907319242], [0.67321881540381412, 0.93152732857257847], [0.67417822959121554, 0.9200035804784289], [0.67214122482315442, 0.92764945851654945], [0.67164236239673092, 0.92684256939487508], [0.67804513322197213, 0.92666665228204315], [0.6761799161969908, 0.92395809094667547], [0.67542733927721299, 0.92361716626504453], [0.67634231767336883, 0.92658004688912743], [0.67439978040601001, 0.92269389652124378], [0.67429561506189239, 0.92744438186414235], [0.67170106709439659, 0.92538278500943594], [0.67276872621711614, 0.91832099282710455], [0.67247922731692655, 0.92554210689024119], [0.67518132475675507, 0.92729301768019035], [0.6763367594646077, 0.92438316567133738], [0.67522650957951036, 0.92475789453983981], [ 0.675,  0.925]]
	for i in xrange(t):
	# 	Q.condition("other_run_x_"+str(i), other_agent_path[i][0])
	# 	Q.condition("other_run_y_"+str(i), other_agent_path[i][1])
		Q.condition("detected_t_"+str(i), False)
	for i in xrange(t, 24):
		Q.condition("detected_t_"+str(i), True)

	Q.condition("other_run_x_"+str(i), other_agent_path[7][0])
	Q.condition("other_run_y_"+str(i), other_agent_path[7][1])
	PS = 5
	SP = 32
	post_sample_traces = run_inference(Q, post_samples=PS, samples=SP)

	fig, ax = setup_plot(poly_map, locs)

	for trace in post_sample_traces:

		# draw field of views first 
		# for i in trace["t_detected"]:
		# 	intersections = trace["intersections-t-"+str(i)]
		# 	# show last isovist
		# 	if not intersections is None:
		# 		intersections = np.asarray(intersections)
		# 		intersections /= 500.0
		# 		if not intersections.shape[0] == 0:
		# 			patches = [ Polygon(intersections, True)]
		# 			p = PatchCollection(patches, cmap=matplotlib.cm.Set2, alpha=0.2)
		# 			colors = 100*np.random.rand(len(patches))
		# 			p.set_array(np.array(colors))
		# 			ax.add_collection(p)

		# draw agent's plan (past in orange and future in grey)
		path = trace["my_plan"]
		t = trace["t"]
		for i in range(0, t):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'orange', linestyle=":", linewidth=1, label="Agent's Plan")
		# for i in range(t, 39):
		# 	ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
		# 		'grey', linestyle=":", linewidth=1)
			
		# mark the runner at time t on its plan
		ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='orange')

		path = trace["other_plan"]

		for i in range(0, t):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'blue', linestyle="--", linewidth=1, label="Other's Plan")
			if i in trace["t_detected"]:
				ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')
			# else:
			# 	ax.scatter( path[i][0],  path[i][1] , s = 30, facecolors='none', edgecolors='grey')
		# mark the runner at time t on its plan
		ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='blue')

		for i in range(t, 39):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'grey', linestyle="--", linewidth=1, label="Other's Plan")
			if i in trace["t_detected"]:
				ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')

	close_plot(fig, ax, plot_name="PO_forward_runs/unknown_inference/IS_run_and_find-"+str(PS)+"-"+str(SP)+"-"+str(int(time.time()))+".eps")


def get_most_probable_goal_location(Q, poly_map, locs, sim_id, other_true_path, character, 
	directory="find_eachother", PS=1, SP=1):
	
	fig, ax = setup_plot(poly_map, locs)
	
	post_sample_traces = run_inference(Q, post_samples=PS, samples=SP)

	goal_list = []
	# show post sample traces on map
	for trace in post_sample_traces:
		inferred_goal = trace["other_run_goal"]
		goal_list.append(inferred_goal)
		# draw agent's plan (past in orange and future in grey)
		path = trace["my_plan"]
		t = trace["t"]
		for i in range(0, t):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'orange', linestyle=":", linewidth=1, label="Agent's Plan")
		# mark the runner at time t on its plan
		ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='orange')
		path = trace["other_plan"]
		for i in range(0, t):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'blue', linestyle="--", linewidth=1, label="Other's Plan")
			if i in trace["t_detected"]:
				ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')
		ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='blue')
		for i in range(t, 39):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'grey', linestyle="--", linewidth=1, label="Other's Plan")
			if i in trace["t_detected"]:
				ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')

	# show true path of other agent
	for i in range(len(other_true_path)-1):
			ax.plot( [other_true_path[i][0], other_true_path[i+1][0] ], [ other_true_path[i][1], other_true_path[i+1][1]], 
				'red', linestyle="--", linewidth=1, alpha = 0.75)
	ax.scatter( other_true_path[t][0],  other_true_path[t][1] , s = 95, facecolors='none', edgecolors='red')

	close_plot(fig, ax, plot_name="PO_forward_runs/"+directory+"/"+sim_id+"/finding-"+character+"-t-"+str(t)+".eps")


	# list with probability for each goal
	goal_probabilities = []
	total_num_inferences = len(goal_list)
	# turn into percents
	for goal in xrange(6):
		goal_cnt = goal_list.count(goal)
		goal_prob = goal_cnt / float(total_num_inferences)
		goal_probabilities.append(goal_prob)

	return goal_probabilities.index(max(goal_probabilities))



# inferred_bob_goal = get_most_detected_goal_PO(Q, poly_map, locs, sim_id, 
# 			bob_path, "B", directory=directory, PS=PS, SP=SP)
def get_most_detected_goal_PO(Q, poly_map, locs, sim_id, other_true_path, character, directory="find_eachother", PS=1, SP=1):
	
	post_sample_traces = run_inference(Q, post_samples=PS, samples=SP)

	fig, ax = setup_plot(poly_map, locs)

	detected_count = []
	inferred_goal = []
	for trace in post_sample_traces:
		d_list = trace["t_detected"]
		detected_count.append(len(d_list))

		if directory == "find_eachother":
			my_inferred_goal = trace["run_goal"]
		if directory == "tom_find_eachother":
			my_inferred_goal = trace["init_run_goal"]
		inferred_goal.append(my_inferred_goal)
		# draw agent's plan (past in orange and future in grey)
		path = trace["my_plan"]
		t = trace["t"]
		for i in range(0, t):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'orange', linestyle=":", linewidth=1, label="Agent's Plan")
		# mark the runner at time t on its plan
		ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='orange')
		path = trace["other_plan"]
		for i in range(0, t):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'blue', linestyle="--", linewidth=1, label="Other's Plan")
			if i in trace["t_detected"]:
				ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')
		ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='blue')
		for i in range(t, 39):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'grey', linestyle="--", linewidth=1, label="Other's Plan")
			if i in trace["t_detected"]:
				ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')

	# show true path of other agent
	for i in range(len(other_true_path)-1):
			ax.plot( [other_true_path[i][0], other_true_path[i+1][0] ], [ other_true_path[i][1], other_true_path[i+1][1]], 
				'red', linestyle="--", linewidth=1, alpha = 0.75)
	ax.scatter( other_true_path[t][0],  other_true_path[t][1] , s = 95, facecolors='none', edgecolors='red')

	close_plot(fig, ax, plot_name="PO_forward_runs/"+directory+"/"+sim_id+"/finding-"+character+"-t-"+str(t)+".eps")

	return inferred_goal[detected_count.index(max(detected_count))]


def get_others_goal_at_most_detected_PO(Q, poly_map, locs, sim_id, other_true_path, character, directory="find_eachother", PS=1, SP=1):
	
	post_sample_traces = run_inference(Q, post_samples=PS, samples=SP)

	fig, ax = setup_plot(poly_map, locs)

	detected_count = []
	inferred_goal = []
	for trace in post_sample_traces:
		d_list = trace["t_detected"]
		detected_count.append(len(d_list))
		other_inferred_goal = trace["other_run_goal"]
		inferred_goal.append(other_inferred_goal)
		# draw agent's plan (past in orange and future in grey)
		path = trace["my_plan"]
		t = trace["t"]
		for i in range(0, t):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'orange', linestyle=":", linewidth=1, label="Agent's Plan")
		# mark the runner at time t on its plan
		ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='orange')
		path = trace["other_plan"]
		for i in range(0, t):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'blue', linestyle="--", linewidth=1, label="Other's Plan")
			if i in trace["t_detected"]:
				ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')
		ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='blue')
		for i in range(t, 39):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'grey', linestyle="--", linewidth=1, label="Other's Plan")
			if i in trace["t_detected"]:
				ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')

	# show true path of other agent
	for i in range(len(other_true_path)-1):
			ax.plot( [other_true_path[i][0], other_true_path[i+1][0] ], [ other_true_path[i][1], other_true_path[i+1][1]], 
				'red', linestyle="--", linewidth=1, alpha = 0.75)
	ax.scatter( other_true_path[t][0],  other_true_path[t][1] , s = 95, facecolors='none', edgecolors='red')

	close_plot(fig, ax, plot_name="PO_forward_runs/"+directory+"/"+sim_id+"/finding-"+character+"-t-"+str(t)+".eps")

	return inferred_goal[detected_count.index(max(detected_count))]

# Q = condition_PO_model(runner_model, alice_start, bob_start, 
# 				t, alice_path, alices_detections, alices_detection_locs_of_bob)

def reverse_condition_PO_model(runner_model, start, other_start, t, path, past_obs, detection_locs_of_other, FULL=False):
	Q = ProgramTrace(runner_model)
	Q.condition("run_start", start)
	Q.condition("other_run_start", other_start)
	Q.condition("t", t) 
	# condition on previous time steps
	for prev_t in xrange(t):
		if prev_t == (t-1):
			Q.condition("other_run_x_"+str(prev_t), path[prev_t][0])
			Q.condition("other_run_y_"+str(prev_t), path[prev_t][1])
		Q.condition("detected_t_"+str(prev_t), past_obs[prev_t])
		if past_obs[prev_t] == True:
			Q.condition("run_x_"+str(prev_t), detection_locs_of_other[prev_t][0])
			Q.condition("run_y_"+str(prev_t), detection_locs_of_other[prev_t][1])
		elif FULL:
			Q.condition("run_x_"+str(prev_t), detection_locs_of_other[prev_t][0])
			Q.condition("run_y_"+str(prev_t), detection_locs_of_other[prev_t][1])

	for i in xrange(t, 40):
		Q.condition("detected_t_"+str(i), True)
	return Q



def condition_PO_model(runner_model, start, other_start, t, path, past_obs, detection_locs_of_other, FULL=False):
	Q = ProgramTrace(runner_model)
	Q.condition("run_start", start)
	Q.condition("other_run_start", other_start)
	Q.condition("t", t)
	Q.condition("same_goal", True)
	# condition on previous time steps
	for prev_t in xrange(t):
		#if prev_t == (t-1):
		Q.condition("run_x_"+str(prev_t), path[prev_t][0])
		Q.condition("run_y_"+str(prev_t), path[prev_t][1])
		Q.condition("detected_t_"+str(prev_t), past_obs[prev_t])
		if past_obs[prev_t] == True:
			Q.condition("other_run_x_"+str(prev_t), detection_locs_of_other[prev_t][0])
			Q.condition("other_run_y_"+str(prev_t), detection_locs_of_other[prev_t][1])
		elif FULL:
			Q.condition("other_run_x_"+str(prev_t), detection_locs_of_other[prev_t][0])
			Q.condition("other_run_y_"+str(prev_t), detection_locs_of_other[prev_t][1])

	for i in xrange(t, 40):
		Q.condition("detected_t_"+str(i), True)
	return Q

# Q.set_obs("other_x_"+str(7), other_plan[7][0])
# Q.set_obs("other_y_"+str(7), other_plan[7][1])
def condition_TOM_PO_model(runner_model, start, other_start, t, path, past_obs, detection_locs_of_other):
	Q = ProgramTrace(runner_model)
	Q.condition("init_run_start", start)
	Q.set_obs("other_run_start", other_start)
	Q.condition("t", t)
	Q.condition("same_goal", True)
	for i in xrange(t):
		if i == (t-1):
			Q.condition("init_run_x_"+str(i), path[i][0])
			Q.condition("init_run_y_"+str(i), path[i][1])
		Q.condition("detected_t_"+str(i), past_obs[i])
		Q.set_obs("detected_t_"+str(i), past_obs[i])
		if past_obs[i] == True:
			Q.set_obs("other_x_"+str(i), detection_locs_of_other[i][0])
			Q.set_obs("other_y_"+str(i), detection_locs_of_other[i][1])
	for i in xrange(t, 24):
		Q.condition("detected_t_"+str(i), True)
		Q.set_obs("detected_t_"+str(i), True)
	return Q


#
# Realize that I didn't change the condition if they do happen to see eachother
# Need to keep track of whether they saw each other or not
# ...
#
#
def simulate_find_eachother_PO(runner_model, locs, poly_map, isovist, directory="find_eachother", PS=1, SP=1):
	x1,y1,x2,y2 = poly_map
	sim_id = str(int(time.time()))

	# personal machine
	#newpath = "/Users/Iris/Documents/Repos/tom-games/PO_forward_runs/"+directory+"/"+str(sim_id) 
	# lab machine
	newpath = "/home/iris/Desktop/tom-games/PO_forward_runs/"+directory+"/"+str(sim_id) 
	
	if not os.path.exists(newpath):
	    os.makedirs(newpath)

	#Alice will start at some location
	alice_start = 4
	alice_path = [locs[alice_start]]

	#Bob will start st some other location
	bob_start = 8
	bob_path = [locs[bob_start]]

	alices_detections = {}
	bobs_detections = {}

	#TODO: still have to add them into condition statements
	alices_detection_locs_of_bob = {}
	bobs_detection_locs_of_alice = {}

	FULL = False
	# for each time step
	for t in xrange(0, 26):
		#Alice will conduct goal inference on observations of bob's location
		if directory == "tom_find_eachother":
			Q = condition_TOM_PO_model(runner_model, alice_start, bob_start, 
				t, alice_path, alices_detections, alices_detection_locs_of_bob)
		else:	
			Q = condition_PO_model(runner_model, alice_start, bob_start, 
				t, alice_path, alices_detections, alices_detection_locs_of_bob, FULL=FULL)

		#get_most_detected_goal_PO
		inferred_bob_goal = get_most_probable_goal_location(Q, poly_map, locs, sim_id, 
			bob_path, "B", directory=directory, PS=PS, SP=SP)

		#Bob will conduct goal inference on observations of alice's location
		if directory == "tom_find_eachother":
			Q = condition_TOM_PO_model(runner_model, bob_start, alice_start, 
				t, bob_path, bobs_detections, bobs_detection_locs_of_alice)
		else:
			Q = condition_PO_model(runner_model, bob_start, alice_start, 
				t, bob_path, bobs_detections, bobs_detection_locs_of_alice, FULL=FULL)

		#get_most_detected_goal_PO
		inferred_alice_goal = get_most_probable_goal_location(Q, poly_map, locs, sim_id, 
			alice_path, "A", directory=directory, PS=PS, SP=SP)

		#Alice will move toward Bob's goal after planning
		alice_plan = run_rrt_opt( np.atleast_2d(alice_path[-1]), 
		np.atleast_2d(locs[inferred_bob_goal]), x1,y1,x2,y2 )
		alice_path.append(alice_plan[1])

		#Bob will move toward Alice's goal after planning
		bob_plan = run_rrt_opt( np.atleast_2d(bob_path[-1]), 
		np.atleast_2d(locs[inferred_alice_goal]), x1,y1,x2,y2 )
		bob_path.append(bob_plan[1])

		
		bobs_step = bob_path[-1]
		alices_step = alice_path[-1]

		alice_detected_bob = was_other_detected(alices_step, bobs_step, isovist)
		bob_detected_alice = was_other_detected(bobs_step, alices_step, isovist)

		alices_detections[t] = alice_detected_bob
		bobs_detections[t] = bob_detected_alice

		alices_detection_locs_of_bob[t] = bobs_step
		bobs_detection_locs_of_alice[t] = alices_step

		# need to store the locations that they were located so they can condition on them




		plot_movements(alice_path, bob_path, sim_id, poly_map, locs, t, code="PO-find_eachother", directory="PO_forward_runs/"+directory+"/"+sim_id)

def was_other_detected(agent_loc, other_agent_loc, isovist):
	#Check if they saw one another
	loc = scale_up(agent_loc)
	other_loc = scale_up(other_agent_loc)

	fv = direction(other_loc, loc)
	intersections = isovist.GetIsovistIntersections(loc, fv)
	is_other_seen = isovist.FindIntruderAtPoint(other_loc, intersections)
	return is_other_seen



def run_conditioned_basic_partial_model(locs, poly_map, isovist):
	runner_model = BasicRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist)
	Q = ProgramTrace(runner_model)
	Q.condition("run_start", 2)
	#Q.condition("run_goal", 6)
	Q.condition("other_run_start", 5)
	# Q.condition("other_run_goal", 8)
	Q.condition("same_goal", False)
	t = 5
	Q.condition("t", t)
	other_agent_path = [[0.42499999999999999, 0.40900000000000003], [0.42637252682699389, 0.45015681250166928], [0.42341590262456913, 0.4990041011765457], [0.4379044695823372, 0.53904127684124392], [0.45788032502031573, 0.55940630406732961], [0.50799614070063526, 0.55464417160785029], [0.54654058973382436, 0.5808784272257409], [0.55952256198212424, 0.61299850534228906], [0.583496600005313, 0.65140274446045765], [0.6005654644648678, 0.69540383781443371], [0.61589998114275823, 0.72963695272460394], [0.63340952989938293, 0.77334914393651799], [0.63415333146793929, 0.81788817596701702], [0.63380864206140508, 0.85068496120846326], [0.6574056904956691, 0.89840736011907418], [0.67365232860101742, 0.92626170086940862], [0.67272997642459853, 0.92476136486233518], [0.670977615078586, 0.9256403675603595], [0.67335036075493437, 0.92761813409888549], [0.67196251574528953, 0.92641384269803029], [0.67098919832850146, 0.92568243002410999], [0.67525871466320719, 0.9229149307167801], [0.67844855732075227, 0.92569329907319242], [0.67321881540381412, 0.93152732857257847], [0.67417822959121554, 0.9200035804784289], [0.67214122482315442, 0.92764945851654945], [0.67164236239673092, 0.92684256939487508], [0.67804513322197213, 0.92666665228204315], [0.6761799161969908, 0.92395809094667547], [0.67542733927721299, 0.92361716626504453], [0.67634231767336883, 0.92658004688912743], [0.67439978040601001, 0.92269389652124378], [0.67429561506189239, 0.92744438186414235], [0.67170106709439659, 0.92538278500943594], [0.67276872621711614, 0.91832099282710455], [0.67247922731692655, 0.92554210689024119], [0.67518132475675507, 0.92729301768019035], [0.6763367594646077, 0.92438316567133738], [0.67522650957951036, 0.92475789453983981], [ 0.675,  0.925]]
	for i in xrange(t):
		Q.condition("other_run_x_"+str(i), other_agent_path[i][0])
		Q.condition("other_run_y_"+str(i), other_agent_path[i][1])
		Q.condition("detected_t_"+str(i), False)
	for i in xrange(t, 24):
		Q.condition("detected_t_"+str(i), False)

	score, trace = Q.run_model()

	fig, ax = setup_plot(poly_map, locs, )

	for i in trace["t_detected"]:
		intersections = trace["intersections-t-"+str(i)]
		# show last isovist
		if not intersections is None:
			intersections = np.asarray(intersections)
			intersections /= 500.0
			if not intersections.shape[0] == 0:
				patches = [ Polygon(intersections, True)]
				p = PatchCollection(patches, cmap=matplotlib.cm.Set2, alpha=0.2)
				colors = 100*np.random.rand(len(patches))
				p.set_array(np.array(colors))
				ax.add_collection(p)



	path = trace["my_plan"]
	t = trace["t"]
	for i in range(0, t):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'orange', linestyle=":", linewidth=1, label="Agent's Plan")
	for i in range(t, 39):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'grey', linestyle=":", linewidth=1)
		


	# mark the runner at time t on its plan
	ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='orange')

	path = trace["other_plan"]

	for i in range(0, t):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'blue', linestyle="--", linewidth=1, label="Other's Plan")
		if i in trace["t_detected"]:
			ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')
		# else:
		# 	ax.scatter( path[i][0],  path[i][1] , s = 30, facecolors='none', edgecolors='grey')
	# mark the runner at time t on its plan
	ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='blue')

	for i in range(t, 39):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'grey', linestyle="--", linewidth=1, label="Other's Plan")
		if i in trace["t_detected"]:
			ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')

	plt.figtext(0.92, 0.85, "Values", horizontalalignment='left', weight="bold") 
	plt.figtext(0.92, 0.80, "A Start: " +str(trace["run_start"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.75, "A Goal: " +str(trace["run_goal"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.70, "B Start: " +str(trace["other_run_start"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.65, "B Goal: " +str(trace["other_run_goal"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.60, "time step: " +str(trace["t"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.55, "A detected B count: " +str(len(trace["t_detected"])), horizontalalignment='left') 
	
	plt.figtext(0.5, 0.01, "Log Score: " +str(score), horizontalalignment='center') 
	close_plot(fig, ax, plot_name="PO_forward_runs/conditioned/single_samples/run_and_avoid-"+str(int(time.time()))+".eps")
	print "score:", score

def run_unconditioned_basic_partial_model(locs, poly_map, isovist):
	runner_model = BasicRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist)
	Q = ProgramTrace(runner_model)

	score, trace = Q.run_model()

	fig, ax = setup_plot(poly_map, locs, )

	for i in trace["t_detected"]:
		intersections = trace["intersections-t-"+str(i)]
		# show last isovist
		if not intersections is None:
			intersections = np.asarray(intersections)
			intersections /= 500.0
			if not intersections.shape[0] == 0:
				patches = [ Polygon(intersections, True)]
				p = PatchCollection(patches, cmap=matplotlib.cm.Set2, alpha=0.2)
				colors = 100*np.random.rand(len(patches))
				p.set_array(np.array(colors))
				ax.add_collection(p)



	path = trace["my_plan"]
	t = trace["t"]
	for i in range(0, t):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'orange', linestyle=":", linewidth=1, label="Agent's Plan")
	for i in range(t, 39):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'grey', linestyle=":", linewidth=1)
		


	# mark the runner at time t on its plan
	ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='orange')

	path = trace["other_plan"]

	for i in range(0, t):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'blue', linestyle="--", linewidth=1, label="Other's Plan")
		if i in trace["t_detected"]:
			ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')
		# else:
		# 	ax.scatter( path[i][0],  path[i][1] , s = 30, facecolors='none', edgecolors='grey')
	# mark the runner at time t on its plan
	ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='blue')

	for i in range(t, 39):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'grey', linestyle="--", linewidth=1, label="Other's Plan")
		if i in trace["t_detected"]:
			ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')

	plt.figtext(0.92, 0.85, "Sampled Values", horizontalalignment='left', weight="bold") 
	plt.figtext(0.92, 0.80, "A Start: " +str(trace["run_start"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.75, "A Goal: " +str(trace["run_goal"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.70, "B Start: " +str(trace["other_run_start"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.65, "B Goal: " +str(trace["other_run_goal"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.60, "time step: " +str(trace["t"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.55, "A detected B count: " +str(len(trace["t_detected"])), horizontalalignment='left') 
	close_plot(fig, ax, plot_name="PO_forward_runs/unconditioned/single_samples/run_and_find-"+str(int(time.time()))+".eps")

	print "time:", trace["t"]
	# print "other_run_start:", trace["other_run_start"]
	# print "other_run_goal:", trace["other_run_goal"]

	print "run_start:", trace["run_start"]
	print "run_goal:", trace["run_goal"]
	print "score:", score

	#print "detected:", trace["detected"]
	print "other_plan:", trace["other_plan"]
	print "my_plan:", trace["my_plan"]
	print "detected times:", trace["t_detected"]


def run_conditioned_tom_partial_model(runner_model, locs, poly_map, isovist, PS=1, SP=1):
	Q = ProgramTrace(runner_model)

	Q.condition("init_run_start", 1)
	Q.set_obs("other_run_start", 8)
	t = 8
	Q.condition("t", t)
	Q.condition("same_goal", True)

	my_plan = [[0.56599999999999995, 0.14600000000000002], [0.55556809280738129, 0.18721437976543853], [0.5445341110551255, 0.23050610975954094], [0.53610663650986889, 0.27570707994484339], [0.51838817809125204, 0.31364677053850654], [0.51518717782727763, 0.35497127647386223], [0.55530733907318341, 0.38068397671159959], [0.56603426039645688, 0.420586401074451], [0.57876353334580199, 0.46649009155864785], [0.5894309668616603, 0.50360276514056979]]
	other_plan = [[0.67500000000000004, 0.92500000000000004], [0.65595794760796733, 0.88364953074871666], [0.63593639118326906, 0.84460049053946695], [0.632454561514043, 0.80177855079821769], [0.62322367768018871, 0.76592206121988504], [0.61070805265174255, 0.72506794767846527], [0.58606327906428324, 0.68850497091510821], [0.56369666918811434, 0.65184792821656079], [0.53867971062239195, 0.61822274529712795], [0.51418992415138365, 0.57765462476393603]]
	detections = [False, False, False, False, False, False, False, True]
	#detection_locs = {}
	# just for thi scenario
	for i in xrange(t):
		Q.condition("detected_t_"+str(i), detections[i])
		Q.set_obs("detected_t_"+str(i), detections[i])
		#only condition my my previous time step
		if (i==(t-1)):
			Q.condition("init_run_x_"+str(i), my_plan[i][0])
			Q.condition("init_run_y_"+str(i), my_plan[i][1])
	for i in xrange(t, 26):
		Q.condition("detected_t_"+str(i), True)
		Q.set_obs("detected_t_"+str(i), True)

	Q.set_obs("other_x_"+str(7), other_plan[7][0])
	Q.set_obs("other_y_"+str(7), other_plan[7][1])

	#score, trace = Q.run_model()
	post_sample_traces = run_inference(Q, post_samples=PS, samples=SP)

	trace = post_sample_traces[0]

	fig, ax = setup_plot(poly_map, locs)

	print trace["t_detected"]
	# for i in trace["t_detected"]:
	# 	intersections = trace["intersections-t-"+str(i)]
	# 	# show last isovist
	# 	if not intersections is None:
	# 		intersections = np.asarray(intersections)
	# 		intersections /= 500.0
	# 		if not intersections.shape[0] == 0:
	# 			patches = [ Polygon(intersections, True)]
	# 			p = PatchCollection(patches, cmap=matplotlib.cm.Set2, alpha=0.2)
	# 			colors = 100*np.random.rand(len(patches))
	# 			p.set_array(np.array(colors))
	# 			ax.add_collection(p)

	nested_post_samples = trace["nested_post_samples"]
	for nested_trace in nested_post_samples:
		print "HERE"
		path = nested_trace["my_plan"]
		for i in range(t-1, 39):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'red', linestyle="-", linewidth=1, label="Other's Plan", alpha=0.2)
			if i in trace["t_detected"]:
				ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')

	path = trace["my_plan"]
	t = trace["t"]
	for i in range(0, t-1):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'orange', linestyle=":", linewidth=1, label="Agent's Plan")
	for i in range(t-1, 39):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'grey', linestyle=":", linewidth=1)


	# mark the runner at time t on its plan
	ax.scatter( path[t-1][0],  path[t-1][1] , s = 95, facecolors='none', edgecolors='orange')

	path = trace["other_plan"]

	for i in range(0, t-1):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'blue', linestyle="--", linewidth=1, label="Other's Plan")
		if i in trace["t_detected"]:
			ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='blue')
		# else:
		# 	ax.scatter( path[i][0],  path[i][1] , s = 30, facecolors='none', edgecolors='grey')
	# mark the runner at time t on its plan
	ax.scatter( path[t-1][0],  path[t-1][1] , s = 95, facecolors='none', edgecolors='blue')

	for i in range(t, 39):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'blue', linestyle="--", linewidth=1, label="Other's Plan")
		if i in trace["t_detected"]:
			ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')

	

	plt.figtext(0.92, 0.85, "Values", horizontalalignment='left', weight="bold") 
	plt.figtext(0.92, 0.80, "A Start: " +str(trace["init_run_start"]), horizontalalignment='left') 
	#plt.figtext(0.92, 0.75, "A Goal: " +str(trace["init_run_goal"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.75, "B Start: " +str(trace["other_run_start"]), horizontalalignment='left') 
	#plt.figtext(0.92, 0.65, "B Goal: " +str(trace["other_run_goal"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.70, "time step: " +str(trace["t"]), horizontalalignment='left') 
	plt.figtext(0.92, 0.65, "A detected B count: " +str(len(trace["t_detected"])), horizontalalignment='left') 
	#close_plot(fig, ax, plot_name="PO_forward_runs/unconditioned/single_samples/tom/tom_run_and_find-"+str(int(time.time()))+".eps")
	close_plot(fig, ax, 
		plot_name="PO_forward_runs/conditioned/single_samples/tom/IS_tom_run_and_find-P"+
		str(PS)+"-S"+str(SP)+"-"+str(int(time.time()))+".eps")
	
	print "time:", trace["t"]




if __name__ == '__main__':
	#plot("test.eps")
	
	locs = None
	poly_map = None
	isovist = None

	map_info = 2

	# ------------- setup for map 2 ---------------
	if map_info == 1:
		locs_map_1 = [[0.4, 0.062] ,
				[0.964, 0.064] ,
				[0.442, 0.37] ,
				[0.1, 0.95] ,
				[0.946, 0.90] ,
				[0.066, 0.538]]
		locs = locs_map_1
		poly_map  = polygons_to_segments( load_polygons( "./map_2.txt" ) )
		isovist = i.Isovist( load_isovist_map( fn="./map_2.txt" ) )

	# ------------- setup for map "paths" large bremen map ---------------
	if map_info == 2:
		locs = [[ 0.100, 1-0.900 ],[ 0.566, 1-0.854 ],[ 0.761, 1-0.665 ],
			[ 0.523, 1-0.604 ],[ 0.241, 1-0.660 ],[ 0.425, 1-0.591 ],
			[ 0.303, 1-0.429 ],[ 0.815, 1-0.402 ],[ 0.675, 1-0.075 ],
			[ 0.432, 1-0.098 ] ]
		poly_map = polygons_to_segments( load_polygons( "./paths.txt" ) )
		isovist = i.Isovist( load_isovist_map() )

	#plots the map and the locations if said so in the function
	plot(poly_map, plot_name="large_map_blank.eps", locs=locs)
	


	#---------------- Basic Runner model -------------------------------
	#runner_model = BasicRunner(seg_map=poly_map, locs=locs, isovist=isovist)

	# simple forward runs of the basic runner model
	#run_basic_forward(runner_model, poly_map, locs)

	# --------- run goal inference on new observations ----------------
	# inferrred_goals, sim_id= goal_inference_while_moving(runner_model, poly_map, locs)
	# line_plotting(inferrred_goals, sim_id)

	# --------- follow the leader using goal inference tools ----------
	# follow_the_leader_goal_inference(runner_model, poly_map, locs)

	# --------- first experiment of agent "collaboration" -------------
	#two_agent_goal_inference_while_moving(runner_model, poly_map, locs)

	#---------- nested collab experiment ------------------------------
	#tom_runner_model = TOMCollabRunner(seg_map=poly_map, locs=locs, isovist=isovist, nested_model=runner_model)
	#two_agent_nested_goal_inference_while_moving(tom_runner_model, poly_map, locs)

	# ---------- plot generative model of simple runner model
	# Q = ProgramTrace(runner_model)
	# Q.condition("run_start", 4)
	# Q.condition("run_goal", 0)
	# score, trace = Q.run_model()
	# plot_runner(poly_map, trace, locs=locs)
	# print("time:", trace["t"])
	# print("start:", trace["run_start"])
	# print("goal:", trace["run_goal"])

	#-----------run basic partially observable model and plot----------
	#run_conditioned_basic_partial_model(locs, poly_map, isovist)
	#run_unconditioned_basic_partial_model(locs, poly_map, isovist)

	#-----------run basic PO model conditioned on other_path, start, goal, and t ---
	#run_inference_PO(locs, poly_map, isovist)

	# -----------run basic partially observable model - SIMULATE FIND EACHOTHER ----
	# runner_model = BasicRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist)
	# simulate_find_eachother_PO(runner_model, locs, poly_map, isovist, directory="find_eachother", PS=3, SP=32)

	#-----------run TOM partially observable model ------
	runner_model = BasicRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist)
	tom_runner_model = TOMRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist, 
		nested_model=runner_model, ps=5, sp=32)
	#-- run single conditioned sample ---//
	#run_conditioned_tom_partial_model(tom_runner_model, locs, poly_map, isovist, PS=5, SP=32)

	simulate_find_eachother_PO(tom_runner_model, locs, poly_map, isovist, 
		directory="tom_find_eachother", PS=5, SP=32)













