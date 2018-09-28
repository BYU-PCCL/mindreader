from my_rrt import *
from methods import dist

ax = None
	
#(2000, 3.0, 10000, 1.)
def optimize_path(x1, y1, x2, y2, orig_path, iters, std):
	points = orig_path[:]
	inner_pt_cnt = len(points)-2
	if inner_pt_cnt == 0:
		return orig_path

	for i in xrange(iters):
		pt_i = 2 + (i % inner_pt_cnt)

		if pt_i < 1:
			continue
		if pt_i >= len(orig_path) - 1:
			continue

		pre_pt = points[pt_i-1]
		pt = points[pt_i]
		next_pt = points[pt_i+1]

		ad_pt = [pt[0] + np.random.randn() * std, pt[1] + np.random.randn() * std]
		curr_dist = dist(pre_pt, pt) + dist(pt, next_pt)
		

		back_int_x, back_int_y, back_intersection_indicators = line_intersect( 
	            pre_pt[0], 
	            pre_pt[1], 
	            ad_pt[0], 
	            ad_pt[1], 
	            x1, y1, x2, y2)
		forward_int_x, forward_int_y, forward_intersection_indicators = line_intersect( 
	            ad_pt[0], 
	            ad_pt[1], 
	            next_pt[0], 
	            next_pt[1], 
	            x1, y1, x2, y2)
		
		back_ok = not back_intersection_indicators.any()
		front_ok = not forward_intersection_indicators.any()

		new_dist = 0.0

		if back_ok:
			if front_ok:
				new_dist = dist(pre_pt, ad_pt) + dist(ad_pt, next_pt)
	    		if new_dist < curr_dist:
	    			points[pt_i] = ad_pt
	    			

	return points

def walk_path(path, speed, times):
	distances_from_start = [0.0]*len(path)
	for i in xrange(1, len(path)):
		distances_from_start[i] = distances_from_start[i-1]+ dist(path[i-1], path[i])

	locations = [-1]*len(times)
	locations[0] = path[0]

	for time_i, t in enumerate(times):
		desired_distance = t * speed
		used_up_time = False
		for i in xrange (1, len(path)):
			prev = path[i-1] 
			cur = path[i] 
			dist_to_prev = dist(prev, cur)
			if distances_from_start[i] >= desired_distance:
				#we overshot, the location is beween i-1 and i
				overshoot = distances_from_start[i] - desired_distance
				past_prev = dist_to_prev - overshoot
				if dist_to_prev == 0:
					frac = 1
				else:
					frac = past_prev / dist_to_prev
				locations[time_i] = [prev[0] * (1.0 - frac) + cur[0] * frac, prev[1] * (1.0 - frac) + cur[1] * frac]
				used_up_time = True
				break
		if not used_up_time:
			locations[time_i] = path[-1]

	assert(len(locations) >= 29)
	return locations


def simplify_path(x1, y1, x2, y2, orig_path):
	points = [orig_path[0]]
	for i in xrange(1, len(orig_path)-1):
		last_pt = points[-1]
		next_pt = orig_path[i]
		int_x, int_y, intersection_indicators = line_intersect( 
	            next_pt[0], 
	            next_pt[1],
	            last_pt[0], 
	            last_pt[1], 
	            x1, y1, x2, y2)
		if intersection_indicators.any():
			print ("intersected")
			print ("i:", i)
			points.append(orig_path[i-1])
			points.append(orig_path[i])

	points.append(orig_path[-1])
	return points


def run_rrt_opt(start_pt, goal_pt, x1, y1, x2, y2, slow=False):
	path = []
	gb = .0001
	# if just_need_step:
	# 	gb = .3
	cnt = 0
	while True:
		path = run_rrt( start_pt, goal_pt, x1, y1, x2, y2, goal_buffer=gb)
		if not path is None:
			if len(path) > 1:
				break


			# 	gb += .002
			#  	print "expanding rrt goal buffer:", gb
			#  	cnt +=1
			#  	if cnt  >= 3:
			#  		return None # expressing that we are probably inside a building now
			# else:
			# 	break
	iters = 100
	std = 1.0/500
	opt_path = optimize_path(x1, y1, x2, y2, path, iters, std)
	sim_path = simplify_path(x1, y1, x2, y2, opt_path)
	#sim_path = simplify_path(x1, y1, x2, y2, path)

	times = np.arange(0, 600, 20) #600 seconds (10 minute) (5 mintues) 30 20-sec intervals
	# change ^ to be 800 seconds.. 40 20-sec intervals. Making the paths longer...
	
	slow = False
	slower = False
	fastest = False
	normal = True

	#speed = 1.75/600

	if slow:
		speed = 1.3/600

	if slower:
		speed = 0.9/600

	if normal:
		speed = 1.75/600

	if fastest:
		speed = 3.0/600

	walking_path = walk_path(sim_path, speed, times)
	#return sim_path
	return walking_path



def get_distances(x1, y1, x2, y2, start_pt, goal_pt):
    distances = []
    for i in xrange(100):
		path = run_rrt( start_pt, goal_pt, x1, y1, x2, y2)
		for j in xrange(len(path)-1):
			distances.append(dist(path[j], path[j+1]))
    return distances
	#print(np.mean(distances))




def close_plot(fig, ax, plot_name=None):
	if plot_name is None:
		plot_name = str(int(time.time()))+".eps"
	print "plot_name:", plot_name

	ax.set_ylim(ymax = 1, ymin = 0)
	ax.set_xlim(xmax = 1, xmin = 0)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	#plt.show()
	fig.savefig(str(int(time.time()))+".eps", bbox_inches='tight')


def _run():
	polygons = load_polygons( "./paths.txt" )
	x1, y1, x2, y2 = polygons_to_segments( polygons )


	start_pt = np.atleast_2d([ 0.241, 1-0.660 ] )
	goal_pt = np.atleast_2d( [ 0.675, 1-0.075 ] )

	#x1, y1, x2, y2 = polygons_to_segments( polygons )

	#path = run_rrt_poly( start_pt, goal_pt, polygons, heat = intensity, plot=False)


	 # Create figure and axes
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)

    #c = np.linspace(0, 10, np_paths.shape[0])
    #img = plt.imread("./cnts_inv.png")
    #ax.imshow(img)

	scale = 1

	# plot original RRT path
	# for i in range( 0, len(path)-1 ):
	#     ax.plot( [ path[i][0] * scale, path[i+1][0] * scale ], [ path[i][1] * scale, path[i+1][1] * scale], 'b' )

	for x in xrange(1):

		# nearest_point = [[0.65052348, 0.83421013]]
		# new_pt = [[0.69031952, 0.86448016]]

		# int_x, int_y, intersection_indicators = line_intersect( 
		# 	0, 
		# 	0, 
		# 	0, 
		# 	1, 
		# 	np.array([0]),np.array([0]),np.array([1]),np.array([0.5]))

		# if intersection_indicators.any():
		# 	print "INTERSECTED"
		# 	print 
		# new_path = [nearest_point, new_pt]

		path = run_rrt( start_pt, goal_pt, x1, y1, x2, y2)
		iters = 100
		std = 1.0/500
		#new_path =run_rrt_opt(start_pt, goal_pt, x1, y1, x2, y2)     
		# #new_path = optimize_path(x1, y1, x2, y2, path, iters, std)
		new_path = path
		# #plot optimized path
		for i in range( 0, len(new_path)-1 ):
			ax.plot( [ new_path[i][0] * scale, new_path[i+1][0] * scale ], [ new_path[i][1] * scale, new_path[i+1][1] * scale], 'green' )
			ax.scatter( new_path[i][0],  new_path[i][1] , s = 10, facecolors='none', edgecolors='red')
		# sim_path = simplify_path(x1, y1, x2, y2, new_path)

		# # plot simplification from original RRT path
		# # for i in range( 0, len(sim_path)-1 ):
		# # ax.plot( [ sim_path[i][0] * scale, sim_path[i+1][0] * scale ], [ sim_path[i][1] * scale, sim_path[i+1][1] * scale], 'limegreen' )

		# # sim_path = simplify_path(x1, y1, x2, y2, new_path)

		# # # plot simplification of optimized path
		# # for i in range( 0, len(sim_path)-1 ):
		# # 	ax.plot( [ sim_path[i][0] * scale, sim_path[i+1][0] * scale ], [ sim_path[i][1] * scale, sim_path[i+1][1] * scale], 'orange' )
		# # 	ax.scatter( sim_path[i][0],  sim_path[i][1] , s = 25, facecolors='none', edgecolors='red')
		# # ax.scatter( sim_path[len(sim_path)-1][0],  sim_path[len(sim_path)-1][1] , s = 25, facecolors='none', edgecolors='red')

	 # #    # ax.scatter( start_pt[0,0] * scale, start_pt[0,1]  * scale)
	 # #    # ax.scatter( goal_pt[0,0] * scale, goal_pt[0,1] * scale)
		# # plt.ylim((0,scale))

		for i in xrange(x1.shape[0]):
			ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'grey' )
			#fig.savefig(str(int(time.time()))+".eps", bbox_inches='tight')
			#print ("----------", i, x1[i,0], x2[i,0], y1[i,0],  y2[i,0] )


		# times = np.arange(0, 600, 20)
		# speed = 1.75/600
		# path =  walk_path(sim_path, speed, times)
		# for i in range(0, len(path)-1):
		# 	# ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
		# 	# 	'black', linestyle="-", linewidth=1, label="Agent's Plan")
		# 	ax.scatter( path[i][0],  path[i][1] , s = 10, facecolors='none', edgecolors='black')




		print new_path

	    # distances = get_distances(x1, y1, x2, y2, start_pt, goal_pt)
	    # print(np.mean(distances) * 500)
	    # print(np.max(distances) * 500)
	    # print(np.min(distances) * 500)
	    # #print(x1[1,0]) *
	    #print(x1.shape) #(323, 1)




	plt.show() #boom

import cProfile
import re
if __name__ == '__main__':
	_run()
	 # cProfile.run('_run()')

	# nearest_point = np.array([0.65052348, 0.83421013])
	# new_pt = np.array([0.69031952, 0.86448016])

	# safety = new_pt - nearest_point
	# print ("new_pt:", new_pt)
	# print ("nearest_point:", nearest_point)
	# print ("safety:", safety, np.sqrt( np.sum( safety*safety )))
	# safety = 1 * 0.001 * safety / np.sqrt( np.sum( safety*safety ) )
	# print ("sagety:", safety)
	# new_pt = new_pt - safety
	# print new_pt
	# 