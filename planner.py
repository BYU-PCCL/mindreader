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

		pre_pt = np.round(points[pt_i-1],3)
		pt = points[pt_i]
		next_pt = np.round(points[pt_i+1],3)

		ad_pt = np.round([pt[0] + np.random.randn() * std, pt[1] + np.random.randn() * std],3)
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

def walk_path(path, speed, times, x1, y1, x2, y2):
	distances_from_start = [0.0]*len(path)
	for i in xrange(1, len(path)):
		distances_from_start[i] = distances_from_start[i-1]+ dist(path[i-1], path[i])

	locations = [-1]*len(times)
	locations[0] = np.round(path[0],3)

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
				locations[time_i] = np.round([prev[0] * (1.0 - frac) + cur[0] * frac, prev[1] * (1.0 - frac) + cur[1] * frac],3)
				used_up_time = True
				break
		if not used_up_time:
			locations[time_i] = np.round(path[-1],3)

	assert(len(locations) >= 29)

	for i in xrange(len(locations)-1):



		if locations[i][0] == locations[i+1][0]:
			if locations[i][1] == locations[i+1][1]:
				continue

		int_x, int_y, intersection_indicators = line_intersect( 
			locations[i][0], 
			locations[i][1], 
			locations[i+1][0], 
			locations[i+1][1], 
			x1, y1, x2, y2)

		if intersection_indicators.any():
			#print "[[",locations[i][0],",", locations[i][1],"],[", locations[i+1][0],",",locations[i+1][1],"]],"
			#print("throwing out RRT")
			return []

	#print ("keeping location")
	#print locations
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
			#print ("intersected")
			#print ("i:", i)
			points.append(np.round(orig_path[i-1],3))
			points.append(np.round(orig_path[i],3))

	points.append(np.round(orig_path[-1],3))
	return points


def run_rrt_opt(start_pt, goal_pt, x1, y1, x2, y2):

	path = []
	gb = .0001
	attempt = 0
	while len(path) == 0:
		attempt+=1
		#print ("------ attempt:", attempt)
	# if just_need_step:
	# 	gb = .3
		cnt = 0
		count = 0
		orig_start_pt = start_pt
		while True:
			path = run_rrt( start_pt, goal_pt, x1, y1, x2, y2, goal_buffer=gb)
			if not path is None:
				if len(path) > 1:
					break
			count += 1
			# XXXX HACK add some noise to start location to get it out of building if stuck
			if count >= 10:
				print (count, ":ERROR:STUCK IN LOOP")
				mu = orig_start_pt 
				sigma = .003
				start_pt = np.round(mu + sigma*np.random.randn(),3)
				print ("Shifting:", orig_start_pt, " to:", start_pt)

			#if elapsed_time > 3.0

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

		walking_path = walk_path(sim_path, speed, times, x1, y1, x2, y2)
		path = walking_path

		if attempt >= 100:
			print (attempt, "ERROR: STUCK IN run_rrt_opt")
			print (sim_path)

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

	for i in xrange(x1.shape[0]):
		ax.plot( [ x1[i,0], x2[i,0] ], [ y1[i,0], y2[i,0]], 'grey' )
		# 	#fig.savefig(str(int(time.time()))+".eps", bbox_inches='tight')


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

		iters = 100
		std = 1.0/500
		path =run_rrt_opt(start_pt, goal_pt, x1, y1, x2, y2)     


		#------------ step by step paths ------------------
		path = run_rrt( start_pt, goal_pt, x1, y1, x2, y2)
		iters = 100
		std = 1.0/500

		

		new_path = optimize_path(x1, y1, x2, y2, path, iters, std)
		print (new_path)
		# # #plot optimized path
		# for i in range( 0, len(new_path)-1 ):
		# 	#ax.plot( [ new_path[i][0] * scale, new_path[i+1][0] * scale ], [ new_path[i][1] * scale, new_path[i+1][1] * scale], 'green' )
		# 	ax.scatter( new_path[i][0],  new_path[i][1] , s = 10, facecolors='none', edgecolors='blue')
		

		sim_path = simplify_path(x1, y1, x2, y2, new_path)
		print(sim_path)

	
		# #plot simplification of optimized path
		# for i in range( 0, len(sim_path)-1 ):
		# 	#ax.plot( [ sim_path[i][0] * scale, sim_path[i+1][0] * scale ], [ sim_path[i][1] * scale, sim_path[i+1][1] * scale], 'orange' )
		# 	ax.scatter( sim_path[i][0],  sim_path[i][1] , s = 25, facecolors='none', edgecolors='red')
		# ax.scatter( sim_path[len(sim_path)-1][0],  sim_path[len(sim_path)-1][1] , s = 25, facecolors='none', edgecolors='red')


		

		times = np.arange(0, 600, 20)
		speed = 1.75/600
		path =  walk_path(sim_path, speed, times, x1, y1, x2, y2)

		print(path)
		#---------------------------------------------------

		for i in range(0, len(path)-1):
			# ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			# 	'black', linestyle="-", linewidth=1, label="Agent's Plan")
			ax.scatter( path[i][0],  path[i][1] , s = 10, facecolors='none', edgecolors='black')




	plt.show() #boom

import cProfile
import re
if __name__ == '__main__':
	#_run()
	 # cProfile.run('_run()')

	 # testing intersections

	# intersections = [[[ 0.649919520401053 , 0.8535176950249271 ],[ 0.6659205901778404 , 0.9084518780701341 ]],
	# 	[[ 0.4802766790545866 , 0.5618623417444708 ],[ 0.5230883410699202 , 0.5934912215173018 ]],
	# 	[[ 0.3957978219564466 , 0.4622388885770752 ],[ 0.4259320392774147 , 0.5039636582409632 ]],
	# 	[[ 0.44034072890764325 , 0.4612678746449117 ],[ 0.4481969675838674 , 0.5185059650269881 ]],
	# 	[[ 0.5007328353330858 , 0.33873245405583635 ],[ 0.545359331125651 , 0.3757525497377037 ]],
	# 	[[ 0.4351362337479053 , 0.465139743128454 ],[ 0.44783382358250984 , 0.5220422746439279 ]]]

	x1, y1, x2, y2 = polygons_to_segments(  load_polygons( "./paths.txt" ) )

	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)

	for i in xrange(x1.shape[0]):
		ax.plot( [ x1[i,0], x2[i,0] ], [ y1[i,0], y2[i,0]], 'grey' )

	# for pair in intersections:
	# 	#print pair

	# 	ax.plot( [ pair[0][0], pair[1][0] ], [ pair[0][1], pair[1][1]], 'red' )

	pts = [[0.25394452, 0.28238767]]

	for i in range(0, len(pts)):
			# ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			# 	'black', linestyle="-", linewidth=1, label="Agent's Plan")
			ax.scatter( np.round(pts[i][0],3),  np.round(pts[i][1],3) , s = 5, facecolors='none', edgecolors='red')



	plt.show() #boom