from my_rrt import *
from methods import dist

ax = None
	
#(2000, 3.0, 10000, 1.)
def optimize_path(x1, y1, x2, y2, orig_path, iters, std):
	SCALE = 1000
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
		

		#ax.scatter( ad_pt[0] * 1000, ad_pt[1]  * 1000)
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

		if back_ok:
			if front_ok:
				new_dist = dist(pre_pt, ad_pt) + dist(ad_pt, next_pt)
	    		if new_dist < curr_dist:
	    			points[pt_i] = ad_pt
	    			

	return points



if __name__ == '__main__':
    polygons = load_polygons( "./paths.txt" )
    x1, y1, x2, y2 = polygons_to_segments( polygons )
    
    start_pt = np.atleast_2d( [182.0/500,12.0/500] )
    goal_pt = np.atleast_2d( [409.0/500,353.0/500] )

    x1, y1, x2, y2 = polygons_to_segments( polygons )

    #path = run_rrt_poly( start_pt, goal_pt, polygons, heat = intensity, plot=False)
    path = run_rrt( start_pt, goal_pt, x1, y1, x2, y2)

     # Create figure and axes
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)

    #c = np.linspace(0, 10, np_paths.shape[0])
    img = plt.imread("./cnts_inv.png")
    ax.imshow(img)

    scale = 1000

    for i in range( 0, len(path)-1 ):
        ax.plot( [ path[i][0] * scale, path[i+1][0] * scale ], [ path[i][1] * scale, path[i+1][1] * scale], 'b' )
     
    iters = 10000 #10000
    std = 1.0/500          
    new_path = optimize_path(x1, y1, x2, y2, path, iters, std)

    for i in range( 0, len(new_path)-1 ):
        ax.plot( [ new_path[i][0] * scale, new_path[i+1][0] * scale ], [ new_path[i][1] * scale, new_path[i+1][1] * scale], 'red' )
     

    ax.scatter( start_pt[0,0] * scale, start_pt[0,1]  * scale)
    ax.scatter( goal_pt[0,0] * scale, goal_pt[0,1] * scale)
    plt.ylim((0,scale))

    for i in xrange(x1.shape[0]):
    	ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'black' )
         
    #print(x1[1,0])
    #print(x1.shape) #(323, 1)


    plt.show() #boom
    