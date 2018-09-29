import numpy as np
import time
import matplotlib.pyplot as plt

STEP_SIZE = .05

# python -m cProfile -s tottime my_rrt.py

# adapted from the original matlab code at
# http://www.mathworks.com/matlabcentral/fileexchange/27205-fast-line-segment-intersection
def line_intersect( X1,Y1,X2,Y2, X3,Y3,X4,Y4 ):
    X4_X3 = X4.T - X3.T
    Y1_Y3 = Y1   - Y3.T
    Y4_Y3 = Y4.T - Y3.T
    X1_X3 = X1   - X3.T
    X2_X1 = X2   - X1
    Y2_Y1 = Y2   - Y1

    numerator_a = X4_X3 * Y1_Y3 - Y4_Y3 * X1_X3
    numerator_b = X2_X1 * Y1_Y3 - Y2_Y1 * X1_X3
    denominator = Y4_Y3 * X2_X1 - X4_X3 * Y2_Y1

    u_a = numerator_a / (denominator+1e-20)
    u_b = numerator_b / (denominator+1e-20)

    INT_X = X1 + X2_X1 * u_a
    INT_Y = Y1 + Y2_Y1 * u_a
    did_intersect = (u_a >= 0) & (u_a <= 1) & (u_b >= 0) & (u_b <= 1)

    return INT_X, INT_Y, did_intersect


#
# loads a set of polygons
#
# note that this appends the first point to also be the last point.
# this function assumes that the list is given in "open" form; by
# appending the first point as the last point, it ensures that the
# resulting polygon is exactly closed.
#
# note that this prepends a single point that is the mean of all the other points.
# this is for drawing the polygons using a GL_TRIANGLE_FAN.  It's a total hack.
#
def load_polygons( fn="./paths.txt" ):
    bdata = []
    for x in open( fn ):
        tmp = np.fromstring( x, dtype=float, sep=' ' )
        tmp = np.reshape( tmp/1000.0, (-1,2) )
        tmp = np.vstack(( np.mean(tmp, axis=0, keepdims=True), tmp, tmp[0,:] ))
        tmp[:,1] = 1.0 - tmp[:,1]  # flip on the y axis
        bdata.append( tmp )
    return bdata

# polygon_list is a list of np arrays
# each nparray is a kx2 matrix, representing x,y points
# first entry is the mean of all the points, which is SKIPPED
# last entry in the matrix is the same as the first
# returns x1,y1, x2,y2
def polygons_to_segments( polygon_list ):
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    for x in polygon_list:
        X1.append( x[1:-1,0:1] )
        Y1.append( x[1:-1,1:2] )
        X2.append( x[2:,0:1] )
        Y2.append( x[2:,1:2] )
    X1 = np.vstack( X1 )
    Y1 = np.vstack( Y1 )
    X2 = np.vstack( X2 )
    Y2 = np.vstack( Y2 )

    return X1, Y1, X2, Y2

def distance_to_other_points( pt, pts ):
    diffs = (pts - pt)**2.0
    return np.sum( diffs, axis=1, keepdims=True )

def run_rrt_poly( start_pt, goal_pt, polygons, bias=0.75, plot=False, step_limit=20000, scale=1):
    '''
    start_pt: 1 x 2 np array
    goal_pt: 1 x 2 np array
    polygons: list (polygons) of n x 2 (x, y) np arrays
    bias: 

    returns a list of length 2 np arrays describing the path from `start_pt` to `goal_pt`
    '''
    x1, y1, x2, y2 = polygons_to_segments( polygons )
    return run_rrt( start_pt, goal_pt, x1, y1, x2, y2, bias, plot, scale=scale )

#--------------------------------------------------------------------------------------------------

# def run_rrt_blown_up( start_pt, goal_pt, endpoint_a_x, endpoint_a_y, endpoint_b_x, endpoint_b_y,  goal_buffer=.005, bias=0.75, plot=False, step_limit=20000, scale=1 ):
#     # blow everything up by 100 and see if it makes a difference
#     _scale = 100.0
#     STEP_SIZE=8
#     #STEP_SIZE= STEP_SIZE*_scale
#     start_pt = start_pt*_scale
#     goal_pt = goal_pt* _scale
#     endpoint_a_x = endpoint_a_x* _scale
#     endpoint_a_y = endpoint_a_y*_scale
#     endpoint_b_x = endpoint_b_x*_scale
#     endpoint_b_y = endpoint_b_y*_scale
#     goal_buffer = goal_buffer*_scale


#     nodes = start_pt
#     parents = np.atleast_2d( [0] )

#     for i in range( 0, step_limit ):
#         random_point = np.random.rand(1,2) * scale
# #        random_point = Q.rand( sz=(1,2), name="rrt_q_%d"%i ) * scale
        
#         # find nearest node
#         distances = distance_to_other_points( random_point, nodes )
#         if np.isnan(np.sum(distances)):
#             return None
#         nearest_ind = np.argmin( distances )

#         nearest_point = nodes[ nearest_ind:nearest_ind+1, : ]


#         # take a step towards the goal
#         if np.random.rand() > bias:
#             ndiff = goal_pt - nearest_point
#         else:
#             ndiff = random_point - nearest_point

#         # if np.isnan(np.sum( ndiff)):
#         #     print np.sum(random_point), np.sum(nearest_point), nearest_point, nearest_ind, nodes.shape, distances

#         ndiff = (scale * STEP_SIZE) * ndiff / np.sqrt( np.sum( ndiff*ndiff ) )

#         new_pt = nearest_point + ndiff

#         # we'd like to expand from nearest_point to new_pt.  Does it cross a wall?
#         int_x, int_y, intersection_indicators = line_intersect( 
#             nearest_point[0,0], 
#             nearest_point[0,1], 
#             new_pt[0,0], 
#             new_pt[0,1], 
#             endpoint_a_x, endpoint_a_y, endpoint_b_x, endpoint_b_y)


#         if intersection_indicators.any():

#         	# d = pt_line_dist(new_pt, endpoint_a_x, endpoint_a_y, endpoint_b_x, endpoint_b_y)

#         	# b = np.where(d < .01)
#         	# #print b[0]
#         	# if len(b[0]) == 0:

# 	        # 	nodes = np.vstack(( nodes, new_pt ))
# 		       #  # if np.isnan(np.sum(nodes)):
# 		       #  #     print "new_pt", new_pt
# 		       #  parents = np.vstack(( parents, nearest_ind ))

           
#             #calculate nearest intersection and trim new_pt
#             intersections = np.atleast_2d( [ int_x[intersection_indicators], int_y[intersection_indicators] ] ).T
#             distances = distance_to_other_points( nearest_point, intersections )
#             closest_intersection_index = np.argmin( distances )
#             new_pt = intersections[ closest_intersection_index:closest_intersection_index+1, : ]

#             safety = new_pt - nearest_point
#             safety = scale * 0.01 * safety / np.sqrt( np.sum( safety*safety ) )
#             new_pt = new_pt - safety

#             int_x, int_y, intersection_indicators = line_intersect( 
#             nearest_point[0,0], 
#             nearest_point[0,1], 
#             new_pt[0,0], 
#             new_pt[0,1], 
#             endpoint_a_x, endpoint_a_y, endpoint_b_x, endpoint_b_y)


#             if intersection_indicators.any():
#             	continue


        
#         if distance_to_other_points( new_pt, goal_pt ) < (goal_buffer * scale):
#         	start_pt = start_pt/_scale
#         	goal_pt = goal_pt/ _scale
#         	nodes = np.vstack(( nodes, new_pt ))
#         	parents = np.vstack(( parents, nearest_ind ))
#         	path = [ new_pt[0,:]/_scale ]
#         	while nearest_ind != 0:
#         		path.append( nodes[nearest_ind,:]/_scale )
#         		nearest_ind = parents[ nearest_ind, 0 ]
#         		path.append( nodes[0,:]/_scale )
#     		path.reverse()
#     		path.append(goal_pt[0,:])

#     		endpoint_a_x = endpoint_a_x/ _scale
#     		endpoint_a_y = endpoint_a_y/_scale
#     		endpoint_b_x = endpoint_b_x/_scale
#     		endpoint_b_y = endpoint_b_y/_scale
#     		goal_buffer = goal_buffer/_scale
#     		STEP_SIZE= STEP_SIZE/_scale
#     		return path
           
#         nodes = np.vstack(( nodes, new_pt ))
#         parents = np.vstack(( parents, nearest_ind ))

#     #print('No path found!')
#     return []

#---------------------------------------------------------------------------------------------------

def run_rrt( start_pt, goal_pt, endpoint_a_x, endpoint_a_y, endpoint_b_x, endpoint_b_y,  goal_buffer=.005, bias=0.75, plot=False, step_limit=20000, scale=1 ):
    nodes = start_pt
    parents = np.atleast_2d( [0] )

    for i in range( 0, step_limit ):
        random_point = np.random.rand(1,2) * scale
#        random_point = Q.rand( sz=(1,2), name="rrt_q_%d"%i ) * scale
        
        # find nearest node
        distances = distance_to_other_points( random_point, nodes )
        if np.isnan(np.sum(distances)):
            return None
        nearest_ind = np.argmin( distances )

        nearest_point = nodes[ nearest_ind:nearest_ind+1, : ]


        # take a step towards the goal
        if np.random.rand() > bias:
            ndiff = goal_pt - nearest_point
        else:
            ndiff = random_point - nearest_point

        # if np.isnan(np.sum( ndiff)):
        #     print np.sum(random_point), np.sum(nearest_point), nearest_point, nearest_ind, nodes.shape, distances

        ndiff = (scale * STEP_SIZE) * ndiff / np.sqrt( np.sum( ndiff*ndiff ) )


        new_pt = nearest_point + ndiff

        # if distance_to_other_points( new_pt, goal_pt ) < (goal_buffer * scale):
        #     #print('i', i)
        #     path = [ new_pt[0,:] ]
        #     while nearest_ind != 0:
        #         path.append( nodes[nearest_ind,:] )
        #         nearest_ind = parents[ nearest_ind, 0 ]
        #     path.append( nodes[0,:] )

        #     if plot == True:
        #         plt.figure()
        #         for i in range(0, endpoint_a_x.shape[0]):
        #             plt.plot( [ endpoint_a_x[i], endpoint_b_x[i] ], [ endpoint_a_y[i], endpoint_b_y[i] ], 'k' )
        #         for i in range( 0, len(path)-1 ):
        #             plt.plot( [ path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1] ], 'b' )
        #         plt.scatter( start_pt[0,0], start_pt[0,1] )
        #         plt.scatter( goal_pt[0,0], goal_pt[0,1] )
        #         plt.show()

        #     path.reverse()
        #     path.append(goal_pt[0,:])
        #     return path

        # we'd like to expand from nearest_point to new_pt.  Does it cross a wall?
        int_x, int_y, intersection_indicators = line_intersect( 
            nearest_point[0,0], 
            nearest_point[0,1], 
            new_pt[0,0], 
            new_pt[0,1], 
            endpoint_a_x, endpoint_a_y, endpoint_b_x, endpoint_b_y)


        if intersection_indicators.any():

        	# d = pt_line_dist(new_pt, endpoint_a_x, endpoint_a_y, endpoint_b_x, endpoint_b_y)

        	# b = np.where(d < .01)
        	# #print b[0]
        	# if len(b[0]) == 0:

	        # 	nodes = np.vstack(( nodes, new_pt ))
		       #  # if np.isnan(np.sum(nodes)):
		       #  #     print "new_pt", new_pt
		       #  parents = np.vstack(( parents, nearest_ind ))

           
            #calculate nearest intersection and trim new_pt
            intersections = np.atleast_2d( [ int_x[intersection_indicators], int_y[intersection_indicators] ] ).T
            distances = distance_to_other_points( nearest_point, intersections )
            closest_intersection_index = np.argmin( distances )
            new_pt = intersections[ closest_intersection_index:closest_intersection_index+1, : ]

            safety = new_pt - nearest_point
            safety = scale * 0.01 * safety / np.sqrt( np.sum( safety*safety ) )
            new_pt = new_pt - safety

            int_x, int_y, intersection_indicators = line_intersect( 
            nearest_point[0,0], 
            nearest_point[0,1], 
            new_pt[0,0], 
            new_pt[0,1], 
            endpoint_a_x, endpoint_a_y, endpoint_b_x, endpoint_b_y)


            if intersection_indicators.any():
            	continue


        
        if distance_to_other_points( new_pt, goal_pt ) < (goal_buffer * scale):
            nodes = np.vstack(( nodes, new_pt ))
            parents = np.vstack(( parents, nearest_ind ))
            #print('i', i)
            path = [ new_pt[0,:] ]
            while nearest_ind != 0:
                path.append( nodes[nearest_ind,:] )
                nearest_ind = parents[ nearest_ind, 0 ]
            path.append( nodes[0,:] )

            if plot == True:
                plt.figure()
                for i in range(0, endpoint_a_x.shape[0]):
                    plt.plot( [ endpoint_a_x[i], endpoint_b_x[i] ], [ endpoint_a_y[i], endpoint_b_y[i] ], 'k' )
                for i in range( 0, len(path)-1 ):
                    plt.plot( [ path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1] ], 'b' )
                plt.scatter( start_pt[0,0], start_pt[0,1] )
                plt.scatter( goal_pt[0,0], goal_pt[0,1] )
                plt.show()

            path.reverse()
            path.append(goal_pt[0,:])
            return path
           
        nodes = np.vstack(( nodes, new_pt ))
        parents = np.vstack(( parents, nearest_ind ))

    #print('No path found!')
    return []

# ==============================================================

from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm

# from: https://gist.github.com/nim65s/5e9902cd67f094ce65b0
def distance_numpy(A, B, P):
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == P) or all(B == P):
        return 0
    if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
        return norm(P - A)
    if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
        return norm(P - B)
    return norm(cross(A-B, A-P))/norm(B-A)


def pt_line_dist(P, X1, Y1, X2, Y2):

	nom = np.absolute( (Y2-Y1)*P[0][0] - (X2-X1)*P[0][1] + (X2*Y1) - (Y2*X1) )
	den = np.sqrt((Y2-Y1)**2 + (X2-X1)**2)
	d = nom/den
	#XXX maybe works not sure
	# A = np.zeros((323,2))
	# A[:,0:1] = endpoint_a_x[0:]
	# A[:,1:] = endpoint_a_y[0:]
	# #print A.shape

	# B = np.zeros((323,2))
	# B[:,0:1] = endpoint_b_x[0:]
	# B[:,1:] = endpoint_b_y[0:]
	# #print B.shape
	# #print P.shape

	# d=np.cross(B-A,P-A)/norm(B-A)
	return d



if __name__ == '__main__':
    polygons = load_polygons( "./paths.txt" )
    
    start_pt = np.atleast_2d( [0.1,0.1] )
    goal_pt = np.atleast_2d( [0.9,0.9] )

    path = run_rrt_poly( start_pt, goal_pt, polygons, plot=True)
    print ("len of path:", len(path))
    print ("path:", path)


    line_intersect( X1,Y1,X2,Y2, X3,Y3,X4,Y4 )