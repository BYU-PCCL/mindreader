
import math
import random as rand
import numpy as np
import sys
import isovist as iso
from numpy import atleast_2d
import pickle
from my_rrt import *
from tqdm import tqdm
import matplotlib.path as mplPath



def InitScreen(xdim, ydim):
	pygame.init()
	pygame.font.init()

	size = (xdim, ydim)
	screen = pygame.display.set_mode(size)

	pygame.display.set_caption("Isovist")
	clock = pygame.time.Clock()

	return screen, clock


def Update():
	pygame.display.update()
	for e in pygame.event.get():
		if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
			sys.exit("Exiting")
		if e.type == MOUSEBUTTONDOWN:
		    return pygame.mouse.get_pos()


def load_polygons( fn="./paths.txt" ):
    bdata = []
    for x in open( fn ):
        tmp = np.fromstring( x, dtype=float, sep=' ' )
        tmp = np.reshape( tmp/1000, (-1,2) )
        tmp = np.vstack(( np.mean(tmp, axis=0, keepdims=True), tmp, tmp[0,:] ))
        #tmp[:,1] = 1.0 - tmp[:,1]  # flip on the y axis
        bdata.append( tmp )
    return bdata

def load_isovist_map( fn="./paths.txt" ):
	
	polygonSegments = []
	for line in open( fn ):
		line = line.strip('\n')
		toList = line.split(' ')
		toList = [(float(x)/1000) for x in toList]
		
		it = iter(toList)
		toList = [toList[i:i+2] for i in range(0, len(toList), 2)]


		for pair in toList:
			pair[0] = int (pair[0] *500)
			pair[1] = int ((1-pair[1]) *500)

		temp = []
		for i in xrange(1,len(toList)):
			pair = (toList[i-1], toList[i])
			temp.append(pair)
		temp.append((toList[0],toList[-1]))
		
		polygonSegments.append(temp)

		load_segs( fn="./paths.txt" )
	dim = 500
	return polygonSegments

def load_segs( fn="./paths.txt" ):
	poly_segs = []
	poly_easy = []
	for line in open( fn ):
		line = line.strip('\n')
		toList = line.split(' ')
		toList = [(float(x)/1000) for x in toList]
		
		it = iter(toList)
		toList = [toList[i:i+2] for i in range(0, len(toList), 2)]
		
		for pair in toList:
			pair[0] = pair[0]
			pair[1] = (1-pair[1])

		toList.append(toList[0])
		poly_segs.append(toList)
		poly_easy.append(mplPath.Path(np.asarray(toList)))

	return poly_segs, poly_easy

def line_intersection(line1, line2):
	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

	def det(a, b):
		return a[0] * b[1] - a[1] * b[0]

	div = det(xdiff, ydiff)
	if div == 0:
		return None, None
		#raise Exception('lines do not intersect')

	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	return x, y

# assuming that the point is in an obstacle
def get_clear_goal(st, poi, poly_segs):
	my_line = [st, poi]
	inter_point = None
	# iter through each polygon
	for poly in poly_segs:
		# iter through each line seg
		for i in xrange(0,len(poly)-2):
			line = [poly[i],poly[i+1]]
			x, y = line_intersection(my_line, line)
			if not x is None:
				inter_point = [x,y]
				break
		if not inter_point is None:
			break

	my_dir = direction(st, poi)
	dx = my_dir[0] 
	dy = my_dir[1]
	dMag = math.sqrt(dx**2 + dy**2)
	ndx = dx/dMag
	ndy = dy/dMag
	ndx *= .025
	ndy *= .025
	return [poi[0]+ndx, poi[1]+ndy]



def point_in_obstacle(point, poly_segs):
	for poly in poly_segs:
		if poly.contains_point(point):
			return True, poly
	return False, None


def load_data(filename="data"):
    try:
        with open(filename+ ".dat") as f:
            x = pickle.load(f)
    except:
        x = []
    return x

def save_data(data, filename="data"):
    with open(filename + ".dat", "wb") as f:
        pickle.dump(data, f)


def scale_up(point):
	return (int(point[0] * 500), int(point[1] * 500))

def scale_down(point):
	loc = np.atleast_2d( [( point[0]/500.0) ,( point[1]/500.0 )] )
	return  loc

def dist(one, two):
	xs = one[0] - two[0]
	ys = one[1] - two[1]
	return math.sqrt(xs**2 + ys**2)

def direction (now, before):
	return (now[0]-before[0], now[1] - before[1])

def combine_paths(paths):
	long_path = []
	for p in paths:
		long_path.extend(p)
	return long_path


#
# ==========================================================================
#
from multiprocessing import Pool
#from path_kde import *
a2d = atleast_2d
def memoize(f):
        """ Memoization decorator for functions taking one or more arguments. """
        class memodict(dict):
                def __init__(self, f):
                        self.f = f
                def __call__(self, *args):

                        key = str([str(x) for x in args ])
                        if self.has_key( key ):
                                return self[ key ]
                        else:
                                rval = self.f( *args )
                                self[key] = rval
                                return rval
                        # return self[args] # XXX more elegant, but numpy arrays are unhashable
#                def __missing__(self, key):
#                        ret = self[key] = self.f( *args )
#                        return ret

        return memodict(f)

#
# ==========================================================================
#

def isovist_area(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in segments(p)))

def segments(p):
    return zip(p, p[1:] + [p[0]])

def single_smart_intruder_rrt_sarg( x ):
        # need to make sure that each process is in a different PRNG
        # state.  initialize seed using getpid(), but only do it once
        # per process (otherwise it will reset the next time the
        # worker is used!)
        if not hasattr( single_smart_intruder_rrt_sarg, '__my_init' ):
                single_smart_intruder_rrt_sarg.__my_init = True # use function attributes as static variables
                import os
                np.random.seed( os.getpid() )
        return single_smart_intruder_rrt( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] )

def single_smart_intruder_rrt(start, goal, X1, Y1, X2, Y2, isovist, X=8):
        rrt = run_rrt( a2d(scale_down(start)), a2d(scale_down(goal)), X1, Y1, X2, Y2)

        try:
            chosen_steps = rand.sample( range(1, len(rrt)), np.minimum(X,len(rrt)-2) )
        except:
            return( rrt, 0 )

        total_isovist_area = 0
        for j in chosen_steps:
            loc = rrt[j]
            prev = rrt[j-1]
            intersections = isovist.GetIsovistIntersections(loc, direction(loc, prev), full_iso=True)
            total_isovist_area += isovist_area(intersections) 
        return( rrt, total_isovist_area )

# XXX hacked out the kwargs because they're not memoize compatible...
@memoize
def smart_intruder_rrt_par(start, goal, X1, Y1, X2, Y2, isovist, N, X):
        areas = []
        rrts = []

#        p = Pool( 12 )
        p = Pool( 6 )
        # we do all of this because Pool.map pickles its arguments, and you can't pickle a lambda...
        params = ((start, goal, X1, Y1, X2, Y2, isovist, X),) * N
        results = p.map( single_smart_intruder_rrt_sarg, params )

	for tmp_retval in results:
                rrts.append( tmp_retval[0] )
                areas.append( tmp_retval[1] )
	minindex = np.argmin(areas)
	return rrts[minindex]

# XXX deprecated in favor of the parallel version above
def smart_intruder_rrt(start, goal, X1, Y1, X2, Y2, isovist, N=30, X=8):
	#GENERATE 30 RRTs:
        areas = []
        rrts = []
	for i in tqdm(xrange(N)):
                tmp_retval = single_smart_intruder_rrt(start, goal, X1, Y1, X2, Y2, isovist, X )
                rrts.append( tmp_retval[0] )
                areas.append( tmp_retval[1] )
	minindex = np.argmin(areas)
	return rrts[minindex]


def travel(curr_i, path, amt=10, bound=30):
	traveled = 0
	while traveled <= amt:
		pre = path[curr_i]

		#If already reached the end of the path
		if curr_i == len(path)-1:
			return curr_i

		curr_i += 1
		curr = path[curr_i]

		distance = dist(scale_up(pre), scale_up(curr))
		#print "distance:", dist(scale(pre), scale(curr))

		if traveled + distance > bound:
			curr_i -=1
			break

		traveled += distance

		
	#print "\ntraveled:", traveled
	return curr_i



def detect(perspective_loc, perspective_pre_loc, other_loc, isovist):
	if dist(other_loc, perspective_loc) <= 200:
		fv = direction(perspective_loc, perspective_pre_loc) 
		if not(fv[0] == 0 or fv[1] == 0):
			intersections = isovist.GetIsovistIntersections(perspective_loc, fv)
			seen = isovist.FindIntruderAtPoint(other_loc, intersections)
			return seen, intersections
	return False, None


def noise_level(intruder_loc, UAV_loc):
	distance = dist(intruder_loc, UAV_loc)
	if distance <= 100:
		if distance <= 20:
			return 5
		if distance <= 40:
			return 4
		if distance <= 60:
			return 3
		if distance <= 80:
			return 2
		return 1

	return 0


