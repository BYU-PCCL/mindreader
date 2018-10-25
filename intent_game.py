import math
import pygame
from pygame.locals import *
import random as rand
import numpy as np
import sys
import isovist as iso
from numpy import atleast_2d
import pickle
from my_rrt import *
from PIL import Image
import PIL.ImageOps
from tqdm import tqdm
import planner
#from new_intruder import *
'''
Init Screen

Creates a pygame display

Returns a screen and a clock

'''
# image = Image.open('./cnts.png')

# inverted_image = PIL.ImageOps.invert(image)

# inverted_image.save('./cnts_inv.png')

def InitScreen(xdim, ydim):
	pygame.init()
	pygame.font.init()

	size = (xdim, ydim)
	screen = pygame.display.set_mode(size)

	pygame.display.set_caption("Isovist")
	clock = pygame.time.Clock()

	return screen, clock

'''
	Updates the pygame screen
	and allows for exiting of the pygame screen
'''

def Update():
	pygame.display.update()
	for e in pygame.event.get():
		if e.type == QUIT:
			sys.exit("Exiting")
		# if e.type == MOUSEBUTTONDOWN:
		#     return pygame.mouse.get_pos()
		#if e.type == KEYDOWN:
			#keys_pressed = pygame.key.get_pressed()
			#return keys_pressed

		    

def load_polygons_here( fn="./paths.txt" ):
    bdata = []
    for x in open( fn ):
        tmp = np.fromstring( x, dtype=float, sep=' ' )
        tmp = np.reshape( tmp/1000, (-1,2) )
        tmp = np.vstack(( np.mean(tmp, axis=0, keepdims=True), tmp, tmp[0,:] ))
        #tmp[:,1] = 1.0 - tmp[:,1]  # flip on the y axis
        bdata.append( tmp )
    return bdata

def load_polygons_rrt( fn="./paths.txt" ):
    bdata = []
    for x in open( fn ):
        tmp = np.fromstring( x, dtype=float, sep=' ' )
        tmp = np.reshape( tmp/1000.0, (-1,2) )
        tmp = np.vstack(( np.mean(tmp, axis=0, keepdims=True), tmp, tmp[0,:] ))
        tmp[:,1] = 1.0 - tmp[:,1]  # flip on the y axis
        bdata.append( tmp )
    return bdata

def load_polygons( fn="./paths.txt" ):
	polygonSegments = []
	for line in open( fn ):
		line = line.strip('\n')
		toList = line.split(' ')
		toList = [(float(x)/1000) for x in toList]
		
		it = iter(toList)
		toList = [toList[i:i+2] for i in range(0, len(toList), 2)]

		for pair in toList:
			#pair[1] = 1.0 - pair[1]
			pair[0] = int (pair[0] *500)
			pair[1] = int (pair[1] *500)

		#toList = [toList[i:i+2] for i in range(0, len(toList), 2)]
		#toList[-1].insert(0, toList[0][0])
		temp = []
		for i in xrange(1,len(toList)):
			pair = (toList[i-1], toList[i])
			temp.append(pair)
		temp.append((toList[0],toList[-1]))
		
		polygonSegments.append(temp)

	dim = 500

	'''border'''
	polygonSegments.append([ 
		[ (-5,-5),(505,-5) ], 
		[ (505,-5),(505,505) ],
		[ (505,505), (-5,505)],
		[ (-5,505), (-5,-5) ]
		])
        #print "toList:", toList
	# for p in polygonSegments:
	# 	print "\n", p
	return polygonSegments

def GetReadablePath(path):
	readable_path = []
	for i in xrange(1, len(path)):
		s_point = path[i-1]
		s_point = (int(s_point[0]*500), int(s_point[1]*500))
		
		e_point = path[i]
		e_point = (int(e_point[0]*500), int(e_point[1]*500))
		pygame.draw.line(screen, (225, 225, 0), s_point, e_point, 2)
		readable_path.append(s_point)
	return readable_path

def scale(point):
	return (int(point[0] * 500), int(point[1] * 500))

def rrt_loc(point):
	loc = np.atleast_2d( [( point[0]/500.0) ,( point[1]/500.0 )] )
	return  loc

def DrawRRT(path, screen, goal, printPoints=False, highlight = None):
	color_in_use = (0,0,255)
	if highlight != None:
		color_in_use = highlight
	e_point = None

	for i in xrange(1, len(path)):
		s = path[i-1]
		e = path[i]
		s_point = (int(s[0]*500), int(s[1]*500))
		e_point = (int(e[0]*500), int(e[1]*500))
		pygame.draw.line(screen, color_in_use, s_point, e_point, 2)

		

	if e_point != None:
		pygame.draw.line(screen, color_in_use, e_point, goal, 2)
	
	# skip = 8

	# for i in xrange(skip-1, len(path), skip):
	# 	s = path[i-skip+1]
	# 	e = path[i]
	# 	s_point = (int(s[0]*500), int(s[1]*500))
	# 	e_point = (int(e[0]*500), int(e[1]*500))
	# 	pygame.draw.line(screen, (255,0,0), s_point, e_point, 1)
	# if e_point != None:
	# 	pygame.draw.line(screen, (255,0,0), e_point, goal, 1)


def direction (now, before):
	return (now[0]-before[0], now[1] - before[1])

def load_data():
    try:
        with open("rrt_paths-new.dat") as f:
            x = pickle.load(f)
    except:
        x = []
    return x

def save_data(data):
    with open("rrt_paths-new.dat", "wb") as f:
        pickle.dump(data, f)

def isovist_area(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in segments(p)))

def segments(p):
    return zip(p, p[1:] + [p[0]])

#def sampleLocsFromPath(path, X=8):

def smart_intruder_rrt(start, goal, X1, Y1, X2, Y2, isovist, N=30, X=8, rrts_=None, screen=None):
	#GENERATE 30 RRTs:
	rrts = []#rrts_
	areas = []
	for i in tqdm(xrange(3)):
		rrts.append(run_rrt( rrt_loc(start), rrt_loc(goal), X1, Y1, X2, Y2))
		chosen_steps = rand.sample(range(1, len(rrts[i])), X)
		total_isovist_area = 0
		for j in chosen_steps:
			loc = rrts[i][j]
			prev = rrts[i][j-1]
			intersections = isovist.GetIsovistIntersections(loc, direction(loc, prev), full_iso=True)
			total_isovist_area += isovist_area(intersections) 



			# diamond = [(0,.12),(.12,0),(0,-.12),(-.12,0)]
			# s = loc
			# pygame.draw.line(screen, (0,0,255), paint_loc(s + diamond[0]), paint_loc(s + diamond[1]), 1)
			# pygame.draw.line(screen, (0,0,255), paint_loc(s + diamond[1]), paint_loc(s + diamond[2]), 1)
			# pygame.draw.line(screen, (0,0,255), paint_loc(s + diamond[2]), paint_loc(s + diamond[3]), 1)
			# pygame.draw.line(screen, (0,0,255), paint_loc(s + diamond[3]), paint_loc(s + diamond[0]), 1)

		areas.append(total_isovist_area)
	minindex = np.argmin(areas)
	return rrts[minindex]




'''
	main function

'''

def main():

	

	'''
	xdim and ydim of the pygame screen 
	'''
	xdim = 500
	ydim = 500
	backgroundFileName = "./intent-map.png"
	background = pygame.image.load(backgroundFileName)
	background = pygame.transform.scale(background, (xdim, ydim))
	backgroundRect = background.get_rect()

	array = np.zeros([xdim, ydim])
	screen, clock = InitScreen(xdim, ydim)
	polygonSegments = load_polygons()

	# Clear canvas
	screen.fill((255,255,255))

	s = pygame.Surface((xdim,ydim))  	# the size of your rect
	s.set_alpha(0)                		# alpha level
	s.fill((255,255,255))           	# this fills the entire surface
	screen.blit(s, (0,0))

	screen.blit(background, backgroundRect)


	#### RRT STUFF
	start_paint = (int(0.1 *500),int(0.1 *500))
	end_paint = (int(0.9*500), int(0.9 *500))

	start = np.atleast_2d( [(0.1 ) ,(0.1 )] )
	end = np.atleast_2d( [(0.9 ),(0.7 )] )

	start = np.atleast_2d( [0.052, 0.21])
	end = np.atleast_2d( [0.842, 0.798])

	#X1, Y1, X2, Y2 
	poly_map = polygons_to_segments(load_polygons_here(fn="./my_intent_map.txt"))
	#print poly_map
	scale = 500
	x1,y1,x2,y2 = poly_map
	#for i in xrange(x1.shape[0]):
		# ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'grey', linewidth=1  )
		#pygame.draw.line(screen, (0, 0, 0), [ x1[i,0] * scale, y1[i,0] * scale ], [ x2[i,0] * scale, y2[i,0] * scale],2)


			
	Update()

	#numPaths = 30
	#numPaths = 1
	

	# rrt_paths = load_data()	
	# # isovist stuff
 # 	isovist = iso.Isovist( polygonSegments )


	# IntruderStart = paint_loc((0.1, 0.1))
	# IntruderGoal = paint_loc((.9,.9))
	# #intruder_path = smart_intruder_rrt(IntruderStart, IntruderGoal, X1, Y1, X2, Y2, isovist, X=8, rrts_=rrt_paths,screen=screen)

	# diamond = [(0,.08),(.08,0),(-.08,0),(-.08,0)]

	# paths = []				
	# for i in xrange(24,25):
	# 	# inbetween = np.atleast_2d([inbetweenPoints[i%10]])
	# 	path = run_rrt( start, end, X1, Y1, X2, Y2)
	# 	# path2 = run_rrt( inbetween, end, X1, Y1, X2, Y2)
	# 	# path.extend(path2)
	# 	paths.append(path)
	# 	path = rrt_paths[i]
	# 	DrawRRT(path, screen, end_paint, printPoints = True, highlight=(12,165,27))
	# 	print "DRAW", i
	# 	pygame.time.delay(2)
	# 	Update()
		
	
	#DrawRRT(intruder_path, screen, end_paint, printPoints = True, highlight=(12,165,27))
		#save_data(paths)
	# pygame.draw.circle(screen, (0,255,0), start_paint, 5)
	# pygame.draw.circle(screen, (255,0,0), end_paint, 5)

	# index = getPathIndex()
	# print index
	red = (200,0,0)
	green = (0,200,0)

	bright_red = (255,0,0)
	bright_green = (0,255,0)
	forward = True

	#path = run_rrt( start, end, x1, y1, x2, y2)

	path = planner.run_rrt_opt( start, end, x1, y1, x2, y2)
	myfont = pygame.font.SysFont("arial", 20)
	my_map = []
	curr_poly = []

	other_player = [0.5,0.6]
	i = 0
	while True:
		keys_pressed = Update()
		# redraw the background
		if i < len(path):
			screen.fill((255,255,255))
			screen.blit(background, backgroundRect)
			#for x in xrange(x1.shape[0]):
				#pygame.draw.line(screen, (0, 0, 0), [ x1[x,0] * scale, y1[x,0] * scale ], [ x2[x,0] * scale, y2[x,0] * scale],2)
		
			player = (int(path[i][0] *500),int(path[i][1] *500))
			pygame.draw.circle(screen, (0,170,100), player, 9)
		else:
			#path = None
			#while path == None:
			if forward:
				#path = run_rrt( end, start, x1, y1, x2, y2)
				path = planner.run_rrt_opt( end, start, x1, y1, x2, y2)
			else:
				#path = run_rrt( start, end, x1, y1, x2, y2)
				path = planner.run_rrt_opt( start, end, x1, y1, x2, y2)
			forward = not forward
			i = -1
		keys_pressed = pygame.key.get_pressed()
		if keys_pressed != None:
			if keys_pressed[K_LEFT]:
				other_player[0] -= .05
			if keys_pressed[K_RIGHT]:
				other_player[0] += .05
			if keys_pressed[K_UP]:
				other_player[1] -= .05
			if keys_pressed[K_DOWN]:
				other_player[1] += .05

		o_player = (int(other_player[0] *500),int(other_player[1] *500))
		pygame.draw.circle(screen, (170,0,100), o_player, 9)


		pygame.time.delay(200)
		i += 1

		                    

if __name__ == '__main__':
    main()

