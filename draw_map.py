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
		if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
			sys.exit("Exiting")
		if e.type == MOUSEBUTTONDOWN:
		    return pygame.mouse.get_pos()

def load_polygons_here( fn="./paths.txt" ):
    bdata = []
    for x in open( fn ):
        tmp = np.fromstring( x, dtype=float, sep=' ' )
        tmp = np.reshape( tmp/1000, (-1,2) )
        tmp = np.vstack(( np.mean(tmp, axis=0, keepdims=True), tmp, tmp[0,:] ))
        #tmp[:,1] = 1.0 - tmp[:,1]  # flip on the y axis
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
	backgroundFileName = "./cnts_inv.png"
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

	#screen.blit(background, backgroundRect)


	#### RRT STUFF
	start_paint = (int(0.1 *500),int(0.1 *500))
	end_paint = (int(0.9*500), int(0.9 *500))

	start = np.atleast_2d( [(0.1 ) ,(0.1 )] )
	end = np.atleast_2d( [(0.9 ),(0.9 )] )
	X1, Y1, X2, Y2 = polygons_to_segments(load_polygons_here())

	
	# for polygon in polygonSegments:
	# 	p =[]
	# 	p.append(tuple(polygon[0][0]))
	# 	for segment in polygon:
	# 		pygame.draw.line(screen, (0, 0, 0), segment[0], segment[1],2)
	# 		p.append(tuple(segment[0]))
		
		# if p != [(-5, -5), (-5, -5), (505, -5), (505, 505), (-5, 505)]:
		# 	pygame.draw.polygon(screen, (0,0,0), p)


	# cand_start_locs = (
	# (0.044, 0.034) ,
	# (0.196, 0.032) ,
	# (0.752, 0.032) ,
	# (0.916, 0.048) ,
	# (0.104, 0.122) ,
	# (0.454, 0.116) ,
	# (0.776, 0.092) ,
	# (0.882, 0.19) ,
	# (0.614, 0.246) ,
	# (0.294, 0.376) ,
	# (0.09, 0.24) ,
	# (0.072, 0.454) ,
	# (0.24, 0.476) ,
	# (0.682, 0.428) ,
	# (0.844, 0.534) ,
	# (0.544, 0.614) ,
	# (0.252, 0.642) ,
	# (0.05, 0.702) ,
	# (0.032, 0.872) ,
	# (0.376, 0.77) ,
	# (0.542, 0.82) ,
	# (0.736, 0.762) ,
	# (0.896, 0.77) ,
	# (0.308, 0.972) ,
	# (0.684, 0.872) ,
	# (0.912, 0.946) )

	
	cand_start_locs = (
	(0.042, 0.038) ,
	(0.098, 0.124) ,
	(0.27, 0.03) ,
	(0.454, 0.114) ,
	(0.614, 0.244) ,
	(0.72, 0.026) ,
	(0.754, 0.106) ,
	(0.79, 0.264) ,
	(0.84, 0.33) ,
	(0.842, 0.534) ,
	(0.682, 0.428) ,
	(0.544, 0.608) ,
	(0.296, 0.382) ,
	(0.234, 0.472) ,
	(0.056, 0.508) ,
	(0.086, 0.238) ,
	(0.046, 0.704) ,
	(0.026, 0.872) ,
	(0.314, 0.974) ,
	(0.372, 0.77) ,
	(0.25, 0.636) ,
	(0.54, 0.818) ,
	(0.628, 0.88) ,
	(0.736, 0.762) ,
	(0.9, 0.736) ,
	(0.912, 0.948))

	# for loc in cand_start_locs:
	# 	#print loc
	# 	pygame.draw.circle(screen, (0,255,0), scale(loc), 5)

			
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
	
	myfont = pygame.font.SysFont("arial", 20)
	while True:
		mouseClick = Update()
		if mouseClick != None:
			loc =  mouseClick[0]/500.0, mouseClick[1]/500.0
			loc =  rrt_loc((mouseClick[0],mouseClick[1]))
			print [round(loc[0][0] ,3), round(loc[0][1] ,3)], ","
			#pygame.draw.circle(screen, (255,255,0), mouseClick, 7)
			pygame.draw.circle(screen, (0,0,255), mouseClick, 2)
			label = myfont.render(str(loc), 3, (0,0,0))
			label_loc = (mouseClick[0] -30, mouseClick[1] - 20)
			screen.blit(label,label_loc)
			
		pygame.time.delay(10)

		                    

if __name__ == '__main__':
    main()

