
from methods import * 
from rrt_smooth import *
#np.random.seed(218)




def main(test_name = "grid", smart_intruder=False, t=0, headless = True):

	record = {}

	cand_locs = (
	(0.042, 0.038) , (0.098, 0.124) , (0.27, 0.03) ,
	(0.454, 0.114) , (0.614, 0.244) , (0.72, 0.026) ,
	(0.754, 0.106) , (0.79, 0.264) , (0.84, 0.33) ,
	(0.842, 0.534) , (0.682, 0.428) , (0.544, 0.608) ,
	(0.296, 0.382) , (0.234, 0.472) , (0.056, 0.508) ,
	(0.086, 0.238) , (0.046, 0.704) , (0.026, 0.872) ,
	(0.314, 0.974) , (0.372, 0.77) , (0.25, 0.636) ,
	(0.54, 0.818) , (0.628, 0.88) , (0.736, 0.762) ,
	(0.9, 0.736) , (0.912, 0.948))

	

	###################################################################################
	#                                  INIT
	###################################################################################
	
	if not headless:
		xdim = 500
		ydim = 500
		backgroundFileName = "./cnts_inv.png"
		background = pygame.image.load(backgroundFileName)
		background = pygame.transform.scale(background, (xdim, ydim))
		backgroundRect = background.get_rect()
		screen, clock = InitScreen(xdim, ydim)
		screen.fill((255,255,255))
		s = pygame.Surface((xdim,ydim))  	# the size of your rect
		s.set_alpha(0)                		# alpha level
		s.fill((255,255,255))           	# this fills the entire surface
		screen.blit(s, (0,0))
		screen.blit(background, backgroundRect)



	########################CORE####################################
	paths = load_data("NaiveAgentPaths/" + test_name + "_paths")
	#paths = combine_paths(paths)


	polygonSegments = load_polygons()

	# ISOVIST
 	isovist = iso.Isovist( polygonSegments )

	X1, Y1, X2, Y2 = polygons_to_segments(load_polygons_here())
	###############################################################

	if not headless:
		for polygon in polygonSegments:
			for segment in polygon:
				pygame.draw.line(screen, (0, 0, 0), segment[0], segment[1],2)
		
		Update()

  ###################################################################################
  ###################################################################################

  	###########################
	# Smooth RRT - RRT refiner
	###########################
	paths = smooth(paths)
	###########################

	INTRUDER_SEEN_COUNT = 0 

	K = 1
	for k in tqdm(xrange(K)):
		###############################################################################
		#                        SAMPLE INTRUDER PATH
		###############################################################################
		# SAMPLE INTRUDER
		while True:
			IntruderStart = scale_up( cand_locs[ np.random.choice(len(cand_locs)) ] )
			IntruderGoal = scale_up( cand_locs[ np.random.choice(len(cand_locs)) ] )
			if np.sum( (np.asarray(IntruderStart) - np.asarray(IntruderGoal))**2.0 ) >= 100000:
				break

		if not headless:
			pygame.draw.circle(screen, (0,255,0), IntruderStart, 5)
			pygame.draw.circle(screen, (255,0,0), IntruderGoal, 5)

			Update()

		record[k] = []
		record[k].append(IntruderStart)
		record[k].append(IntruderGoal)


		# GET INTRUDER PATH AND DRAW INTENDED PATH
		intruder_path = None
		if smart_intruder:
			intruder_path = smart_intruder_rrt_par( IntruderStart, IntruderGoal,
			                                   X1, Y1, X2, Y2,
			                                   isovist, 10, 8 )
		else:
			intruder_path = run_rrt( scale_down(IntruderStart), scale_down(IntruderGoal), X1, Y1, X2, Y2)
		
		###########################
		# Smooth RRT - RRT refiner
		###########################
		intruder_path = smooth(intruder_path)
		###########################
		###############################################################################
		#				         UAV THINK/THOUGHT (PRE-PROGRAMMED)
		###############################################################################


		#THOUGHT
		UAV_path = paths


		###############################################################################
		#                                SEARCH
		###############################################################################


		intruder_seen = False
		time_seen = -1

		###############################################################################
		#                          Travel by distance
		###############################################################################

		N = 3
		#Intruder remembers what he hears
		heard = [0]*N


		#ai = 0
		ai = np.random.randint(len(UAV_path)-1)
		ii = 0

		on_going = True

		time_step = 0
		while(on_going):
			if not headless:
				s.fill((255,255,255))           	# this fills the entire surface
				screen.blit(s, (0,0))
				screen.blit(background, backgroundRect)
				for polygon in polygonSegments:
					for segment in polygon:
						pygame.draw.line(screen, (0, 0, 0), segment[0], segment[1],2)
				pygame.draw.circle(screen, (0,255,0), IntruderStart, 5)
				pygame.draw.circle(screen, (255,0,0), IntruderGoal, 5)
				
				Update()

			time_step += 1

			pre_ii = ii
			pre_ai = ai

			print "INTRUDER:"
			ii = travel(ii, intruder_path, amt=10, bound=14)
			print "\nAGENT:"
			ai = travel(ai, UAV_path, amt=25, bound=31)


			##################################### DRAW ###############################
			if not headless:
				s_point = (int(intruder_path[pre_ii][0]*500), int(intruder_path[pre_ii][1]*500))
				e_point = (int(intruder_path[ii][0]*500), int(intruder_path[ii][1]*500))
				pygame.draw.line(screen, (78,230,83), s_point, e_point, 2)


				s_point = (int(UAV_path[pre_ai][0]*500), int(UAV_path[pre_ai][1]*500))
				e_point = (int(UAV_path[ai][0]*500), int(UAV_path[ai][1]*500))
				pygame.draw.line(screen, (0,93,212), s_point, e_point, 2)


				Update()
				pygame.time.delay(70)


			############################################################################

			Intruder_curr_loc = scale_up(intruder_path[ii])
			UAV_curr_loc = scale_up(UAV_path[ai])

			intruderColor = (255,0,0)

			# Draw Polygon for intersections (isovist)
			isovist_surface = pygame.Surface((500,500)) 
			isovist_surface.set_alpha(80)

			intruder_seen, intersections = detect(UAV_curr_loc, scale_up(UAV_path[pre_ai]), Intruder_curr_loc, isovist)
			
			if not headless:
				if intersections != None:
					pygame.draw.polygon(isovist_surface, intruderColor, intersections)
					screen.blit(isovist_surface, isovist_surface.get_rect())
					Update()
			if intruder_seen:
				INTRUDER_SEEN_COUNT += 1
				time_seen = time_step
				print "Intruder Detected!"
				
				break
			

			###############################################
			#            Intruder Listens and Looks
			###############################################

			noise = noise_level(Intruder_curr_loc, UAV_curr_loc)
			
			heard.append(noise)

			#Remember only N sounds
			heard = heard[1:]

			#Did it get closer or further away
			if 0:
				if heard[-2] == 0 and heard[-1] == 0:
					print "No Noise"
				elif heard[-1] > heard[-2]:
					print "HEARD: Closer"
				elif heard[-1] <= heard[-2]:
					print "HEARD: Farther"

			#Direction sound came from
			sound_dir = direction (UAV_curr_loc, Intruder_curr_loc)

			#Look
			UAV_seen = detect(Intruder_curr_loc, scale_up(intruder_path[pre_ii]), UAV_curr_loc, isovist)

			
			if 0:
				print "INTRUDER LOC:", Intruder_curr_loc
				print "heard:", heard
				print "heard dir:", sound_dir
				print "UAV_seen:", UAV_seen

			###############################################
			print ii, len(intruder_path) - 1
			#If Intruder Reaches Goal
			if ii == len(intruder_path) - 1:
				on_going = False
				print "Intruder Reaches Goal"
				break

			#If UAV Reaches Goal - Start Path AGAIN
			if ai == len(UAV_path) - 1:
				ai = 0

			

		record[k].append(intruder_seen)
		record[k].append(time_seen)

	record["INTRUDER_SEEN_COUNT"] = INTRUDER_SEEN_COUNT
	record["INTRUDER_TIME_STEPS"] = len(intruder_path)
	#save_data(record, "NA_NI_dist_data/" + test_name + "_sample_data_" + str(t))



	########################## END ########################################

	if not headless:
		myfont = pygame.font.SysFont("arial", 20)
		while True:
			mouseClick = Update()
			pygame.time.delay(10)

			                    

if __name__ == '__main__':
    main(test_name = "grid", headless=False)
