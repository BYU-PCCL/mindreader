# mindreader

# API 

## path_kde.py

#### Functions:

  make_gauss(x, y, ss=100)

      Parameters:
      - x and y are points between 0 and 500 (0-1 scaled up for displaying purposes)
      - standard deviation squared, or the variance/ spread of the data

      Description:
      - returns (500 x 500)  matrix with multivariate Gaussians centered at x, y with a spread of ss
   

  make_heatmap(pts, ss=100)

      Parameters:
      - pts is a list of tuples, x and y values [0,1]
      - ss is the variance

      Description:
      - combines all the multivariate Gaussians centered at each x, y point then divides by the number of points to compute the expected value for each point

  path_to_heatmap(path, ss=100)

      Parameters:
      - path is a list a tuples, x, y points values between 0 and 1 inclusive
      - ss is variance

      Description:
      - uses make_heatmap to create a 500x500 heatmap for every single point in the path given

  multiple_paths_to_heatmap(set_of_rrts, cnt=300, ss=100)

      Parameters:
      - cnt represents how many time steps we want to go through

      Description:
      - for every time step, we get all of the points at that timestep from each rrt in set_of_rrts.  Once we have all of those points, we make a heat map and store it into ‘heatmap’ at that particular time step (cnt)

 ## erps.py

      Description: 
      - stands for elementary random primitives. 
      - the purpose of the file is to write up several erps that we can use in our generative probabilistic programs
      - examples of erps are distributions we commonly use are:
        -	Bernoulli / Flip
        - Normal/Gaussian
        - Uniform
      - distributions can vary from discrete to continuous
      -	each erp needs to have 
        1. a way to sample from 
        2. a way to score a value ‘X’ – the score should be the log likelihood

## q.py

This file contains two different inference algorithms, BBVI (Black box variations Inference) and PF (Particle Filter). In our previous work, we used BBVI, but now we are focusing on the Particle Filter. 

### Particle Filter

   Internal variables:
   -	cnt : the number of particles
   -	model: the model each particle represents
   -	cond_data_db: dictionary of the random variables in the generative program that are conditioned and their respective conditioned value. Example Key:Value -> “start_position”: [0,0], where the generative program now conditions the variable named “start_position” to the x,y position of 0,0. 
   -	cur_trace_score: the current log likelihood of the trace
   -	choice, randn, flip, rand, beta: all erps registered to the inference algorithm

#### Functions:

  init_particles(glob_conds, state_conds)
  
      Parameters:
      - glob_conds: the global variables we want to condition and their values (in the intruder model, its the uav_type and path)
      - state_conds: the state variables we want to condition and their values (in the intruder model, its the time step the intruder is in its respective path)

      Internal variables:
      - part_gs : list of particles
      - part_score: parallel list of particles with their scores (log likelihoods)  
      - part_state: the state the particle is in

      Description:
      - For every particle, we initialize the model with the global conditions/or samples from the prior and append the global variables to the list 'part_gs'
      - we do the same for each particle, but instead with the state conditions and append the sampled initial state structure into 'part_state'
      - we initialize the score of each particle with 0, i.e. log(1)

  step(obs_t, glob_conds, state_conds)

      Parameters:
      - obs_t : the observations at timestep 't'
      - glob_conds : the global model conditioned variables?
      - state_conds : the state's conditioned variables

      Description:
      - the first thing we do is check if we have observations at time 't', if we do, we condition the model with those observations
      - next, for every particle, we
        1. get the global variables and state structure
        2. we call 'trans' to transition the model given the glob_conds and state_conds. This gives us a new state for the particle
        3. we call 'obs' to get either sampled observations from the prior, or the observations conditioned
        4. set the particle to the new transitioned state
        5. since we are in log space, we add the trace score to the current particle's score

      - we return the transitioned state and the observations

  condition(name, value)

      Parameters:
      - name : the name of the random variable in the generative model
      - value : the value assigned to that random variable

      Description:
      - assigns the given value to the random variable

  set_model(model):

      Description:
      - sets the model to the class's model

  make_erp(erp_class, args)

      Description:
      - this returns the class/model that represents the elementary random primitive where you can sample from and score a value using the pdf (probability density function)

  do_erp(erp_class, args)

      Description:
      - this function runs the erp function class given 'erp_class'. It will run the erp on conditioned data, or sample from the prior. Then it scores it and returns the sample. 
      
## dautograd.py

#### Functions:

  dgrad_named(fun, argname)

      Parameters:
      - fun: a function
      - argname: the name we desire to take the gradient in respect to

      Description:
      - returns a function which computes the gradient of `fun` with respect to named argument`argname`. 
      - the returned function takes the same arguments as `fun`, but returns the gradient instead. 
      - the function `fun` should be scalar-valued. The gradient has the same type as the argument.

  dforward_pass(fun, args, kwargs, argname)

      Parameters:
      - fun: a function
      - argname: the name we desire to take the gradient in respect to

      Description:

## heatcube.py

      Internal Variables
      - cand_locs: a discrete set of locations
      - polygonSegments: loads polygons from maps
      - isovist: isovist object
      - X1, Y1, X2, Y2: line segments for polygons from map
      - types: the names of the naive agent paths

      Description:
      - for every naive agent path, the data gets loaded and transformed into a heatmap
      - each heatmap is stored into an array 'results'
      - each heatmap is (flattened) by taking the expectation over time steps
      - next we take the expectation over all the heatmaps
      - data is saved as './heatcube.npy'

## intruder.py

   BasicIntruder Description:
   - the intruder model knows where it begins and ends (start and goal locations)
   - it has a belief over UAV locations and the type it is (assuming the UAV is a naive agent with a deterministic path of navigation)
   - the intruder must plan a path that minimizes detection probability
   - the intruder is unaware at what timestep the UAV is at

   Internal Variables:
   - isovist : isovist object
   - intersection_cache: cache of precomputer intersections from the isovist


#### Functions

  precompute_and_save_intersections()

      Description:
      - for every type of deterministic naive agent plan, compute the isovists as it walks from start to goal location (of the respective path)
      - cashe each of the intersections into './int_cache.cp'


  condition(Q, obs_t)

      Parameters:
      - Q: the inference algorithm
      - obs_t: tuple with seen observation in index 0 and heard observation in index 1

      Description:
      - conditions the inference algorithm's seen and heard observations in obs_t

  global_init(Q)

      Parameters:
      - Q: the inference algorithm

      Description:
      - selects the uav_type from a uniform distribution
      - returns the uav_type and its respective path

  state_init(Q, gs)

      Parameters:
      - Q: the inference algorithm
      - gs: (stands for gloab state object) tuple of the uav_type and uav_path

      Description:
      - samples a time step from a uniform distribution to represent the initial location of the uav
      - returns the sampled time step

  trans(Q, gs, ts)

      Parameters:
      - Q: inference algorithm
      - gs: global state object, a tuple of the uav_tupe and uav_path
      -

      Description:
      - gets the uav_location based on the time step given and the uav_path
      - returns the next uav_loc (step + 1)

  obs(Q, gs, ts, state_conds)

      Parameters:
      - Q: inference algorithm 
      - gs: global state object, a tuple of the uav_type and uav_path
      - ts: current time step state, a time step
      - state_conds: conditioned values, the intruder location

      Description:
      - gets the uav_location based on the time step given and the uav_path
      - gets the cached intersections for the isovists for the uav_path
      - based on the intruder location, check if agent detected the intruder
      - sample 'intruder_seen' from a weighted flip erp
        - currently this is set up as always False
      - based on the the proximity of the agent to the intruder, get noise level agent makes
      - samples a 'heard' sample from a normal distribution (randn erp)
      - returns the seen and heard samples 

## my_rrt.py

  line_intersect(X1, Y1, X2, Y2, X3, Y3, X4, Y4)

      Parameters:
      - (all the following are numpy arrays (n x 1) (1 x n) )
      - X1, Y1: point(s) for one end of the 1st line segment (may be for multiple line segments)
      - X2, Y2: point(s) for other end of the 1st line segment (may be for multiple line segments)
      - X3, Y3: point(s) for one end of the 2nd line segment (may be for multiple line segments)
      - X4, Y4: point(s) for other end of the 2nd line segment (may be for multiple line segments)

      Description:
      - adapted from http://www.mathworks.com/matlabcentral/fileexchange/27205-fast-line-segment-intersection
      - generates intersection analysis between the line segment sets given in XY1 and XY2. Code can handle coincident and parallel lines.
      - The main emphasis is on speed. The code is fully vectorized and it runs pretty fast (orders of magnitude) compared to some of the previous postings.

  load_polygons(fn)

      Parameters:
      - fn: filename

      Description:
      - loads a set of polygons
      - appends the first point to also be the last point
      - by appending the first point as the last point, it ensures that the resulting polygon is exactly closed.
      - this prepends a single point that is the mean of all the other points.
      - this is for drawing the polygons using a GL_TRIANGLE_FAN.


  polygons_to_segments(polygon_list)

      Parameters:
      - polygon_list: list of numpy arrays

      Description:
      - each nparray is a kx2 matrix, representing x,y points
      - first entry is the mean of all the points, which is SKIPPED
      - last entry in the matrix is the same as the first
      - returns x1,y1, x2,y2

  distance_to_other_points(pt, pts)

      Parameters:
      - pt: point
      - pts: all other points

      Description:
      - computes euclidean distance between pt and pts

  run_rrt_poly(start_pt, goal_pt, polygons)

      Parameters:
      - start_pt: start point (x,y coordinates)
      - goal_pt: goal point
      - polygons: the map in line segment form

      Description:
      - calls polygons_to_segments(...) on 'polygons'
      - calls run_rrt(...)

  biased_flip(prob)

      Parameters:
      - prob: the probability, value between 0 and 1

      Description:
      - randomly samples a number between 1 and 100
      - if the sample is less than the sqrt(prob * 100) returns 1

  mag(one, two)

      Parameters:
      - one: tuple of x y values
      - two: tuple of x y values

      Description:
      - finds the euclidean distance (magnitude) between the two points

  parent_count(nearest_ind, parents)

      Parameters:
      - nearest_ind: index
      - parents: list of parent indexes

      Description:
      - gets the nearest index from list of parent indexes, until it reaches the 0th index (root)
      - returns count

  parent_count_mem(nearest_ind, parents)

      Parameters:
      - nearest_ind: index
      - parents: list of parent indexes

      Description:
      - added memory lookup for parent count for time variable in heat cube

  run_rrt(start_pt, goal_point, endpoint_a_x, endpoint_a_y, endpoint_b_x, endpoint_b_y, heat)

      Parameters:
      start_pt: start point (x, y)
      goal_pt: goal point 
      endpoint_a_x: x coordinate of end point of first end point (point a)
      endpoint_a_y: y coordinate of end point of first end point (point a)
      endpoint_b_x: x coordinate of end point of second end point (point b)
      endpoint_b_y: y coordinate of end point of second end point (point b)
      heat: the heatcube (500, 500, time_steps)

      Description:
      - loops untl we hit the step_limit=20000
      - select a random point, random_point
      - flip to move toward the goal point
      - find the nearest point from random_point
      - if we move toward to the goal point, we do that instead of taking a step towards to the random point
      - however if the next step lands in an obstacle, we start again by selecting a new random point
      - if 'heat' is not None, and the next step is in a danger zone, we do a biased flip to decide if take the step or not
      - once we reach the goal (or very near it) we perculate back to the root to get the path. 
      - return the path
      - if the goal is not reached within the step limit, we return an empty list

  create_temp_heat_map()

      Description:
      - creates a temporary heat cube for testing purposes (500,500, time_steps). In this case time_steps = 200.

## int_test.py

   File Description:
     - this is a simple test of our particle filter and model
     - we want to infer the naive_uav's deterministic path and its current location on the path as it navigates based on observations. We are assuming observations are only sound at this point 

    Description:
    - loads 
      - isovist
      - polygon segments X1,Y1,X2,Y2
      - all uav path types
    - initalize the model as BasicIntruder
    - set particle count as 1000
    - instantiate the particle filter 'pf'
    - initialize all particles
      - every particle is assigned a uav_type with its respect path
      - every particle samples a state (time step within path)
    - We randomly choose a uav_type, uav_path, and uav_loc_on_route to represent what's actually happening. (we want the pf to infer the path and the location on the path)
    - Filtering
      - for 1000 iterations
      - condition the intruder location (0.1, 0.1)
      - we get the uav_loc and determine noise_level
      - set seen to false
      - make heard (i.e. noise_level) and seen as observations 'obs'
      - let pf step with obs and condiitoned on int_loc (intruder location)
        - all particles take a step forward according to their assigned uav_type/path and time step location
      - simulate the uav stepping (update uav_loc_on_route)
      - TODO: resample particles
      
## sim_i.py 
Simulation file for naive agents. Can be run as "headless=False" to view the behavior of the agents.

## NaiveAgentPaths
Contains files of pre-programmed naive agent paths

## rrt_smooth.py
Refines rrt paths

## methods.py
Contains commonly used the methods

## isovist.py
Calculates isovist for map

## paths
Contains line segments of map

## MODELS - General information

 Objects/Models must support the following methods:

  - X.global_init
    - input: none
    - output: global_vars
  - X.state_init
    - input: state_conds
    - output: sampled initial state structure
  - X.trans
    - input: previous state
    - output: sampled next state
  - X.obs
    - input: state
    - output: sampled observation (obs_t)
  - X.cond
    - input: obs_t
    - output: none
    - describes how to condition an observation
