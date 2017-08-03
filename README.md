# mindreader

# API 


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


## path_kde.py

#### Functions:

  make_gauss(x, y, ss=100)

      Parameters:
      - x and y are points between 0 and 500 (0-1 scaled up for displaying purposes)
      - standard deviation squared, or the variance/ spread of the data

      Description:
      - Returns (500 x 500)  matrix with multivariate Gaussians centered at x, y with a spread of ss
   

  make_heatmap(pts, ss=100)

      Parameters:
      - Pts is a list of tuples, x and y values [0,1]
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
      - The purpose of the file is to write up several erps that we can use in our generative probabilistic programs
      - Examples of erps are distributions we commonly use are:
        -	Bernoulli / Flip
        - Normal/Gaussian
        - Uniform
      - Distributions can vary from discrete to continuous
      -	Each erp needs to have 
        1. a way to sample from 
        2. a way to score a value ‘X’ – the score should be the log likelihood

## q.py

This file contains two different inference algorithms, BBVI (Black box variations Inference) and PF (Particle Filter). In our previous work, we used BBVI, but now we are focusing on the Particle Filter. 

### Particle Filter

      Internal variables:
      -	cnt : the number of particles
      -	model: the model each particle represents
      -	cond_data_db: dictionary of the random variables in the generative program that are conditioned and their respective conditioned value. Example Key:Value -> “start_position”: [0,0], where the generative program now conditions the variable named “start_position” to the x,y position of 0,0. 
      -	Cur_trace_score: the current log likelihood of the trace
      -	Choice, randn, flip, rand, beta: all erps registered to the inference algorithm

#### Functions:

  init_particles(glob_conds, state_conds)
  
      Parameters:
      - glob_conds: the global variables we want to condition and their values
      - state_conds: the state variables we want to condition and their values

      Internal variables:
      - part_gs : list of particles
      - part_score: parallel list of particles with their scores (log likelihoods)  
      - part_state: the state the particle is in

      Description:
      - For every particle, we initialize the model with the global conditions and append the global variables to the list 'part_gs'
      - We do the same for each particle, but instead with the state conditions and append the sampled initial state structure into 'part_state'
      - We initialize the score of each particle with 0, i.e. log(1)

  step(obs_t, glob_conds, state_conds)

      Parameters:
      - obs_t : the observations at timestep 't'
      - glob_conds : the global model conditioned variables?
      - state_conds : the state's conditioned variables

      Description:
      - the first thing we do is check if we have observations at time 't', if we do, we condition the model with those observations
      - Next, for every particle, we
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
      - Assigns the given value to the random variable

  set_model(model):

      Description:
      - sets the model to the class's model

  make_erp(erp_class, args)

      Description:
      - This returns the class/model that represents the elementary random primitive where you can sample from and score a value using the pdf (probability density function)

  do_erp(erp_class, args)

      Description:
      - this function runs the erp function class given 'erp_class'. It will run the erp on conditioned data, or sample from the prior. Then it scores it and returns the sample. 


## sim_i.py 
Simulation file for naive agents. Can be run as "headless=True" to view the behavior of the agents.

## NaiveAgentPaths
Contains files of pre-programmed naive agent paths

## rrt_smooth.py
Refines rrt paths

## methods.py
Contains commonly used the methods

## isovist.py
Calculates isovist for map

## my_rrt.py
Creates a path between points

## paths
Contains line segments of map

