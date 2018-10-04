import autograd.numpy as np
import scipy.stats as ss
import pickle
import sys
import copy
from erps import *


class ProgramTrace(object):
	def __init__(self, model=None):
		self.model = model
		self.cond_data_db = {}
		self.cur_trace_score = 0.0
		self.trace = {}
		self.cache = {}
		self.obs = {}

		# register the available elementary random primitives
		self.choice = self.make_erp( choice_erp )
		self.randn = self.make_erp( randn_erp )
		self.flip = self.make_erp( flip_erp )
		self.rand = self.make_erp( rand_erp )
		self.beta = self.make_erp( beta_erp )
		self.lflip = self.make_erp( logflip_erp )
		self.clflip = self.make_erp( complogflip_erp )
		self.trnorm = self.make_erp( truncn_erp )

	def get_copy(self):
		q = ProgramTrace()

		q.model = self.model # can share
		q.cond_data_db = copy.deepcopy(self.cond_data_db)
		q.cur_trace_score = copy.deepcopy(self.cur_trace_score)
		q.cache = copy.deepcopy(self.cache)
		q.trace = copy.deepcopy(self.trace)
		q.obs = copy.deepcopy(self.obs)

		q.choice = q.make_erp( choice_erp )
		q.randn = q.make_erp( randn_erp )
		q.flip = q.make_erp( flip_erp )
		q.rand = q.make_erp( rand_erp )
		q.beta = q.make_erp( beta_erp )
		q.lflip = q.make_erp( logflip_erp )
		q.clflip = q.make_erp( complogflip_erp )
		q.trnorm = q.make_erp( truncn_erp )
		return q


	def set_obs(self, name=None, value=None):
		self.obs[name] = value

	def get_obs(self, name=None):
		if name in self.obs:
			return self.obs[name]
		return None

	def fetch_obs(self):
		return self.obs

	def get_trace(self):
		return self.trace

	def get_score(self):
		return self.cur_trace_score

	def condition(self, name=None, value=None):
		self.cond_data_db[name] = value

	def fetch_condition(self, name=None):
		# print "name:", name
		# print self.cond_data_db
		return self.cond_data_db[name]

	def get_cond_names(self):
		return self.cond_data_db.keys()
	
	def set_model(self, model=None):
		self.model = model

	def run_model(self):
		self.trace = {}
		self.cur_trace_score = 0.0
		self.model.run(self)
		# if not success:
		# 	print ("ERROR: RRT IS STUCK")
		# 	return -5000, {}
		return self.cur_trace_score, self.trace

	def keep(self, name=None, value=None):
		self.trace[name] = value

	def fetch(self, name=None):
		if name in self.trace:
			return self.trace[name]
		return None

	def keep_update(self, name=None, value=None):
		self.keep(name=name, value=value)
		self.cur_trace_score += value

	def add_trace(self, name=None, trace=None, score=None):
		self.keep(name=name, value=trace)
		self.cur_trace_score += score

	def make_erp( self, erp_class ):
		return lambda *args, **kwargs: self.do_erp( erp_class, *args, **kwargs )


	def do_erp( self, erp_class, *args, **kwargs ):

		if kwargs.has_key( 'name' ):
			name = kwargs['name']
			del kwargs['name']
		else:
			raise(Exception('All ERPs must have a name!'))

		if self.cond_data_db.has_key( name ):
			# only if it's constrained, add the log score 
			# joint probability / proposal, cancel out prior sample scores
			new_val = self.cond_data_db[ name ]

			erp_score = erp_class.score( new_val, *args, **kwargs )
			
			# special case where I have to call the same random variable again
			# specifically when the start and goal locations are the same
			# if name in self.trace:
			# 	prev_val = self.trace[name]
			# 	prev_score = erp_class.score(prev_val, *args, **kwargs)
			# 	self.cur_trace_score -= pre_score

			self.cur_trace_score += erp_score
			#print ("name:", name, "score:", erp_score)
			#print  ("-------------------------------------------total score:", self.cur_trace_score)
		else:
		    # we always sample from the prior
			new_val = erp_class.sample( *args, **kwargs )

		self.trace[name] = new_val

		return new_val


		








