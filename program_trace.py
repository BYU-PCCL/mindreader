import autograd.numpy as np
import scipy.stats as ss

from erps import *


class ProgramTrace(object):
	def __init__(self, model=None):
		self.model = model
		self.cond_data_db = {}
		self.cur_trace_score = 0.0

		# register the available elementary random primitives
		self.choice = self.make_erp( choice_erp )
		self.randn = self.make_erp( randn_erp )
		self.flip = self.make_erp( flip_erp )
		self.rand = self.make_erp( rand_erp )
		self.beta = self.make_erp( beta_erp )


	def condition(self, name=None, value=None):
		self.cond_data_db[name] = value
	
	def set_model(self, model=None):
		self.model = model

	def run_model(self):
		self.cur_trace_score = 0.0
		self.model.run(self)
		return self.cur_trace_score

	def make_erp( self, erp_class ):
		return lambda *args, **kwargs: self.do_erp( erp_class, *args, **kwargs )

	def do_erp( self, erp_class, *args, **kwargs ):

		if kwargs.has_key( 'name' ):
			name = kwargs['name']
			del kwargs['name']
		else:
			raise(Exception('All ERPs must have a name!'))

		if self.cond_data_db.has_key( name ):
			print "has name:", name
			new_val = self.cond_data_db[ name ]
		else:
		    # we always sample from the prior
			print "sample from prior for name:", name
			new_val = erp_class.sample( *args, **kwargs )

		erp_score = erp_class.score( new_val, *args, **kwargs )
		print "erp_score:", erp_score
		self.cur_trace_score += erp_score

		return new_val


		








