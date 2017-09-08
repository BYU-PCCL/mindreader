import autograd.numpy as np
import scipy.stats as ss
from dautograd import dgrad_named

from erps import *


class Program_Trace(object):
	def __init__(self, model=none):
		self.model = model

		self.var_type_db = {}
		self.var_params_db = {}
		self.con_data_db = {}
		self.cur_trace_score = 0.0
		self.inject_q_objs = False

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
		return self.model.run(self)

	def make_erp(self, erp_class):
		erp_class.var_grads = {}
		for p in erp_class.diffparms():
			erp_class.var_grads[p] = dgrad_named(erp_class, p)

		return lambda *args, **kwargs: self.do_var_erp(erp_class, *args, **kwargs)

	def analyze(self):
		if self.model == None:
			raise(Exception('Must specify a model before analysis'))
		pass

	def do_var_erp( self, erp_class, *args, **kwargs ):

		if kwargs.has_key( 'name' ):
			name = kwargs['name']
			del kwargs['name']
		else:
			raise(Exception('All ERPs must have a name!'))

		self.var_type_db[ name ] = erp_class

		if self.var_params_db.has_key( name ):
			var_params = self.var_params_db[ name ]
		else:
			var_params = erp_class.new_var_params( *args, **kwargs )
			self.var_params_db[ name ] = var_params

		if self.cond_data_db.has_key( name ):
			new_val = self.cond_data_db[ name ]
			trace_score = erp_class.score( new_val, *args, **kwargs )
			my_score = -trace_score

		else:
			# we always sample from the variational distribution
			new_val = erp_class.sample( **var_params )
			# score under the variational parameters and the trace parameters
			var_score = erp_class.score( new_val, **var_params )
			trace_score = erp_class.score( new_val, *args, **kwargs )

			tmp = {}
			for p in erp_class.diffparms():            
			    tmp[p] = erp_class.var_grads[p]( new_val, **var_params )
			self.cur_grad_db[name] = tmp
			my_score = var_score - trace_score

		# XXX broadcast this score only to parents in the dependency graph!

		self.cur_trace_score += my_score

		if self.inject_q_objs:
			return new_val
		else:
			return new_val




if __name__ == '__main__':
		p = Program_Trace()
		








