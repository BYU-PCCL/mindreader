#import numpy as np
import autograd.numpy as np
import scipy.stats as ss
from dautograd import dgrad_named

from erps import *

#
# ==================================================================
#
'''

Expects a model object X.  This object must support the following methods:

- X.global_init
  input: none
  output: global_vars

- X.state_init
  input: state_conds
  output: sampled initial state structure

- X.trans
  input: previous state
  output: sampled next state

- X.obs
  input: state
  output: sampled observation (obs_t)

- X.cond
  input: obs_t
  output: none
    describes how to condition an observation

'''


class PF( object ):
    def __init__( self, model=None, cnt=100 ):

        self.cnt = cnt  # number of particles
        self.model = model
        self.cond_data_db = {}

        self.cur_trace_score = 0.0

        # register the available elementary random primitives
        self.choice = self.make_erp( choice_erp )
        self.randn = self.make_erp( randn_erp )
        self.flip = self.make_erp( flip_erp )
        self.rand = self.make_erp( rand_erp )
        self.beta = self.make_erp( beta_erp )

    def init_particles( self, glob_conds=None, state_conds=None ):

        self.part_gs = []
        self.part_score = []
        self.part_state = []

        for i in range( self.cnt ):
            gs = self.model.global_init( self, glob_conds )
            self.part_gs.append( gs )

            ts = self.model.state_init( self, gs, state_conds )
            self.part_state.append( ts )

            self.part_score.append( 0.0 )  # log(1.0)

    def step( self, obs_t=None, glob_conds=None, state_conds=None ):

        self.cond_data_db = {}
        if not obs_t == None:
            self.model.condition( self, obs_t )

        for i in range( self.cnt ):
            gs = self.part_gs[i]
            ts = self.part_state[i]

            # new state
            tsp = self.model.trans( self, gs, ts,
                                    glob_conds=glob_conds, state_conds=state_conds )

            self.cur_trace_score = 0.0

            # generate (or condition on) observation
            obs = self.model.obs( self, gs, tsp,
                                  glob_conds=glob_conds, state_conds=state_conds )

            self.part_state[i] = tsp
            self.part_score[i] = self.part_score[i] + self.cur_trace_score

        return tsp, obs


    def resample( self ):
        pass

    def condition( self, name=None, value=None ):
        self.cond_data_db[ name ] = value

    def set_model( self, model=None ):
        self.model = model # should be an object

    def make_erp( self, erp_class ):
        return lambda *args, **kwargs: self.do_erp( erp_class, *args, **kwargs )

    def do_erp( self, erp_class, *args, **kwargs ):

        if kwargs.has_key( 'name' ):
            name = kwargs['name']
            del kwargs['name']
        else:
            raise(Exception('All ERPs must have a name!'))

        if self.cond_data_db.has_key( name ):
            new_val = self.cond_data_db[ name ]
        else:
            # we always sample from the prior
            new_val = erp_class.sample( *args, **kwargs )

        erp_score = erp_class.score( new_val, *args, **kwargs )

        self.cur_trace_score += erp_score

        return new_val

    def analyze( self ):
        pass

# done with PF class

#
# ==================================================================
#
    
class BBVI( object ):

    def __init__( self, model=None ):

        self.model = model
        self.inject_q_objs = False

        self.var_type_db = {}
        self.var_params_db = {}
        self.cond_data_db = {}

        # this gets reset at each gradient iteration
        self.grad_db = {}

        # this gets reset at each rollout
        self.cur_grad_db = {}
        self.cur_trace_score = 0.0

        self.always_sample = False

        # register the available elementary random primitives
        self.choice = self.make_erp( choice_erp )
        self.randn = self.make_erp( randn_erp )
        self.flip = self.make_erp( flip_erp )
        self.rand = self.make_erp( rand_erp )
        self.beta = self.make_erp( beta_erp )

        self.opt_callback = None

#
# ------------------------------------------------------
#

    def condition( self, name=None, value=None ):
        self.cond_data_db[ name ] = value

    def make_erp( self, erp_class ):

        # we use autograd to automatically create a differentiable
        # score function for each ERP type.
        erp_class.var_grads = {}
        for p in erp_class.diffparms():
            erp_class.var_grads[p] = dgrad_named( erp_class.score, p )

        return lambda *args, **kwargs: self.do_var_erp( erp_class, *args, **kwargs )

    def set_model( self, model=None ):
        self.model = model # should be an object

#
# ------------------------------------------------------
#

    def run_model( self ):
        self.cur_trace_score = 0.0
        self.cur_grad_db = {}
        return self.model.run( self ) # passes this Q object in as second parameter

#
# ------------------------------------------------------
#

    def analyze( self ):
        if self.model == None:
            raise(Exception('Must specify a model before analysis'))

        self.inject_q_objs = True
        self.run_model()
        self.inject_q_objs = False

        #
        # collect and analyze results!
        #

        # initialize all of the gradient accumulators
        for k in self.cur_grad_db:
            self.grad_db[ k ] = {}
            for j in self.cur_grad_db[k]:
                self.grad_db[ k ][ j ] = 0.0

#
# ------------------------------------------------------
#

    def calc_rolled_out_gradient( self, cnt=100 ):

        np.set_printoptions( precision=2, linewidth=200 )

        scores = []

        # normalize
        for k in self.cur_grad_db:
            for j in self.cur_grad_db[k]:
                self.grad_db[ k ][ j ] = []

        # perform cnt rollouts
        for i in range(cnt):
            self.run_model()
            scores.append( self.cur_trace_score )
            for k in self.cur_grad_db:
                for j in self.cur_grad_db[k]:
                    self.grad_db[ k ][ j ].append( self.cur_grad_db[ k ][ j ] )

        # our baseline is just the median
        scores = np.atleast_2d( scores )
        total_score = np.sum( scores )
#        scores = scores - np.mean( scores )
        scores = scores - np.median( scores )

        # normalize
        for k in self.cur_grad_db:
            for j in self.cur_grad_db[k]:
#                print "%s - %s" % (k,j)
#                tmp = self.grad_db[ k ][ j ]
#                print tmp[0].shape
#                self.grad_db[ k ][ j ] = np.dot( scores, np.vstack( tmp ) ) / cnt

                tmp = self.grad_db[ k ][ j ]
                tmpdims = len(tmp[0].shape)
                grads = np.stack( tmp, axis=tmpdims )

                newshape = np.ones((tmpdims+1),dtype=int)
                newshape[tmpdims] = scores.shape[1]
                tmpscores = np.reshape( scores, newshape )

                self.grad_db[ k ][ j ] = np.sum( tmpscores * grads, axis=tmpdims ) / cnt


        return total_score / float(cnt)
#
# ------------------------------------------------------
#
    
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

#
# ------------------------------------------------------
#
    
    def opt_sgd( self, alpha=0.01, itercnt=100, rolloutcnt=10 ):
        results = []
        scores = []

        for iter in range( 0, itercnt ):
            score = self.calc_rolled_out_gradient( cnt=rolloutcnt )
            print "\n%d: %.2f" % ( iter, score )
            results.append( np.copy( self.var_params_db[ "gloc" ][ "p" ] ) )
            scores.append( score )

            for k in self.grad_db:
                for j in self.grad_db[ k ]:
                    self.var_params_db[ k ][ j ] -= alpha * self.grad_db[ k ][ j ]

        return results, scores
#
# ------------------------------------------------------
#

    def opt_adam( self, alpha=0.01, beta_1=0.9, beta_2=0.999, epsilon=10e-8, itercnt=100, rolloutcnt=10 ):
        results = []
        scores = []

        m_t = {}
        v_t = {}
        for k in self.grad_db:
            m_t[k] = {}
            v_t[k] = {}
            for j in self.grad_db[ k ]:
                m_t[k][j] = 0
                v_t[k][j] = 0

        for iter in range( 0, itercnt ):

            score = self.calc_rolled_out_gradient( cnt=rolloutcnt )
#            print "\n%d: %.2f" % ( iter, score )

            if not self.opt_callback == None:
                self.opt_callback()
#            results.append( np.copy( self.var_params_db[ "gloc" ][ "p" ] ) )
            scores.append( score )

            for k in self.grad_db:
                for j in self.grad_db[ k ]:
                    m_t[k][j] = beta_1 * m_t[k][j] + (1.0 - beta_1) * self.grad_db[ k ][ j ]
                    v_t[k][j] = beta_2 * v_t[k][j] + (1.0 - beta_2) * self.grad_db[ k ][ j ]**2.0

            for k in self.grad_db:
                for j in self.grad_db[ k ]:
                    m_hat_t = m_t[k][j] / (1.0 - beta_1**float(iter+1))
                    v_hat_t = v_t[k][j] / (1.0 - beta_2**float(iter+1))

                    new_val = self.var_params_db[ k ][ j ] - alpha * m_hat_t / (v_hat_t**0.5 + epsilon)

                    erp_class = self.var_type_db[ k ]
                    new_val = erp_class.project_param( j, new_val )
            
                    self.var_params_db[ k ][ j ] = new_val

        return results, scores
 
#
# ==================================================================
#
