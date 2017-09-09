import autograd.numpy as np
import scipy.stats as ss
import scipy.special as sp

from autograd.core import primitive

a2d = np.atleast_2d

beta = primitive( sp.beta )

def make_grad_beta( ans, X, sz=(1,1), a=1, b=1 ):
    def gradient_product( g ):
        pass
    return gradient_product

beta.defgrad( make_grad_beta )

#
# ==================================================================
#

class choice_erp:
    @staticmethod
    def diffparms():
        return ["p"]

    @staticmethod
    def sample( p=[1.0] ):
        p = p.ravel()
        p /= np.sum( p )  # ties the parameters together
        return np.random.choice( range(len(p)), p=p )

    @staticmethod
    def score( X, p=[1.0] ):
        p = p.ravel()
        p /= np.sum( p )  # ties the parameters together
        return np.sum( np.log( p[X] ) )

    @staticmethod        
    def new_var_params( p=[1.0] ):
        cnt = np.prod( p.shape )
        return { "p": (1.0/(float(cnt))) * np.ones( p.shape ) }

    @staticmethod
    def project_param( name, val ):
        if name == 'p':
            val = a2d( val )
            val[val<0.0] = 0.0
            # val[val>1.0] = 1.0 # XXX optional, since we renormalize
            return val
        else:
            return val

# -------------------------------------------
    
class randn_erp:
    @staticmethod
    def diffparms():
        return ["mu","sigma"]

    @staticmethod
    def sample( sz=(1,1), mu=0.0, sigma=1.0 ):
        return mu + sigma*np.random.randn( sz[0], sz[1] )
    
    @staticmethod
    def score( X, sz=(1,1), mu=0.0, sigma=1.0 ):
        return np.sum( ss.norm.logpdf( X, loc=mu, scale=sigma ) )

    @staticmethod    
    def new_var_params( sz=(1,1), mu=0.0, sigma=1.0 ):
        return { "sz": sz,
                 "mu": np.zeros( sz ),
                 "sigma": np.ones( sz ) }

    @staticmethod
    def project_param( name, val ):
        return val

# -------------------------------------------
    
class rand_erp:

    # we use np.abs to cover the case where min > max.  see note below.

    @staticmethod
    def diffparms():
        return ["min", "max"]

    @staticmethod
    def sample( sz=(1,1), min=0, max=1 ):
        return np.minimum(min,max) + np.abs(max-min) * np.random.rand( sz[0], sz[1] )
    
    @staticmethod
    def score( X, sz=(1,1), min=0, max=1 ):
        return np.sum( np.log( np.abs(max-min) ) )

    @staticmethod    
    def new_var_params( sz=(1,1), min=0, max=0 ):
        return { "sz": sz,
                 "min": np.zeros(sz),
                 "max": np.ones(sz) }

    @staticmethod
    def project_param( name, val ):
        # XXX one way to project these parameters is to swap the
        # values if min is ever > max.  But, the API only projects one
        # parameter at a time, and swapping involves a joint
        # projection.  that's why we use np.abs to cover the case
        # where min>max
        val[ val<0.0 ] = 0.0
        val[ val>1.0 ] = 1.0
        return val

# -------------------------------------------

class beta_erp:
    @staticmethod
    def diffparms():
        return ["a", "b"]

    @staticmethod
    def sample( sz=(1,1), a=1, b=1 ):
        return np.random.beta( a, b, size=sz )

    # \frac{1}{B(\alpha, \beta)} x^{\alpha - 1}(1 - x)^{\beta - 1}XXX    
    @staticmethod
    def score( X, sz=(1,1), a=1, b=1 ):
        return (1.0/np.beta()) * np.power( X, a-1 ) * np.power( 1.0-X, b-1 )

    @staticmethod    
    def new_var_params( sz=(1,1), a=1, b=1 ):
        return { "sz": sz,
                 "a": np.ones(sz),
                 "b": np.ones(sz) }

    @staticmethod
    def project_param( name, val ):
        val[ val<0.0 ] = 0.0
        return val

# -------------------------------------------

class flip_erp:
    @staticmethod
    def diffparms():
        return ["p"]

    @staticmethod
    def sample( sz=None, p=0.5 ):
        p = a2d( p )
        if sz == None:
            sz = p.shape
        #print "p=", p
        #print "sz=", sz
        return np.random.rand( *sz ) < p

    @staticmethod
    def score( X, sz=None, p=0.5 ):
        epsilon = 1e-20
        return np.sum( X * np.log(p+epsilon) - (1.0-X)*np.log(1.0-p+epsilon) )

    @staticmethod
    def new_var_params( sz=None, p=0.5 ):
        p = a2d( p )
        if sz == None:
            sz = p.shape
        return { "sz": sz,
                 "p": 0.5*np.ones( sz ) }

    @staticmethod
    def project_param( name, val ):
        if name == 'p':
            val = a2d( val )
            val[val<0.0] = 0.0
            val[val>1.0] = 1.0
            return val
        else:
            return val
