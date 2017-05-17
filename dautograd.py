from autograd import *
from autograd.core import *

def dgrad_named(fun, argname):
    """
    Returns a function which computes the gradient of `fun` with respect to
    named argument`argname`. The returned function takes the same
    arguments as `fun`, but returns the gradient instead. The function `fun`
    should be scalar-valued. The gradient has the same type as the argument."""
    @attach_name_and_doc(fun, argname, 'Gradient')
    def gradfun(*args,**kwargs):
         return backward_pass(*dforward_pass(fun,args,kwargs,argname))
    return gradfun

def dforward_pass(fun, args, kwargs, argname):
    tape = CalculationTape()
    arg_wrt = kwargs[argname]
    start_node = new_node(safe_type(getval(arg_wrt)), [tape])
    args = list(args)
    kwargs[argname] = merge_tapes(start_node, arg_wrt)
    try: end_node = fun(*args, **kwargs)
    except Exception as e: add_extra_error_message(e)
    return start_node, end_node, tape
