from .util import collect, remove
from .coretypes import *

# https://github.com/ContinuumIO/datashape/blob/master/docs/source/types.rst

dimension_types = (Fixed, Var, Ellipsis)

isunit = lambda x: isinstance(x, Unit)

def isdimension(ds):
    """ Is a component a dimension?

    >>> isdimension(Fixed(10))
    True
    >>> isdimension(Var())
    True
    >>> isdimension(int32)
    False
    """
    return isinstance(ds, dimension_types)


def ishomogenous(ds):
    """ Does datashape contain only one dtype?

    >>> ishomogenous(int32)
    True
    >>> ishomogenous(var * (3 * string))
    True
    >>> ishomogenous(var * Record([('name', string), ('amount', int32)]))
    False
    """
    return len(set(remove(isdimension, collect(isunit, ds)))) == 1
