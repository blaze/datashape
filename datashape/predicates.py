from .util import collect
from .coretypes import *

# https://github.com/ContinuumIO/datashape/blob/master/docs/source/types.rst

dimension_types = (Fixed, Var, Ellipsis)

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
