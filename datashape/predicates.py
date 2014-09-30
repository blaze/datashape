from .util import collect, dshape
from .internal_utils import remove
from .coretypes import *

# https://github.com/ContinuumIO/datashape/blob/master/docs/source/types.rst

__all__ = ['isdimension', 'ishomogeneous', 'istabular', 'isfixed', 'isscalar']

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


def ishomogeneous(ds):
    """ Does datashape contain only one dtype?

    >>> ishomogeneous(int32)
    True
    >>> ishomogeneous('var * 3 * string')
    True
    >>> ishomogeneous('var * {name: string, amount: int}')
    False
    """
    ds = dshape(ds)
    return len(set(remove(isdimension, collect(isunit, ds)))) == 1


def _dimensions(ds):
    """ Number of dimensions of datashape

    Interprets records as dimensional

    >>> _dimensions(int32)
    0
    >>> _dimensions(10 * int32)
    1
    >>> _dimensions('var * 10 * int')
    2
    >>> _dimensions('var * {name: string, amount: int}')
    2
    """
    ds = dshape(ds)
    if isdimension(ds[0]):
        return 1 + _dimensions(ds.subarray(1))
    if isinstance(ds[0], Record):
        return 1 + max(map(_dimensions, ds[0].types))
    if len(ds) == 1 and isunit(ds[0]):
        return 0
    if isinstance(ds[0], Option):
        return _dimensions(ds[0].ty)
    raise NotImplementedError('Can not compute dimensions for %s' % ds)


def isfixed(ds):
    """ Contains no variable dimensions

    >>> isfixed('10 * int')
    True
    >>> isfixed('var * int')
    False
    >>> isfixed('10 * {name: string, amount: int}')
    True
    >>> isfixed('10 * {name: string, amounts: var * int}')
    False
    """
    ds = dshape(ds)
    if isinstance(ds[0], TypeVar):
        return None  # don't know
    if isinstance(ds[0], Var):
        return False
    if isinstance(ds[0], Record):
        return all(map(isfixed, ds[0].types))
    if len(ds) > 1:
        return isfixed(ds.subarray(1))
    return True


def istabular(ds):
    """ Can be represented by a two dimensional with fixed columns

    >>> istabular('var * 3 * int')
    True
    >>> istabular('var * {name: string, amount: int}')
    True
    >>> istabular('var * 10 * 3 * int')
    False
    >>> istabular('10 * var * int')
    False
    """
    ds = dshape(ds)
    return _dimensions(ds) == 2 and isfixed(ds.subarray(1))


def isscalar(ds):
    """ Has no dimensions

    >>> isscalar('int')
    True
    >>> isscalar('3 * int')
    False
    >>> isscalar('{name: string, amount: int}')
    True
    """
    ds = dshape(ds)
    return not ds.shape
