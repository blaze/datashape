from .util import collect, dshape
from .internal_utils import remove
from .coretypes import *

# https://github.com/ContinuumIO/datashape/blob/master/docs/source/types.rst

__all__ = ['isdimension', 'ishomogeneous', 'istabular', 'isfixed', 'isscalar',
        'isrecord', 'iscollection', 'isnumeric', 'isboolean', 'isdatelike',
        'isreal']

dimension_types = (Fixed, Var, Ellipsis, int)

def isscalar(ds):
    """ Is this dshape a single dtype?

    >>> isscalar('int')
    True
    >>> isscalar('?int')
    True
    >>> isscalar('{name: string, amount: int}')
    False
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape) and len(ds) == 1:
        ds = ds[0]
    if isinstance(ds, Option):
        ds = ds.ty
    return isinstance(ds, Unit)


def isrecord(ds):
    """ Is this dshape a record type?

    >>> isrecord('{name: string, amount: int}')
    True
    >>> isrecord('int')
    False
    >>> isrecord('?{name: string, amount: int}')
    True
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape) and len(ds) == 1:
        ds = ds[0]
    if isinstance(ds, Option):
        ds = ds.ty
    return isinstance(ds, Record)


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
    return len(set(remove(isdimension, collect(isscalar, ds)))) == 1


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
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape) and len(ds) == 1:
        ds = ds[0]
    if isinstance(ds, Option):
        return _dimensions(ds.ty)
    if isinstance(ds, Record):
        return 1 + max(map(_dimensions, ds.types))
    if isinstance(ds, Tuple):
        return 1 + max(map(_dimensions, ds.dshapes))
    if isinstance(ds, DataShape) and isdimension(ds[0]):
        return 1 + _dimensions(ds.subshape[0])
    if isscalar(ds):
        return 0
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


def iscollection(ds):
    """ Is a collection of items, has dimension

    >>> iscollection('5 * int32')
    True
    >>> iscollection('int32')
    False
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    return isdimension(ds[0])


def isnumeric(ds):
    """ Has a numeric measure

    >>> isnumeric('int32')
    True
    >>> isnumeric('3 * ?real')
    True
    >>> isnumeric('string')
    False
    >>> isnumeric('var * {amount: ?int32}')
    False
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape):
        ds = ds.measure
    if isinstance(ds, Option):
        ds = ds.ty
    return isinstance(ds, Unit) and np.issubdtype(to_numpy_dtype(ds), np.number)


def isreal(ds):
    """ Has a numeric measure

    >>> isreal('float32')
    True
    >>> isreal('3 * ?real')
    True
    >>> isreal('string')
    False
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape):
        ds = ds.measure
    if isinstance(ds, Option):
        ds = ds.ty
    return isinstance(ds, Unit) and 'float' in str(ds)


def isboolean(ds):
    """ Has a boolean measure

    >>> isboolean('bool')
    True
    >>> isboolean('3 * ?bool')
    True
    >>> isboolean('int')
    False
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape):
        ds = ds.measure
    if isinstance(ds, Option):
        ds = ds.ty
    return ds == bool_


def isdatelike(ds):
    """ Has a date or datetime measure

    >>> isdatelike('int32')
    False
    >>> isdatelike('3 * datetime')
    True
    >>> isdatelike('?datetime')
    True
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape):
        ds = ds.measure
    if isinstance(ds, Option):
        ds = ds.ty
    return ds in (date_, datetime_)
