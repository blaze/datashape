from __future__ import print_function, division, absolute_import

import numpy as np
from dateutil.parser import parse as dateparse
from datetime import datetime, date
from multipledispatch import dispatch
from multipledispatch.utils import _toposort, groupby
from time import strptime

from .coretypes import (int32, int64, float64, bool_, complex128, datetime_,
                        Option, isdimension, var, from_numpy, Tuple,
                        Record, string, Null, DataShape, real, date_)
from .py2help import _strtypes


__all__ = ['discover']


null = Null()


@dispatch(int)
def discover(i):
    return int64


@dispatch(float)
def discover(f):
    return float64


@dispatch(bool)
def discover(b):
    return bool_


@dispatch(complex)
def discover(z):
    return complex128


@dispatch(datetime)
def discover(dt):
    return datetime_


@dispatch((type(None), Null))
def discover(i):
    return null


bools = {'False': False,
         'false': False,
         'True': True,
         'true': True}


string_coercions = [int, float, bools.__getitem__, dateparse]


@dispatch(_strtypes)
def discover(s):
    if not s:
        return null
    for f in string_coercions:
        try:
            return discover(f(s))
        except:
            pass

    return string


@dispatch((tuple, list))
def discover(seq):
    types = list(map(discover, seq))
    typ = unite(types)
    if not typ:
        return Tuple(types)
    else:
        return len(types) * typ

def isnull(ds):
    return ds == null or ds == DataShape(null)


def unite(dshapes):
    """ Unite possibly disparate datashapes to common denominator

    >>> unite([10 * (2 * int32), 20 * (2 * int32)])
    dshape("var * 2 * int32")

    >>> unite([int32, int32, null, int32])
    option[int32]
    """
    if len(set(dshapes)) == 1:
        return dshapes[0]
    if any(map(isnull, dshapes)):
        base = unite([ds for ds in dshapes if not isnull(ds)])
        if base:
            return Option(base)
    try:
        if all(isdimension(ds[0]) for ds in dshapes):
            dims = [ds[0] for ds in dshapes]
            base = unite([ds.subshape[0] for ds in dshapes])
            if base:
                if len(set(dims)) == 1:
                    return dims[0] * unite([ds.subshape[0] for ds in dshapes])
                else:
                    return var * unite([ds.subshape[0] for ds in dshapes])
    except KeyError:
        pass

    # All tuples of the same length
    if (all(isinstance(ds, Tuple) for ds in dshapes) and
        len(set(map(len, dshapes))) == 1):  # same length
        bases = [unite([ds.dshapes[i] for ds in dshapes])
                                      for i in range(len(dshapes[0].dshapes))]
        if not any(b is null for b in bases):
            return Tuple(bases)

    # All records with the same names
    if (all(isinstance(ds, Record) for ds in dshapes) and
            len(set(tuple(ds.names) for ds in dshapes)) == 1): # same names
        names = dshapes[0].names
        values = [unite([unpack(ds.fields[name]) for ds in dshapes])
                                                 for name in names]
        if not any(v is null for v in values):
            return Record(list(zip(names, values)))

    if len(dshapes) > 10:
        return lowest_common_dshape(dshapes)


def unpack(ds):
    """ Unpack DataShape constructor if unnecessary

    Record packs inputs in DataShape containers.  This unpacks it.

    >>> from datashape import dshape
    >>> unpack(dshape('string'))
    ctype("string")
    """
    if isinstance(ds, DataShape) and len(ds) == 1:
        return ds[0]
    else:
        return ds


@dispatch(dict)
def discover(d):
    return Record([[k, discover(d[k])] for k in sorted(d)])


@dispatch(np.number)
def discover(n):
    return from_numpy((), type(n))


@dispatch(np.ndarray)
def discover(X):
    return from_numpy(X.shape, X.dtype)


# E.g. int64 can be turned into a string
edges = [(string, int64),
         (string, real),
         (string, date_),
         (string, datetime_),
         (int64, int32),
         (real, int64),
         (string, null)]

edges = groupby(lambda x: x[1], edges)
edges = dict((k, set(a for a, b in v)) for k, v in edges.items())

def lowest_common_dshape(dshapes):
    """ Find common shared dshape

    >>> lowest_common_dshape([int32, int64, float64])
    ctype("float64")

    >>> lowest_common_dshape([int32, int64])
    ctype("int64")

    >>> lowest_common_dshape([string, int64])
    ctype("string")
    """
    s = _toposort(edges)
    return max(dshapes, key=s.index)
