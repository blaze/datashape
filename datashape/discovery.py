from __future__ import print_function, division, absolute_import

import numpy as np
from dateutil.parser import parse as dateparse
from datetime import datetime, date
from multipledispatch import dispatch
from time import strptime

from .coretypes import (int32, int64, float64, bool_, complex128, datetime_,
                        Option, isdimension, var, from_numpy, Tuple,
                        Record, string)
from .py2help import _strtypes


__all__ = ['discover']


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


bools = {'False': False,
         'false': False,
         'True': True,
         'true': True}


string_coercions = [int, float, bools.__getitem__, dateparse]


@dispatch(_strtypes)
def discover(s):
    if not s:
        return None
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


def unite(dshapes):
    """ Unite possibly disparate datashapes to common denominator

    >>> unite([10 * (2 * int32), 20 * (2 * int32)])
    dshape("var * 2 * int32")

    >>> unite([int32, int32, None, int32])
    option[int32]
    """
    if len(set(dshapes)) == 1:
        return dshapes[0]
    if any(ds is None for ds in dshapes):
        base = unite(list(filter(None, dshapes)))
        if base:
            return Option(base)
    if all(isdimension(ds[0]) for ds in dshapes):
        dims = [ds[0] for ds in dshapes]
        if len(set(dims)) == 1:
            return dims[0] * unite([ds.subshape[0] for ds in dshapes])
        else:
            return var * unite([ds.subshape[0] for ds in dshapes])

    if (all(isinstance(ds, Tuple) for ds in dshapes) and
        len(set(map(len, dshapes))) == 1):
        bases = [unite([ds.dshapes[i] for ds in dshapes])
                                      for i in range(len(dshapes))]
        print(bases)
        if not any(b is None for b in bases):
            return Tuple(bases)


@dispatch(dict)
def discover(d):
    return Record([[k, discover(d[k])] for k in sorted(d)])


@dispatch(np.number)
def discover(n):
    return from_numpy((), type(n))


@dispatch(np.ndarray)
def discover(X):
    return from_numpy(X.shape, X.dtype)
