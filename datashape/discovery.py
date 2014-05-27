from datashape.coretypes import *
from multipledispatch import dispatch
from time import strptime
from dateutil.parser import parse as dateparse
from datetime import datetime, date


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


@dispatch(str)
def discover(s):
    for f in string_coercions:
        try:
            return discover(f(s))
        except:
            pass

    return string


@dispatch((tuple, list))
def discover(seq):
    types = list(map(discover, seq))
    if len(set(types)) == 1:
        return len(seq) * types[0]
    return Tuple(types)


@dispatch(dict)
def discover(d):
    return Record([[k, discover(d[k])] for k in sorted(d)])
