from datashape.coretypes import *
from multipledispatch import dispatch


__all__ = ['discover']


@dispatch(int)
def discover(i):
    return int64


@dispatch(float)
def discover(f):
    return float64


@dispatch(bool)
def discover(f):
    return bool


bools = {'False': False,
         'false': False,
         'True': True,
         'true': True}


string_coercions = [int, float, bools.__getitem__]


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
