from datashape.coretypes import *
from multipledispatch import dispatch


@dispatch(int)
def discover(i):
    return int64


@dispatch(float)
def discover(f):
    return float64


string_coercions = [int, float]


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
