from datashape.coretypes import *
from multipledispatch import dispatch

@dispatch(int)
def discover(i):
    return int64

@dispatch(float)
def discover(f):
    return float64

@dispatch(str)
def discover(s):
    return string

@dispatch((tuple, list))
def discover(seq):
    types = list(map(discover, seq))
    if len(set(types)) == 1:
        return len(seq) * types[0]
