from __future__ import absolute_import

import numbers

from itertools import chain

import datashape
from .dispatch import dispatch
from .coretypes import Record, CType, DataShape, Var, Fixed, Option


__all__ = 'to_json', 'from_json'


@dispatch(basestring)
def to_json(s):
    return s


@dispatch(Record)
def to_json(ds):
    return dict(names=ds.names,
                types=dict((name, to_json(typ)) for name, typ in ds.fields))


@dispatch((Option, CType))
def to_json(u):
    return to_json(str(u))


@dispatch(Var)
def to_json(v):
    return 'var'


@dispatch(Fixed)
def to_json(f):
    return int(f)


@dispatch(DataShape)
def to_json(ds):
    return dict(shape=list(map(to_json, ds.shape)),
                measure=to_json(ds.measure))


@dispatch(dict)
def from_json(d):
    if 'measure' in d and 'shape' in d:
        # toplevel dshape
        shape = tuple(from_json(dim) for dim in d['shape'])
        measure = from_json(d['measure']),
        return DataShape(*chain(shape, measure))
    else:
        # record dshape
        types = d['types']
        return Record([(f, from_json(types[f])) for f in d['names']])


@dispatch(basestring)
def from_json(s):
    try:
        return datashape.dshape(s).measure
    except datashape.DataShapeSyntaxError:
        return getattr(datashape, s, s)


@dispatch(numbers.Number)
def from_json(n):
    return n
