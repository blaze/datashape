from __future__ import absolute_import

import json

from .dispatch import dispatch
from .coretypes import Record, CType, DataShape, Var, Fixed


@dispatch((str, unicode))
def to_json(s):
    return s


@dispatch(Record)
def to_json(ds):
    return dict((name, to_json(typ)) for name, typ in ds.fields)


@dispatch(CType)
def to_json(u):
    return to_json(str(u))


def jsonify(dshape):
    return json.dumps(to_json(dshape))


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
