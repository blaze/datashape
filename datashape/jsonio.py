from __future__ import absolute_import

import numbers
import json

from itertools import chain

import datashape
from .dispatch import dispatch
from .coretypes import Record, CType, DataShape, Var, Fixed, Option, String
from .py2help import basestring


__all__ = 'jsonify', 'unjsonify', 'to_json', 'from_json'


def jsonify(ds):
    """Convert a datashape to a JSON string.

    Parameters
    ----------
    ds : DataShape
        A DataShape

    Returns
    -------
    s : str
        A valid JSON string

    Examples
    --------
    >>> from datashape import dshape
    >>> jsonify(dshape("int32"))
    '{"shape": [], "measure": "int32"}'
    >>> jsonify(dshape("10 * int32"))
    '{"shape": [10], "measure": "int32"}'
    >>> jsonify(dshape('?string["A"]'))
    '{"shape": [], "measure": "?string[\\'A\\']"}'
    >>> jsonify(dshape("var * {a: ?int32, b: float64}"))
    '{"shape": ["var"], "measure": {"names": ["a", "b"], "types": {"a": "?int32", "b": "float64"}}}'
    """
    return json.dumps(to_json(ds))


def unjsonify(js):
    """Reconstruct a DataShape instance from a JSON blob.

    Parameters
    ----------
    js : str
        A JSON blob encoding a DataShape

    Returns
    -------
    ds : DataShape
        A DataShape instance

    Examples
    --------
    >>> unjsonify('{"shape": [], "measure": "int32"}')
    dshape("int32")
    >>> unjsonify('{"shape": [10, 20], "measure": "?string[\\'A\\']"}')
    dshape("10 * 20 * ?string['A']")
    >>> unjsonify('''{
    ...     "shape": ["var"],
    ...     "measure": {
    ...         "names": ["a", "b"],
    ...         "types": {
    ...             "a": "?int32",
    ...             "b": "float64"
    ...         }
    ...     }
    ... }''')
    dshape("var * {a: ?int32, b: float64}")
    """
    return from_json(json.loads(js))


@dispatch(basestring)
def to_json(s):
    return s


@dispatch(Record)
def to_json(ds):
    return dict(names=ds.names,
                types=dict((name, to_json(typ)) for name, typ in ds.fields))


@dispatch((Option, CType, String))
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
    if hasattr(datashape, s):
        return getattr(datashape, s)
    return datashape.dshape(s).measure


@dispatch(numbers.Number)
def from_json(n):
    return n
