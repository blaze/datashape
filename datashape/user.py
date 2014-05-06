from datashape.dispatch import dispatch
from .coretypes import *
from .util import dshape
import sys
from datetime import date, time, datetime


__all__ = ['validate', 'issubschema', 'subset_dshape']


basetypes = np.generic, int, float, str, date, time, datetime


def isdimension(unit):
    return isinstance(unit, (Fixed, Var))


@dispatch(np.dtype, basetypes)
def validate(schema, value):
    return np.issubdtype(type(value), schema)


@dispatch(CType, basetypes)
def validate(schema, value):
    return validate(to_numpy_dtype(schema), value)


@dispatch(DataShape, (tuple, list))
def validate(schema, value):
    head = schema[0]
    return ((len(schema) == 1 and validate(head, value))
        or (isdimension(head)
       and (isinstance(head, Var) or int(head) == len(value))
       and all(validate(DataShape(*schema[1:]), item) for item in value)))


@dispatch(DataShape, object)
def validate(schema, value):
    if len(schema) == 1:
        return validate(schema[0], value)


@dispatch(Record, dict)
def validate(schema, d):
    return all(validate(sch, d.get(k)) for k, sch in schema.parameters[0])


@dispatch(String, str)
def validate(schema, value):
    return True


@dispatch(Record, (tuple, list))
def validate(schema, seq):
    return all(validate(sch, item) for (k, sch), item
                                    in zip(schema.parameters[0], seq))


@dispatch(str, object)
def validate(schema, value):
    return validate(dshape(schema), value)


@dispatch(type, object)
def validate(schema, value):
    return isinstance(value, schema)


@dispatch(tuple, object)
def validate(schemas, value):
    return any(validate(schema, value) for schema in schemas)


@dispatch(object, object)
def validate(schema, value):
    return False


@dispatch(Time, time)
def validate(schema, value):
    return True


@dispatch(Date, date)
def validate(schema, value):
    return True


@dispatch(DateTime, datetime)
def validate(schema, value):
    return True


@dispatch(DataShape, np.ndarray)
def validate(schema, value):
    return issubschema(from_numpy(value.shape, value.dtype), schema)


@dispatch(object, object)
def issubschema(a, b):
    return issubschema(dshape(a), dshape(b))


@dispatch(DataShape, DataShape)
def issubschema(a, b):
    if a == b:
        return True
    # TODO, handle cases like float < real
    # TODO, handle records {x: int, y: int, z: int} < {x: int, y: int}

    return None  # We don't know, return something falsey


def subset_dshape(index, ds):
    """ The DataShape of an indexed subarray

    >>> print(subset_dshape(0, 'var * {name: string, amount: int32}'))
    { name : string, amount : int32 }

    >>> print(subset_dshape(slice(0, 3), 'var * {name: string, amount: int32}'))
    3 * { name : string, amount : int32 }

    >>> print(subset_dshape('x', '{x: int, y: int}'))
    int32

    >>> print(subset_dshape((slice(0, 7, 2), 'amount'),
    ...                     'var * {name: string, amount: int32}'))
    3 * int32

    >>> print(subset_dshape((slice(0, 5), slice(0, 3), 5),
    ...                     '10 * var * 10 * int32'))
    5 * 3 * int32

    >>> print(subset_dshape(0, '{name: string, amount: int}'))
    string
    """

    if isinstance(ds, str):
        return subset_dshape(index, dshape(ds))
    if isinstance(index, int) and isdimension(ds[0]):
        return dshape(ds.subarray(1))
    if isinstance(ds[0], Record) and isinstance(index, str):
        return ds[0][index]
    if isinstance(ds[0], Record) and isinstance(index, int):
        return ds[0].parameters[0][index][1]
    if isinstance(index, slice) and isdimension(ds[0]):
        if None in (index.stop, index.start):
            return var * ds.subarray(1)
        count = index.stop - index.start
        if index.step is not None:
            count //= index.step
        return count * ds.subarray(1)
    if isinstance(index, tuple):
        if len(index) == 1:
            return subset_dshape(index[0], ds)
        else:
            ds2 = subset_dshape(index[1:], dshape(ds.subarray(1)))
            return subset_dshape(index[0], ds[0] * ds2)
    raise NotImplementedError()
