# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import operator
import ctypes
import sys

from . import py2help
from . import parser
from . import type_symbol_table
from .validation import validate
from . import coretypes
from itertools import chain
from .internal_utils import reverse_dict


__all__ = ['dshape', 'dshapes', 'has_var_dim', 'has_ellipsis',
           'cat_dshapes', 'from_ctypes', 'from_cffi', 'to_ctypes']

subclasses = operator.methodcaller('__subclasses__')

PY3 = sys.version_info[:2] >= (3, 0)

#------------------------------------------------------------------------
# Utility Functions for DataShapes
#------------------------------------------------------------------------

def dshapes(*args):
    """
    Parse a bunch of datashapes all at once.

    >>> a, b = dshapes('3 * int32', '2 * var * float64')
    """
    return [dshape(arg) for arg in args]


def dshape(o):
    """
    Parse a blaze type. For a thorough description see
    http://blaze.pydata.org/docs/datashape.html

    >>> ds = dshape('2 * int32')
    >>> ds[1]
    ctype("int32")
    """
    if isinstance(o, coretypes.DataShape):
        return o
    if isinstance(o, py2help._strtypes):
        ds = parser.parse(o, type_symbol_table.sym)
    elif isinstance(o, (coretypes.CType, coretypes.String,
                        coretypes.Record, coretypes.JSON,
                        coretypes.Date, coretypes.Time, coretypes.DateTime,
                        coretypes.Unit)):
        ds = coretypes.DataShape(o)
    elif isinstance(o, coretypes.Mono):
        ds = o
    elif isinstance(o, (list, tuple)):
        ds = coretypes.DataShape(*o)
    else:
        raise TypeError('Cannot create dshape from object of type %s' % type(o))
    validate(ds)
    return ds


def cat_dshapes(dslist):
    """
    Concatenates a list of dshapes together along
    the first axis. Raises an error if there is
    a mismatch along another axis or the measures
    are different.

    Requires that the leading dimension be a known
    size for all data shapes.
    TODO: Relax this restriction to support
          streaming dimensions.

    >>> cat_dshapes(dshapes('10 * int32', '5 * int32'))
    dshape("15 * int32")
    """
    if len(dslist) == 0:
        raise ValueError('Cannot concatenate an empty list of dshapes')
    elif len(dslist) == 1:
        return dslist[0]

    outer_dim_size = operator.index(dslist[0][0])
    inner_ds = dslist[0][1:]
    for ds in dslist[1:]:
        outer_dim_size += operator.index(ds[0])
        if ds[1:] != inner_ds:
            raise ValueError(('The datashapes to concatenate much'
                              ' all match after'
                              ' the first dimension (%s vs %s)') %
                              (inner_ds, ds[1:]))
    return coretypes.DataShape(*[coretypes.Fixed(outer_dim_size)] + list(inner_ds))


def collect(pred, expr):
    """ Collect terms in expression that match predicate

    >>> from datashape import Unit, dshape
    >>> predicate = lambda term: isinstance(term, Unit)
    >>> dshape = dshape('var * {value: int64, loc: 2 * int32}')
    >>> sorted(set(collect(predicate, dshape)), key=str)
    [Fixed(2), ctype("int32"), ctype("int64"), Var()]
    """
    if pred(expr):
        return [expr]
    if isinstance(expr, coretypes.Record):
        return chain.from_iterable(collect(pred, typ) for typ in expr.types)
    if isinstance(expr, coretypes.Mono):
        return chain.from_iterable(collect(pred, typ) for typ in expr.parameters)
    if isinstance(expr, (list, tuple)):
        return chain.from_iterable(collect(pred, item) for item in expr)


def has_var_dim(ds):
    """Returns True if datashape has a variable dimension

    Note currently treats variable length string as scalars.

    >>> has_var_dim(dshape('2 * int32'))
    False
    >>> has_var_dim(dshape('var * 2 * int32'))
    True
    """
    return has((coretypes.Ellipsis, coretypes.Var), ds)


def has(typ, ds):
    if isinstance(ds, typ):
        return True
    if isinstance(ds, coretypes.Record):
        return any(has(typ, t) for t in ds.types)
    if isinstance(ds, coretypes.Mono):
        return any(has(typ, p) for p in ds.parameters)
    if isinstance(ds, (list, tuple)):
        return any(has(typ, item) for item in ds)
    return False


def has_ellipsis(ds):
    """Returns True if the datashape has an ellipsis

    >>> has_ellipsis(dshape('2 * int'))
    False
    >>> has_ellipsis(dshape('... * int'))
    True
    """
    return has(coretypes.Ellipsis, ds)


#------------------------------------------------------------------------
# DataShape Conversion
#------------------------------------------------------------------------
def _from_cffi_internal(ffi, ctype):
    k = ctype.kind
    if k == 'struct':
        # TODO: Assuming the field offsets match
        #       blaze kernels - need to sync up blaze, dynd,
        #       cffi, numpy, etc so that the field offsets always work!
        #       Also need to make sure there are no bitsize/bitshift
        #       values that would be incompatible.
        return coretypes.Record([(f[0], _from_cffi_internal(ffi, f[1].type))
                                 for f in ctype.fields])
    elif k == 'array':
        if ctype.length is None:
            # Only the first array can have the size
            # unspecified, so only need a single name
            dsparams = [coretypes.TypeVar('N')]
        else:
            dsparams = [coretypes.Fixed(ctype.length)]
        ctype = ctype.item
        while ctype.kind == 'array':
            dsparams.append(coretypes.Fixed(ctype.length))
            ctype = ctype.item
        dsparams.append(_from_cffi_internal(ffi, ctype))
        return coretypes.DataShape(*dsparams)
    elif k == 'primitive':
        cn = ctype.cname
        if cn in ['signed char', 'short', 'int', 'long', 'long long']:
            so = ffi.sizeof(ctype)
            if so == 1:
                return coretypes.int8
            elif so == 2:
                return coretypes.int16
            elif so == 4:
                return coretypes.int32
            elif so == 8:
                return coretypes.int64
            else:
                raise TypeError('cffi primitive "%s" has invalid size %d' %
                                (cn, so))
        elif cn in ['unsigned char', 'unsigned short',
                        'unsigned int', 'unsigned long',
                        'unsigned long long']:
            so = ffi.sizeof(ctype)
            if so == 1:
                return coretypes.uint8
            elif so == 2:
                return coretypes.uint16
            elif so == 4:
                return coretypes.uint32
            elif so == 8:
                return coretypes.uint64
            else:
                raise TypeError('cffi primitive "%s" has invalid size %d' %
                                (cn, so))
        elif cn == 'float':
            return coretypes.float32
        elif cn == 'double':
            return coretypes.float64
        else:
            raise TypeError('Unrecognized cffi primitive "%s"' % cn)
    elif k == 'pointer':
        raise TypeError('a pointer can only be at the outer level of a cffi type '
                        'when converting to blaze datashape')
    else:
        raise TypeError('Unrecognized cffi kind "%s"' % k)


def from_cffi(ffi, ctype):
    """
    Constructs a blaze dshape from a cffi type.
    """
    # Allow one pointer dereference at the outermost level
    if ctype.kind == 'pointer':
        ctype = ctype.item
    return _from_cffi_internal(ffi, ctype)

typedict = {
    ctypes.c_int8: coretypes.int8,
    ctypes.c_int16: coretypes.int16,
    ctypes.c_int32: coretypes.int32,
    ctypes.c_int64: coretypes.int64,
    ctypes.c_uint8: coretypes.uint8,
    ctypes.c_uint16: coretypes.uint16,
    ctypes.c_uint32: coretypes.uint32,
    ctypes.c_uint64: coretypes.uint64,
    ctypes.c_float: coretypes.float32,
    ctypes.c_double: coretypes.float64
}


revtypedict = reverse_dict(typedict)


def to_ctypes(dshape):
    """
    Constructs a ctypes type from a datashape

    >>> to_ctypes(coretypes.float64)  # doctest: +SKIP
    <class 'ctypes.c_double'>
    """
    if len(dshape) == 1:
        ctype = revtypedict.get(dshape)
        if ctype:
            return ctype
        if dshape == coretypes.complex_float32:
            class Complex64(ctypes.Structure):
                _fields_ = [('real', ctypes.c_float),
                            ('imag', ctypes.c_float)]
                _blaze_type_ = coretypes.complex_float32
            return Complex64
        elif dshape == coretypes.complex_float64:
            class Complex128(ctypes.Structure):
                _fields_ = [('real', ctypes.c_double),
                            ('imag', ctypes.c_double)]
                _blaze_type_ = coretypes.complex_float64
            return Complex128
        elif isinstance(dshape, coretypes.Record):
            fields = [(name, to_ctypes(dshape.fields[name]))
                      for name in dshape.names]

            # generate a class based on the fields and types of the record
            return type('temp', (ctypes.Structure,), {'_fields_': fields})
        else:
            raise TypeError("Cannot convert datashape %r into ctype" % dshape)
    # Create arrays
    else:
        if isinstance(dshape[0], (coretypes.TypeVar, coretypes.Ellipsis)):
            num = 0
        else:
            num = int(dshape[0])
        return num * to_ctypes(dshape.subarray(1))


# FIXME: Add a field
def from_ctypes(ctype):
    """
    Constructs a blaze dshape from a ctypes type.

    >>> from_ctypes(ctypes.c_int)
    ctype("int32")
    """
    if issubclass(ctype, ctypes.Structure):
        fields = []
        if hasattr(ctype, '_blaze_type_'):
            return ctype._blaze_type_
        for nm, tp in ctype._fields_:
            child_ds = from_ctypes(tp)
            fields.append((nm, child_ds))
        ds = coretypes.Record(fields)
        # TODO: Validate that the ctypes offsets match
        #       the C offsets blaze uses
        return ds
    elif issubclass(ctype, ctypes.Array):
        dstup = []
        while issubclass(ctype, ctypes.Array):
            dstup.append(coretypes.Fixed(ctype._length_))
            ctype = ctype._type_
        dstup.append(from_ctypes(ctype))
        return coretypes.DataShape(*dstup)

    coretype = typedict.get(ctype)

    if coretype:
        return coretype
    else:
        raise TypeError('Cannot convert ctypes %r into '
                        'a blaze datashape' % ctype)
