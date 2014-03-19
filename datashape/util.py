# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import inspect
import operator
import ctypes
import collections
import string
import sys
from functools import partial

from . import py2help
from . import parser
from . import type_symbol_table
from .error import UnificationError
from .validation import validate
from . import coretypes
from .typesets import TypeSet


__all__ = ['dshape', 'dshapes', 'has_var_dim', 'has_ellipsis',
           'cat_dshapes', 'dummy_signature', 'verify',
           'from_ctypes', 'from_cffi', 'to_ctypes',
           'gensym']


PY3 = (sys.version_info[:2] >= (3,0))

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

    """
    if isinstance(o, py2help._strtypes):
        ds = parser.parse(o, type_symbol_table.sym)
    elif isinstance(o, (coretypes.CType, coretypes.String,
                        coretypes.Record, coretypes.JSON)):
        ds = coretypes.DataShape(o)
    elif isinstance(o, coretypes.Mono):
        ds = o
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


def has_var_dim(ds):
    """Returns True if datashape has a variable dimension

    Note currently treats variable length string as scalars.
    """
    test = []
    if isinstance(ds, (coretypes.Ellipsis, coretypes.Var)):
        return True
    elif isinstance(ds, coretypes.Record):
        test = ds.types
    elif isinstance(ds, coretypes.Mono):
        test = ds.parameters
    elif isinstance(ds, (list, tuple)):
        test = ds
    for ds_t in test:
        if has_var_dim(ds_t):
            return True
    return False


def has_ellipsis(ds):
    """Returns True if the datashape has an ellipsis
    """
    test = []
    if isinstance(ds, coretypes.Ellipsis):
        return True
    elif isinstance(ds, coretypes.Record):
        test = ds.types
    elif isinstance(ds, coretypes.Mono):
        test = ds.parameters
    elif isinstance(ds, (list, tuple)):
        test = ds
    for ds_t in test:
        if has_ellipsis(ds_t):
            return True
    return False


def dummy_signature(f):
    """Create a dummy signature for `f`"""
    from . import coretypes as T
    argspec = inspect.getargspec(f)
    n = len(argspec.args)
    return T.Function(*[T.TypeVar(gensym()) for i in range(n + 1)])


def verify(t1, t2):
    """Verify that two immediate type constructors are valid for unification"""
    if not isinstance(t1, coretypes.Mono) or not isinstance(t2, coretypes.Mono):
        if t1 != t2:
            raise UnificationError("%s != %s" % (t1, t2))
        return

    args1, args2 = t1.parameters, t2.parameters
    tcon1, tcon2 = type_constructor(t1), type_constructor(t2)

    if tcon1 != tcon2:
        raise UnificationError(
            "Got differing type constructors %s and %s" % (tcon1, tcon2))

    if len(args1) != len(args2):
        raise UnificationError("%s got %d and %d arguments" % (
            tcon1, len(args1), len(args2)))


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
        return Record([(f[0], _from_cffi_internal(ffi, f[1].type))
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
        if cn in ['signed char', 'short', 'int',
                        'long', 'long long']:
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

def to_ctypes(dshape):
    """
    Constructs a ctypes type from a datashape
    """
    if len(dshape) == 1:
        if dshape == coretypes.int8:
            return ctypes.c_int8
        elif dshape == coretypes.int16:
            return ctypes.c_int16
        elif dshape == coretypes.int32:
            return ctypes.c_int32
        elif dshape == coretypes.int64:
            return ctypes.c_int64
        elif dshape == coretypes.uint8:
            return ctypes.c_uint8
        elif dshape == coretypes.uint16:
            return ctypes.c_uint16
        elif dshape == coretypes.uint32:
            return ctypes.c_uint32
        elif dshape == coretypes.uint64:
            return ctypes.c_uint64
        elif dshape == coretypes.float32:
            return ctypes.c_float
        elif dshape == coretypes.float64:
            return ctypes.c_double
        elif dshape == coretypes.complex_float32:
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
            class temp(ctypes.Structure):
                _fields_ = fields
            return temp
        else:
            raise TypeError("Cannot convert datashape %r into ctype" % dshape)
    # Create arrays
    else:
        if isinstance(dshape[0], (coretypes.TypeVar, coretypes.Ellipsis)):
            num = 0
        else:
            num = int(dshape[0])
        return num*to_ctypes(dshape.subarray(1))


# FIXME: Add a field
def from_ctypes(ctype):
    """
    Constructs a blaze dshape from a ctypes type.
    """
    if issubclass(ctype, ctypes.Structure):
        fields = []
        if hasattr(ctype, '_blaze_type_'):
            return ctype._blaze_type_
        for nm, tp in ctype._fields_:
            child_ds = from_ctypes(tp)
            fields.append((nm, child_ds))
        ds = Record(fields)
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
    elif ctype == ctypes.c_int8:
        return coretypes.int8
    elif ctype == ctypes.c_int16:
        return coretypes.int16
    elif ctype == ctypes.c_int32:
        return coretypes.int32
    elif ctype == ctypes.c_int64:
        return coretypes.int64
    elif ctype == ctypes.c_uint8:
        return coretypes.uint8
    elif ctype == ctypes.c_uint16:
        return coretypes.uint16
    elif ctype == ctypes.c_uint32:
        return coretypes.uint32
    elif ctype == ctypes.c_uint64:
        return coretypes.uint64
    elif ctype == ctypes.c_float:
        return coretypes.float32
    elif ctype == ctypes.c_double:
        return coretypes.float64
    else:
        raise TypeError('Cannot convert ctypes %r into '
                        'a blaze datashape' % ctype)

# Class to hold Pointer temporarily
def _PointerDshape(object):
    def __init__(self, dshape):
        self.dshape = dshape



#------------------------------------------------------------------------
# Temporary names
#------------------------------------------------------------------------

def make_temper():
    """Return a function that returns temporary names"""
    temps = collections.defaultdict(int)

    def temper(name=""):
        varname = name.rstrip(string.digits)
        count = temps[varname]
        temps[varname] += 1
        if varname and count == 0:
            return varname
        return varname + str(count)

    return temper

def make_stream(seq, _temp=make_temper()):
    """Create a stream of temporaries seeded by seq"""
    while 1:
        for x in seq:
            yield _temp(x)

gensym = partial(next, make_stream(string.ascii_uppercase))
