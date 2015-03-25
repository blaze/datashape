# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

"""
This defines the DataShape type system, with unified
shape and data type.
"""

import ctypes
import operator
from math import ceil

import numpy as np

from .py2help import _inttypes, _strtypes, unicode
from .internal_utils import IndexCallable


# Classes of unit types.
DIMENSION = 1
MEASURE = 2


class Type(type):
    _registry = {}

    def __new__(meta, name, bases, dct):
        cls = type(name, bases, dct)
        # Don't register abstract classes
        if not dct.get('abstract'):
            Type._registry[name] = cls
        return cls

    @classmethod
    def register(cls, name, type):
        # Don't clobber existing types.
        if name in cls._registry:
            raise TypeError('There is another type registered with name %s'
                            % name)

        cls._registry[name] = type

    @classmethod
    def lookup_type(cls, name):
        return cls._registry[name]


class Mono(object):
    """
    Monotype are unqualified 0 parameters.

    Each type must be reconstructable using its parameters:

        type(datashape_type)(*type.parameters)
    """

    composite = False
    __metaclass__ = Type

    def __init__(self, *params):
        self._parameters = params

    @property
    def parameters(self):
        if hasattr(self, '__slots__'):
            return tuple([getattr(self, slot) for slot in self.__slots__])
        else:
            return self._parameters

    def info(self):
        return type(self), self.parameters

    def __eq__(self, other):
        return type(self) == type(other) and self.info() == other.info()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.info())

    @property
    def shape(self):
        return ()

    def __len__(self):
        return 1

    def __getitem__(self, key):
        lst = [self]
        return lst[key]

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__,
                           ", ".join(map(repr, self.parameters)))

    # Form for searching signature in meta-method Dispatch Table
    def sigform(self):
        return self

    # Monotypes are their own measure
    @property
    def measure(self):
        return self

    def subarray(self, leading):
        """Returns a data shape object of the subarray with 'leading'
        dimensions removed. In the case of a measure such as CType,
        'leading' must be 0, and self is returned.
        """
        if leading >= 1:
            raise IndexError(('Not enough dimensions in data shape '
                            'to remove %d leading dimensions.') % leading)
        else:
            return self

    def __mul__(self, other):
        if isinstance(other, _strtypes):
            import datashape
            return datashape.dshape(other).__rmul__(self)
        if isinstance(other, _inttypes):
            other = Fixed(other)
        if isinstance(other, DataShape):
            return other.__rmul__(self)

        return DataShape(self, other)

    def __rmul__(self, other):
        if isinstance(other, _strtypes):
            import datashape
            return self * datashape.dshape(other)
        if isinstance(other, _inttypes):
            other = Fixed(other)

        return DataShape(other, self)

    def __getstate__(self):
        return self.parameters

    def __setstate__(self, state):
        if hasattr(self, '__slots__'):
            for slot, val in zip(self.__slots__, state):
                setattr(self, slot, val)
        else:
            self._parameters = state


class Unit(Mono):
    """
    Unit type that does not need to be reconstructed.
    """


class Ellipsis(Mono):
    """
    Ellipsis (...). Used to indicate a variable number of dimensions.
    E.g.:

        ... * float32    # float32 array w/ any number of dimensions
        A... * float32   # float32 array w/ any number of dimensions,
                        # associated with type variable A
    """
    __slots__ = 'typevar',

    def __init__(self, typevar=None):
        self.typevar = typevar

    def __str__(self):
        if self.typevar:
            return str(self.typevar) + '...'
        return '...'

    def __repr__(self):
        return 'Ellipsis("%s")' % (str(self),)


class Null(Unit):
    """
    The null datashape.
    """
    def __str__(self):
        return expr_string('null', None)


class IntegerConstant(Unit):
    """
    An integer which is a parameter to a type constructor. It is itself a
    degenerate type constructor taking 0 parameters.

    ::
        1, int32   # 1 is Fixed

    """
    __slots__ = 'val',
    cls = None

    def __init__(self, i):
        assert isinstance(i, _inttypes)
        self.val = i

    def __str__(self):
        return str(self.val)

    def __eq__(self, other):
        if isinstance(other, _inttypes):
            return self.val == other
        elif isinstance(other, IntegerConstant):
            return self.val == other.val
        else:
            raise TypeError("Cannot compare type %r to type %r" %
                            (type(self).__name__, type(other).__name__))

    def __hash__(self):
        return hash(self.val)


class StringConstant(Unit):
    """
    Strings at the level of the constructor.

    ::
        string(3, "utf-8")   # "utf-8" is StringConstant
    """
    __slots__ = 'val',

    def __init__(self, i):
        assert isinstance(i, _strtypes)
        self.val = i

    def __str__(self):
        return repr(self.val)

    def __eq__(self, other):
        if isinstance(other, _strtypes):
            return self.val == other
        elif isinstance(other, StringConstant):
            return self.val == other.val
        else:
            raise TypeError("Cannot compare type %r to type %r" %
                            (type(self).__name__, type(other).__name__))

    def __hash__(self):
        return hash(self.val)


class Date(Unit):
    """ Date type """
    cls = MEASURE
    __slots__ = ()

    def __str__(self):
        return 'date'

    def to_numpy_dtype(self):
        return np.dtype('datetime64[D]')


class Time(Unit):
    """ Time type """
    cls = MEASURE
    __slots__ = 'tz',

    def __init__(self, tz=None):
        if tz is not None and not isinstance(tz, _strtypes):
            raise ValueError('tz parameter to time datashape must be a string')
        # TODO validate against Olson tz database
        self.tz = tz

    def __str__(self):
        if self.tz is None:
            return 'time'
        else:
            return 'time[tz=%r]' % str(self.tz)


class DateTime(Unit):
    """ DateTime type """
    cls = MEASURE
    __slots__ = 'tz',

    def __init__(self, tz=None):
        if tz is not None and not isinstance(tz, _strtypes):
            raise ValueError('tz parameter to datetime datashape ' +
                             'must be a string')
        # TODO validate against Olson tz database
        self.tz = tz

    def __str__(self):
        if self.tz is None:
            return 'datetime'
        else:
            return 'datetime[tz=%r]' % str(self.tz)

    def to_numpy_dtype(self):
        return np.dtype('datetime64[us]')


_units = set(['ns', 'us', 'ms', 's', 'm', 'h', 'D', 'W', 'M', 'Y'])


_unit_aliases = {'year': 'Y', 'week': 'W', 'day': 'D', 'date': 'D', 'hour':
                 'h', 'second': 's', 'millisecond': 'ms', 'microsecond':
                 'us', 'nanosecond': 'ns'}


def normalize_time_unit(s):
    """ Normalize time input to one of 'year', 'second', 'millisecond', etc..
    Example
    -------
    >>> normalize_time_unit('milliseconds')
    'ms'
    >>> normalize_time_unit('ms')
    'ms'
    >>> normalize_time_unit('nanoseconds')
    'ns'
    >>> normalize_time_unit('nanosecond')
    'ns'
    """
    s = s.strip()
    if s in _units:
        return s
    if s in _unit_aliases:
        return _unit_aliases[s]
    if s[-1] == 's' and len(s) > 2:
        return normalize_time_unit(s.rstrip('s'))

    raise ValueError("Do not understand time unit %s" % s)


class TimeDelta(Unit):
    cls = MEASURE
    __slots__ = 'unit',

    def __init__(self, unit='us'):
        self.unit = normalize_time_unit(str(unit))

    def __str__(self):
        return 'timedelta[unit=%r]' % self.unit

    def __repr__(self):
        return '%s(unit=%r)' % (type(self).__name__, self.unit)

    def to_numpy_dtype(self):
        return np.dtype('timedelta64[%s]' % self.unit)


class Units(Unit):
    """ Units type for values with physical units """
    cls = MEASURE
    __slots__ = 'unit', 'tp'

    def __init__(self, unit, tp=None):
        if not isinstance(unit, _strtypes):
            raise ValueError('unit parameter to units datashape ' +
                             'must be a string')
        if tp is None:
            tp = DataShape(float64)
        elif not isinstance(tp, DataShape):
            raise ValueError('tp parameter to units datashape ' +
                             'must be a datashape type')
        self.unit = unit
        self.tp = tp

    def __str__(self):
        if self.tp == DataShape(float64):
            return 'units[%r]' % (self.unit)
        else:
            return 'units[%r, %s]' % (self.unit, self.tp)


class Bytes(Unit):
    """ Bytes type """
    cls = MEASURE
    __slots__ = ()

    def __str__(self):
        return 'bytes'


_canonical_string_encodings = {
    u'A' : u'A',
    u'ascii' : u'A',
    u'U8' : u'U8',
    u'utf-8' : u'U8',
    u'utf_8' : u'U8',
    u'utf8' : u'U8',
    u'U16' : u'U16',
    u'utf-16' : u'U16',
    u'utf_16' : u'U16',
    u'utf16' : u'U16',
    u'U32' : u'U32',
    u'utf-32' : u'U32',
    u'utf_32' : u'U32',
    u'utf32' : u'U32'
}


class String(Unit):
    """ String container

    >>> String()
    ctype("string")
    >>> String(10, 'ascii')
    ctype("string[10, 'A']")
    """
    cls = MEASURE
    __slots__ = 'fixlen', 'encoding'

    def __init__(self, *args):
        if len(args) == 0:
            fixlen, encoding = None, None
        if len(args) == 1:
            if isinstance(args[0], _strtypes):
                fixlen, encoding = None, args[0]
            if isinstance(args[0], _inttypes + (IntegerConstant,)):
                fixlen, encoding = args[0], None
        if len(args) == 2:
            fixlen, encoding = args

        encoding = encoding or 'U8'
        if isinstance(encoding, str):
            encoding = unicode(encoding)
        try:
            encoding = _canonical_string_encodings[encoding]
        except KeyError:
            raise ValueError('Unsupported string encoding %s' %
                             repr(encoding))

        if isinstance(fixlen, IntegerConstant):
            fixlen = fixlen.val

        self.encoding = encoding
        self.fixlen = fixlen

        # Put it in a canonical form

    def __str__(self):
        if self.fixlen is None and self.encoding == 'U8':
            return 'string'
        elif self.fixlen is not None and self.encoding == 'U8':
            return 'string[%i]' % self.fixlen
        elif self.fixlen is None and self.encoding != 'U8':
            return 'string[%s]' % repr(self.encoding).strip('u')
        else:
            return 'string[%i, %s]' % (self.fixlen,
                                       repr(self.encoding).strip('u'))

    def __repr__(self):
        s = str(self)
        return 'ctype("%s")' % s.encode('unicode_escape').decode('ascii')

    def to_numpy_dtype(self):
        """

        >>> String().to_numpy_dtype()
        dtype('O')
        >>> String(30).to_numpy_dtype()
        dtype('<U30')
        >>> String(30, 'A').to_numpy_dtype()
        dtype('S30')
        """
        if self.fixlen:
            if self.encoding == 'A':
                return np.dtype('S%d' % self.fixlen)
            else:
                return np.dtype('U%d' % self.fixlen)

        from .py2help import unicode
        # Create a dtype with metadata indicating it's
        # a string in the same style as the h5py special_dtype
        return np.dtype('O', metadata={'vlen': unicode})


class DataShape(Mono):
    """
    Composite container for datashape elements.

    Elements of a datashape like ``Fixed(3)``, ``Var()`` or ``int32`` are on,
    on their own, valid datashapes.  These elements are collected together into
    a composite ``DataShape`` to be complete.

    This class is not intended to be used directly.  Instead, use the utility
    ``dshape`` function to create datashapes from strings or datashape
    elements.

    Examples
    --------

    >>> from datashape import Fixed, int32, DataShape, dshape

    >>> DataShape(Fixed(5), int32)  # Rare to DataShape directly
    dshape("5 * int32")

    >>> dshape('5 * int32')         # Instead use the dshape function
    dshape("5 * int32")

    >>> dshape([Fixed(5), int32])   # It can even do construction from elements
    dshape("5 * int32")

    See Also
    --------
    datashape.dshape
    """

    __metaclass__ = Type
    composite = False

    def __init__(self, *parameters, **kwds):
        if len(parameters) == 1 and isinstance(parameters[0], _strtypes):
            raise TypeError("DataShape constructor for internal use.\n"
                            "Use dshape function to convert strings into "
                            "datashapes.\nTry:\n\tdshape('%s')"
                            % parameters[0])
        if len(parameters) > 0:
            self._parameters = tuple(map(_launder, parameters))
            if getattr(self._parameters[-1], 'cls', MEASURE) != MEASURE:
                raise TypeError(('Only a measure can appear on the'
                                ' last position of a datashape, not %s') %
                                repr(self._parameters[-1]))
            for dim in self._parameters[:-1]:
                if getattr(dim, 'cls', DIMENSION) != DIMENSION:
                    raise TypeError(('Only dimensions can appear before the'
                                    ' last position of a datashape, not %s') %
                                    repr(dim))
        else:
            raise ValueError('the data shape should be constructed from 2 or'
                             ' more parameters, only got %s' % len(parameters))
        self.composite = True

        name = kwds.get('name')
        if name:
            self.name = name
            self.__metaclass__._registry[name] = self
        else:
            self.name = None

        ###
        # TODO: Why are low-level concepts like strides and alignment on
        # TODO: the datashape?
        ###

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, index):
        return self.parameters[index]

    def __str__(self):
        if self.name:
            res = self.name
        else:
            res = ' * '.join(map(str, self.parameters))

        return res

    def __repr__(self):
        s = pprint(self)
        if '\n' in s:
            return 'dshape("""%s""")' % s
        else:
            return 'dshape("%s")' % s

    @property
    def shape(self):
        return self.parameters[:-1]

    @property
    def measure(self):
        return self.parameters[-1]

    def sigform(self):
        """Return a data shape object with Fixed dimensions replaced
        by TypeVar dimensions.
        """
        newparams = [TypeVar('i%d' % n)
                     for n in range(len(self.parameters) - 1)]
        newparams.append(self.parameters[-1])
        return DataShape(*newparams)

    def subarray(self, leading):
        """Returns a data shape object of the subarray with 'leading'
        dimensions removed.

        >>> from datashape import dshape
        >>> dshape('1 * 2 * 3 * int32').subarray(1)
        dshape("2 * 3 * int32")
        >>> dshape('1 * 2 * 3 * int32').subarray(2)
        dshape("3 * int32")
        """
        if leading >= len(self.parameters):
            raise IndexError('Not enough dimensions in data shape '
                             'to remove %d leading dimensions.' % leading)
        elif leading in [len(self.parameters) - 1, -1]:
            return DataShape(self.parameters[-1])
        else:
            return DataShape(*self.parameters[leading:])

    def __rmul__(self, other):
        if isinstance(other, _inttypes):
            other = Fixed(other)
        return DataShape(other, *self)

    @property
    def subshape(self):
        return IndexCallable(self._subshape)

    def _subshape(self, index):
        """ The DataShape of an indexed subarray

        >>> from datashape import dshape

        >>> ds = dshape('var * {name: string, amount: int32}')
        >>> print(ds.subshape[0])
        {name: string, amount: int32}

        >>> print(ds.subshape[0:3])
        3 * {name: string, amount: int32}

        >>> print(ds.subshape[0:7:2, 'amount'])
        4 * int32

        >>> print(ds.subshape[[1, 10, 15]])
        3 * {name: string, amount: int32}

        >>> ds = dshape('{x: int, y: int}')
        >>> print(ds.subshape['x'])
        int32

        >>> ds = dshape('10 * var * 10 * int32')
        >>> print(ds.subshape[0:5, 0:3, 5])
        5 * 3 * int32

        >>> ds = dshape('var * {name: string, amount: int32, id: int32}')
        >>> print(ds.subshape[:, [0, 2]])
        var * {name: string, id: int32}

        >>> ds = dshape('var * {name: string, amount: int32, id: int32}')
        >>> print(ds.subshape[:, ['name', 'id']])
        var * {name: string, id: int32}

        >>> print(ds.subshape[0, 1:])
        {amount: int32, id: int32}
        """
        from .predicates import isdimension
        if isinstance(index, _inttypes) and isdimension(self[0]):
            return self.subarray(1)
        if isinstance(self[0], Record) and isinstance(index, _strtypes):
            return self[0][index]
        if isinstance(self[0], Record) and isinstance(index, _inttypes):
            return self[0].parameters[0][index][1]
        if isinstance(self[0], Record) and isinstance(index, list):
            rec = self[0]
            # Translate strings to corresponding integers
            index = [self[0].names.index(i) if isinstance(i, _strtypes) else i
                     for i in index]
            return DataShape(Record([rec.parameters[0][i] for i in index]))
        if isinstance(self[0], Record) and isinstance(index, slice):
            rec = self[0]
            return DataShape(Record(rec.parameters[0][index]))
        if isinstance(index, list) and isdimension(self[0]):
            return len(index) * self.subarray(1)
        if isinstance(index, slice):
            if isinstance(self[0], Fixed):
                n = int(self[0])
                start = index.start or 0
                stop = index.stop or n
                if start < 0:
                    start = n + start
                if stop < 0:
                    stop = n + stop
                count = stop - start
            else:
                start = index.start or 0
                stop = index.stop
                if not stop:
                    if start < 0:
                        count = -start
                    else:
                        count = var
                if (stop is not None and
                    start is not None and
                    (stop >= 0) == (start >= 0)):
                    count = stop - start
                else:
                    count = var

            if count != var and index.step is not None:
                count = int(ceil(count / index.step))

            return count * self.subarray(1)
        if isinstance(index, tuple):
            if not index:
                return self
            elif index[0] is None:
                return 1 * self._subshape(index[1:])
            elif len(index) == 1:
                return self._subshape(index[0])
            else:
                ds = self.subarray(1)._subshape(index[1:])
                return (self[0] * ds)._subshape(index[0])
        raise NotImplementedError()

    def __setstate__(self, state):
        self.__init__(*state)


class Option(Mono):
    """
    Measure types which may or may not hold data. Makes no
    indication of how this is implemented in memory.
    """
    __slots__ = 'ty',

    def __init__(self, ds):
        self.ty = _launder(ds)

    @property
    def shape(self):
        return self.ty.shape

    @property
    def itemsize(self):
        return self.ty.itemsize

    def __str__(self):
        return '?%s' % str(self.ty)

    __repr__ = __str__


class CType(Unit):
    """
    Symbol for a sized type mapping uniquely to a native type.
    """
    cls = MEASURE
    __slots__ = 'name', '_itemsize', '_alignment'

    def __init__(self, name, itemsize, alignment):
        self.name = name
        self._itemsize = itemsize
        self._alignment = alignment
        Type.register(name, self)

    @classmethod
    def from_numpy_dtype(self, dt):
        """
        From Numpy dtype.

        >>> from datashape import CType
        >>> from numpy import dtype
        >>> CType.from_numpy_dtype(dtype('int32'))
        ctype("int32")
        >>> CType.from_numpy_dtype(dtype('i8'))
        ctype("int64")
        >>> CType.from_numpy_dtype(dtype('M8'))
        DateTime(None)
        >>> CType.from_numpy_dtype(dtype('U30'))
        ctype("string[30, 'U32']")
        """
        try:
            return Type.lookup_type(dt.name)
        except KeyError:
            pass
        if np.issubdtype(dt, np.datetime64):
            unit, _ = np.datetime_data(dt)
            defaults = {'D': date_, 'Y': date_, 'M': date_, 'W': date_}
            return defaults.get(unit, datetime_)
        elif np.issubdtype(dt, np.timedelta64):
            unit, _ = np.datetime_data(dt)
            return TimeDelta(unit=unit)
        elif np.issubdtype(dt, np.unicode_):
            return String(dt.itemsize // 4, 'U32')
        elif np.issubdtype(dt, np.str_) or np.issubdtype(dt, np.bytes_):
            return String(dt.itemsize, 'ascii')
        raise NotImplementedError("NumPy datatype %s not supported" % dt)

    @property
    def itemsize(self):
        """The size of one element of this type."""
        return self._itemsize

    @property
    def c_itemsize(self):
        """The size of one element of this type, with C-contiguous storage."""
        return self._itemsize

    @property
    def c_alignment(self):
        """The alignment of one element of this type."""
        return self._alignment

    def to_numpy_dtype(self):
        """
        To Numpy dtype.
        """
        # Fixup the complex type to how numpy does it
        s = self.name
        s = {'complex[float32]': 'complex64',
             'complex[float64]': 'complex128'}.get(s, s)
        return np.dtype(s)

    def __str__(self):
        return self.name

    def __repr__(self):
        s = str(self)
        return 'ctype("%s")' % s.encode('unicode_escape').decode('ascii')


class Fixed(Unit):
    """
    Fixed dimension.
    """
    cls = DIMENSION
    __slots__ = 'val',

    def __init__(self, i):
        # Use operator.index, so Python integers, numpy int scalars, etc work
        i = operator.index(i)

        if i < 0:
            raise ValueError('Fixed dimensions must be positive')

        self.val = i

    def __index__(self):
        return self.val

    def __int__(self):
        return self.val

    def __eq__(self, other):
        if type(other) is Fixed:
            return self.val == other.val
        elif isinstance(other, _inttypes):
            return self.val == other
        else:
            return False

    __hash__ = Mono.__hash__

    def __str__(self):
        return str(self.val)


class Var(Unit):
    """ Variable dimension """
    cls = DIMENSION
    __slots__ = ()

    def __str__(self):
        return 'var'


class TypeVar(Unit):
    """
    A free variable in the signature. Not user facing.
    """
    # cls could be MEASURE or DIMENSION, depending on context
    __slots__ = 'symbol',

    def __init__(self, symbol):
        if not symbol[0].isupper():
            raise ValueError(('TypeVar symbol %r does not ' +
                              'begin with a capital') % symbol)
        self.symbol = symbol

    def __repr__(self):
        return "TypeVar(%s)" % self

    def __str__(self):
        return str(self.symbol)


class Implements(Mono):
    """
    Type representing a constraint on the subtype term (which must be a
    TypeVar), namely that it must belong to a given type set.
    """

    @property
    def typevar(self):
        return self.parameters[0]

    @property
    def typeset(self):
        return self.parameters[1]

    def __repr__(self):
        return '%s: %s' % (self.typevar, self.typeset.name)


class Function(Mono):
    """
    Used for function signatures.
    """
    def __init__(self, *parameters):
        self._parameters = parameters

    @property
    def restype(self):
        return self.parameters[-1]

    @property
    def argtypes(self):
        return self.parameters[:-1]

    # def __repr__(self):
    #     return " -> ".join(map(repr, self.parameters))

    def __str__(self):
        return '(%s) -> %s' % (', '.join(map(str, self.parameters[:-1])),
                               self.parameters[-1])


def _launder(x):
    """ Clean up types prior to insertion into DataShape

    >>> from datashape import dshape
    >>> _launder(5)         # convert ints to Fixed
    Fixed(5)
    >>> _launder('int32')   # parse strings
    ctype("int32")
    >>> _launder(dshape('int32'))
    ctype("int32")
    >>> _launder(Fixed(5))  # No-op on valid parameters
    Fixed(5)
    """
    if isinstance(x, _inttypes):
        x = Fixed(x)
    if isinstance(x, _strtypes):
        return Type.lookup_type(x)
    if isinstance(x, DataShape) and len(x) == 1:
        return x[0]
    return x


def _launder_key(k):
    """

    >>> _launder_key('Hello')
    'Hello'
    >>> _launder_key(0)
    '0'
    """
    if isinstance(k, _strtypes):
        return k
    return str(k)


class CollectionPrinter(object):
    def __repr__(self):
        s = str(self)
        strs = ('"""%s"""' if '\n' in s else '"%s"') % s
        return 'dshape(%s)' % strs


class Record(CollectionPrinter, Mono):
    """
    A composite data structure of ordered fields mapped to types.

    Properties
    ----------

    fields: tuple of (name, type) pairs
        The only stored data, also the input to ``__init__``
    dict: dict
        A dictionary view of ``fields``
    names: list of strings
        A list of the names
    types: list of datashapes
        A list of the datashapes

    Example
    -------

    >>> Record([['id', 'int'], ['name', 'string'], ['amount', 'real']])
    dshape("{id: int32, name: string, amount: float64}")
    """
    cls = MEASURE

    def __init__(self, fields):
        """
        Parameters
        ----------
        fields : list/OrderedDict of (name, type) entries
            The fields which make up the record.
        """
        # This is passed in with a OrderedDict so field order is
        # preserved. Using RecordDecl there is some magic to also
        # ensure that the fields align in the order they are
        # declared.
        fields = tuple((_launder_key(k), _launder(v)) for k, v in fields)
        self._parameters = (tuple(map(tuple, fields)),)

    @property
    def fields(self):
        return self._parameters[0]

    @property
    def dict(self):
        return dict(self.fields)

    @property
    def names(self):
        return [n for n, t in self.fields]

    @property
    def types(self):
        return [t for n, t in self.fields]

    def to_numpy_dtype(self):
        """
        To Numpy record dtype.
        """
        return np.dtype([(str(name), to_numpy_dtype(typ))
                         for name, typ in self.fields])

    def __getitem__(self, key):
        return self.dict[key]

    def __str__(self):
        return pprint(self)


class Tuple(CollectionPrinter, Mono):
    """
    A product type.
    """
    __slots__ = 'dshapes',
    cls = MEASURE

    def __init__(self, dshapes):
        """
        Parameters
        ----------
        dshapes : list of dshapes
            The datashapes which make up the tuple.
        """
        dshapes = [DataShape(ds) if not isinstance(ds, DataShape) else ds
                   for ds in dshapes]
        self.dshapes = tuple(dshapes)

    def __str__(self):
        return '(%s)' % ', '.join(map(str, self.dshapes))

    def to_numpy_dtype(self):
        """
        To Numpy record dtype.
        """
        return np.dtype([('f%d' % i, to_numpy_dtype(typ))
                         for i, typ in enumerate(self.parameters[0])])


class JSON(Mono):
    """ JSON measure """
    cls = MEASURE
    __slots__ = ()

    def __str__(self):
        return 'json'


bool_ = CType('bool', 1, 1)
char = CType('char', 1, 1)

int8 = CType('int8', 1, 1)
int16 = CType('int16', 2, ctypes.alignment(ctypes.c_int16))
int32 = CType('int32', 4, ctypes.alignment(ctypes.c_int32))
int64 = CType('int64', 8, ctypes.alignment(ctypes.c_int64))

# int is an alias for int32
int_ = int32
Type.register('int', int_)

uint8 = CType('uint8', 1, 1)
uint16 = CType('uint16', 2, ctypes.alignment(ctypes.c_uint16))
uint32 = CType('uint32', 4, ctypes.alignment(ctypes.c_uint32))
uint64 = CType('uint64', 8, ctypes.alignment(ctypes.c_uint64))

float16 = CType('float16', 2, ctypes.alignment(ctypes.c_uint16))
float32 = CType('float32', 4, ctypes.alignment(ctypes.c_float))
float64 = CType('float64', 8, ctypes.alignment(ctypes.c_double))
# float128 = CType('float128', 16)

# real is an alias for float64
real = float64
Type.register('real', real)

complex_float32 = CType('complex[float32]', 8,
                        ctypes.alignment(ctypes.c_float))
complex_float64 = CType('complex[float64]', 16,
                        ctypes.alignment(ctypes.c_double))
Type.register('complex64', complex_float32)
complex64 = complex_float32

Type.register('complex128', complex_float64)
complex128 = complex_float64
# complex256 = CType('complex256', 32)

# complex is an alias for complex[float64]
complex_ = complex_float64

date_ = Date()
time_ = Time()
datetime_ = DateTime()
timedelta_ = TimeDelta()
Type.register('date', date_)
Type.register('time', time_)
Type.register('datetime', datetime_)
Type.register('timedelta', timedelta_)

null = Null()
Type.register('null', null)

c_byte = int8
c_short = int16
c_int = int32
c_longlong = int64

c_ubyte = uint8
c_ushort = uint16
c_ulonglong = uint64

if ctypes.sizeof(ctypes.c_long) == 4:
    c_long = int32
    c_ulong = uint32
else:
    c_long = int64
    c_ulong = uint64

if ctypes.sizeof(ctypes.c_void_p) == 4:
    intptr = c_ssize_t = int32
    uintptr = c_size_t = uint32
else:
    intptr = c_ssize_t = int64
    uintptr = c_size_t = uint64
Type.register('intptr', intptr)
Type.register('uintptr', uintptr)

c_half = float16
c_float = float32
c_double = float64

# TODO: Deal with the longdouble == one of float64/float80/float96/float128
# situation

# c_longdouble = float128

half = float16
single = float32
double = float64

void = CType('void', 0, 1)
object_ = pyobj = CType('object',
                        ctypes.sizeof(ctypes.py_object),
                        ctypes.alignment(ctypes.py_object))

na = Null
NullRecord = Record(())
bytes_ = Bytes()

string = String()
json = JSON()

Type.register('float', c_float)
Type.register('double', c_double)

Type.register('bytes', bytes_)

Type.register('string', String())

var = Var()


class NotNumpyCompatible(Exception):
    """
    Raised when we try to convert a datashape into a NumPy dtype
    but it cannot be ceorced.
    """
    pass


def to_numpy_dtype(ds):
    """ Throw away the shape information and just return the
    measure as NumPy dtype instance."""
    return to_numpy(ds.measure)[1]


def to_numpy(ds):
    """
    Downcast a datashape object into a Numpy (shape, dtype) tuple if
    possible.

    >>> from datashape import dshape, to_numpy
    >>> to_numpy(dshape('5 * 5 * int32'))
    ((5, 5), dtype('int32'))
    >>> to_numpy(dshape('10 * string[30]'))
    ((10,), dtype('<U30'))
    """

    shape = tuple()
    dtype = None

    if isinstance(ds, DataShape):
        # The datashape dimensions
        for dim in ds[:-1]:
            if isinstance(dim, IntegerConstant):
                shape += (dim,)
            elif isinstance(dim, Fixed):
                shape += (dim.val,)
            elif isinstance(dim, TypeVar):
                shape += (-1,)
            else:
                raise NotNumpyCompatible('DataShape dimension %s is not'
                                         ' NumPy-compatible' % dim)

        # The datashape measure
        msr = ds[-1]
    else:
        msr = ds

    try:
        dtype = msr.to_numpy_dtype()
    except AttributeError:
        raise NotNumpyCompatible('DataShape measure %s is not NumPy-compatible'
                                 % msr)

    if type(dtype) != np.dtype:
        raise NotNumpyCompatible('Internal Error: Failed to produce NumPy '
                                 'dtype')
    return shape, dtype


def from_numpy(shape, dt):
    """
    Upcast a (shape, dtype) tuple if possible.

    >>> from datashape import from_numpy
    >>> from numpy import dtype
    >>> from_numpy((5, 5), dtype('int32'))
    dshape("5 * 5 * int32")

    >>> from_numpy((10,), dtype('S10'))
    dshape("10 * string[10, 'A']")
    """
    dtype = np.dtype(dt)

    if dtype.kind == 'S':
        measure = String(dtype.itemsize, 'A')
    elif dtype.kind == 'U':
        measure = String(dtype.itemsize // 4, 'U32')
    elif dtype.fields:
        fields = [(name, dtype.fields[name]) for name in dtype.names]
        rec = [(name, from_numpy(t.shape, t.base))  # recurse into nested dtype
               for name, (t, _) in fields]  # _ is the byte offset: ignore it
        measure = Record(rec)
    else:
        measure = CType.from_numpy_dtype(dtype)

    if not shape:
        return measure
    return DataShape(*tuple(map(Fixed, shape)) + (measure,))


def expr_string(spine, const_args, outer='()'):
    assert len(outer) == 2

    lhs, rhs = outer

    if const_args:
        return '%s%s%s%s' % (spine, lhs, ','.join(map(str, const_args)), rhs)
    return str(spine)


def free(ds):
    """
    Return the free variables (TypeVar) of a datashape type (Mono).
    """
    if isinstance(ds, TypeVar):
        return [ds]
    elif isinstance(ds, Mono) and not isinstance(ds, Unit):
        result = []
        for x in ds.parameters:
            result.extend(free(x))
        return result
    else:
        return []


def print_unicode_string(s):
    try:
        return s.decode('unicode_escape').encode('ascii')
    except AttributeError:
        return s


def pprint(ds, width=80):
    """ Pretty print a datashape

    >>> from datashape import dshape, pprint
    >>> print(pprint(dshape('5 * 3 * int32')))
    5 * 3 * int32

    >>> ds = dshape('5000000000 * {a: (int, float32, real, string, datetime), b: {c: 5 * int, d: var * 100 * float32}}')
    >>> print(pprint(ds))
    5000000000 * {
      a: (int32, float32, float64, string, datetime),
      b: {c: 5 * int32, d: var * 100 * float32}
      }

    Record measures print like full datashapes
    >>> print(pprint(ds.measure, width=30))
    {
      a: (
        int32,
        float32,
        float64,
        string,
        datetime
        ),
      b: {
        c: 5 * int32,
        d: var * 100 * float32
        }
      }

    Control width of the result
    >>> print(pprint(ds, width=30))
    5000000000 * {
      a: (
        int32,
        float32,
        float64,
        string,
        datetime
        ),
      b: {
        c: 5 * int32,
        d: var * 100 * float32
        }
      }
    >>>
    """
    result = ''

    if isinstance(ds, DataShape):
        if ds.shape:
            result += ' * '.join(map(str, ds.shape))
            result += ' * '
        ds = ds[-1]

    if isinstance(ds, Record):
        pairs = ['%s: %s' % (name if ' ' not in name else
                             repr(print_unicode_string(name)),
                             pprint(typ, width - len(result) - len(name)))
                 for name, typ in zip(ds.names, ds.types)]
        short = '{%s}' % ', '.join(pairs)
        if len(result + short) < width:
            return result + short
        else:
            long = '{\n%s\n}' % ',\n'.join(pairs)
            return result + long.replace('\n', '\n  ')

    elif isinstance(ds, Tuple):
        typs = [pprint(typ, width-len(result))
                for typ in ds.dshapes]
        short = '(%s)' % ', '.join(typs)
        if len(result + short) < width:
            return result + short
        else:
            long = '(\n%s\n)' % ',\n'.join(typs)
            return result + long.replace('\n', '\n  ')
    else:
        result += str(ds)
    return result
