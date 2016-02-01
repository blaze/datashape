Datashape Overview
==================

Datashape is a data layout language for array programming. It is designed
to describe in-situ structured data without requiring transformation
into a canonical form.

Similar to NumPy, datashape includes ``shape`` and ``dtype``, but combined
together in the type system.

Units
-----

Single named types in datashape are called ``unit`` types. They represent
either a dtype like ``int32`` or ``datetime``, or a single dimension
like ``var``. Dimensions and a single dtype are composed together in
a datashape type.

Primitive Types
~~~~~~~~~~~~~~~

DataShape includes a variety of dtypes corresponding to C/C++
types, similar to NumPy.

.. cssclass:: table-striped

================ =========================================================
Bit type         Description
================ =========================================================
bool             Boolean (True or False) stored as a byte
int8             Byte (-128 to 127)
int16            Two's Complement Integer (-32768 to 32767)
int32            Two's Complement Integer (-2147483648 to 2147483647)
int64            Two's Complement Integer (-9223372036854775808 to 9223372036854775807)
uint8            Unsigned integer (0 to 255)
uint16           Unsigned integer (0 to 65535)
uint32           Unsigned integer (0 to 4294967295)
uint64           Unsigned integer (0 to 18446744073709551615)
float16          Half precision float: sign bit, 5 bits exponent,
                 10 bits mantissa
float32          Single precision float: sign bit, 8 bits exponent,
                 23 bits mantissa
float64          Double precision float: sign bit, 11 bits exponent,
                 52 bits mantissa
complex[float32] Complex number, represented by two 32-bit floats (real
                 and imaginary components)
complex[float64] Complex number, represented by two 64-bit floats (real
                 and imaginary components)
================ =========================================================

Additionally, there are types which are not fully specified at the
bit/byte level.

.. cssclass:: table-striped

==========  =========================================================
Bit type    Description
==========  =========================================================
string      Variable length Unicode string.
bytes       Variable length array of bytes.
json        Variable length Unicode string which contains JSON.
date        Date in the proleptic Gregorian calendar.
time        Time not attached to a date.
datetime    Point in time, combination of date and time.
units       Associates physical units with numerical values.
==========  =========================================================

Many python types can be mapped to datashape types:

.. cssclass:: table-striped

==================  =========================================================
Python type         Datashape
==================  =========================================================
int                 int32
bool                bool
float               float64
complex             complex[float64]
str                 string
unicode             string
datetime.date       date
datetime.time       time
datetime.datetime   datetime or datetime[tz='<timezone>']
datetime.timedelta  units['microsecond', int64]
bytes               bytes
bytearray           bytes
buffer              bytes
==================  =========================================================

String Types
~~~~~~~~~~~~

To Blaze, all strings are sequences of unicode code points, following
in the footsteps of Python 3. The default Blaze string atom, simply
called "string", is a variable-length string which can contain any
unicode values. There is also a fixed-size variant compatible with
NumPy's strings, like ``string[16, "ascii"]``.

Dimensions
----------

An asterisk (*) between two types signifies an array. A datashape
consists of 0 or more ``dimensions`` followed by a ``dtype``.

For example, an integer array of size three is::

    3 * int

In this type, 3 is is a ``fixed`` dimension, which means it is a dimension
whose size is always as given. Other dimension types include ``var``.

Comparing with NumPy, the array created by
``np.empty((2, 3), 'int32')`` has datashape ``2 * 3 * int32``.

Records
~~~~~~~

Record types are ordered struct dtypes which hold a collection of
types keyed by labels. Records look similar to Python
dictionaries but the order the names appear is important.

Example 1::

    {
        name   : string,
        age    : int,
        height : int,
        weight : int
    }

Example 2::

    {
        r: int8,
        g: int8,
        b: int8,
        a: int8
    }

Records are themselves types declaration so they can be nested,
but cannot be self-referential:

Example 2::

    {
        a: { x: int, y: int },
        b: { x: int, z: int }
    }

Datashape Traits
~~~~~~~~~~~~~~~~

While datashape is a very general type system, there are a number
of patterns a datashape might fit in.

Tabular datashapes have just one dimension, typically ``fixed`` or
``var``, followed by a record containing only simple types, not
nested records. This can be intuitively thought of as data which
will fit in a SQL table.::

    var * { x : int, y : real, z : date }

Homogenous datashapes are arrays that have a simple dtype, the kind
of data typically used in numeric computations. For example,
a 3D velocity field might look like::

    100 * 100 * 100 * 3 * real

Type Variables
~~~~~~~~~~~~~~

Type variables are a separate class of types that express free variables
scoped within type signatures. Holding type variables as first order
terms in the signatures encodes the fact that a term can be used in many
concrete contexts with different concrete types.

For example the type capable of expressing all square two dimensional
matrices could be written as a datashape with type variable ``A``,
constraining the two dimensions to be the same::

    A * A * int32

A type capable of rectangular variable length arrays of integers
can be written as two free type vars::

    A * B * int32

.. note::

   Any name beginning with an uppercase letter is parsed as a symbolic type
   (as opposed to concrete). Symbolic types can be used both as dimensions and
   as data types.

Option
~~~~~~

An option type represents data which may be there or not. This is like
data with ``NA`` values in R, or nullable columns in SQL. Given a type
like ``int``, it can be transformed by prefixing it with a question mark
as ``?int``, or equivalently using the type constructor ``option[int]``

For example a ``5 * ?int`` array can model the Python data:

::

    [1, 2, 3, None, None, 4]

