Datashape Overview
==================

Datashape is a generalization of ``dtype`` and ``shape`` into a
type system which lets us overlay high level structure on existing
data in Table and Array objects.

Just like in traditional NumPy, the preferred method of implementing
generic vector operators is through ad-hoc polymorphism. Numpy's style
of ad-hoc polymorphism allows ufunc objects to have different behaviors
when "viewed" at different types. The runtime system then chooses an
appropriate implementation for each application of the function, based
on the types of the arguments. Blaze simply extends this specialization
to data structure and data layout as well as data type ( dtype ).

Many of the ideas behind datashape are generalizations and combinations
of notions found in Numpy:

.. cssclass:: table-bordered

+----------------+----------------+
| Numpy          | Blaze          |
+================+================+
| Broadcasting   | Unification    |
+----------------+----------------+
| Shape          |                |
+----------------+ Datashape      |
| Dtype          |                |
+----------------+----------------+
| Ufunc          | Gufunc         |
+----------------+----------------+

Units
-----

Datashape types that are single values are called **unit** types. They
represent a fixed type that has no internal structure. For example
``int32``.

In Blaze there are two classes of units: **dtypes** and
**dimensions**. Measures are units of data, while dimensions are
units of shape. The combination of dtype and dimension in datashape
constructors uniquely describe the space of possible values
of a table or array object.

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
complex[float64] omplex number, represented by two 64-bit floats (real
                 and imaginary components)
================ =========================================================

Additionally, there are types which are not fully specified at the
bit/byte level.

.. cssclass:: table-striped

==========  =========================================================
Bit type    Description
==========  =========================================================
string      Variable length Unicode string.
bytes       Variable length arrays of bytes.
json        Variable length Unicode string which contains JSON.
date        Dates in the proleptic Gregorian calendar.
time        Times not attached to a date.
datetime    Points in time, combination of date and time.
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
unicode values.

Endianness
~~~~~~~~~~

The datashape does not specify endianness, data types
are in native endianness when processed by Blaze functions.

Dimensions
--------

An asterisk (*) between two types signifies an array. A datashape
consists of 0 or more **dimensions** followed by a **dtype**.

Example::

    A * B

The array operator has the additional constraint that the first
operand cannot be a dtype. This permits types of the form::

    1 * int32
    1 * 1 * int32

But forbids types of the form::

    1 * 1
    int32 * 1
    int32 * int32

Fixed
~~~~~

The unit shape type is a dimension unit type. They are represented
as just integer values at the top level of the datatype. These are
identical to ``shape`` parameters in NumPy. For example::

    2 * int32

The previous signature Is an equivalent to the shape and dtype of a
NumPy array of the form::

    np.empty(2, 'int32')

A 2 by 3 matrix of integers has datashape::

    2 * 3 * int32

With the corresponding NumPy array::

    np.empty((2, 3), 'int32')

Records
~~~~~~~

Record types are ordered struct-like objects which hold a collection of
types keyed by labels. Records are also in the class of **dtypes**.
Records are sugared to look like Python dictionaries but
are themselves type constructors of variable number of type arguments.

Example 1::

    {
        name   : string,
        age    : int,
        height : int,
        weight : int
    }

Example 2::

    {
        r: int32,
        g: int32,
        b: int32,
        a: int8
    }

Records are themselves types declaration so they can be nested,
but cannot be self-referential:

Example 2::

    {
        a: { x: int, y: int };
        b: { x: int, y: int }
    }

Composite datashapes that terminate in record types are called
**table-like**, while any other terminating type is called
**array-like**.

Example of table-like::

    3 * { x : int, y : float }

Example of array-like::

    2 * 3 * int32


Type Variables
~~~~~~~~~~~~~~

**Type variables** a seperate class of types expressed as free variables
scoped within the type signature. Holding type variables as first order
terms in the signatures encodes the fact that a term can be used in many
concrete contexts with different concrete types.

Type variables that occur once in a type signature are referred to as
**free**, while type variables that appear multiple types are **rigid**.

For example the type capable of expressing all square two dimensional
matrices could be written as a combination of rigid type vars::

    A * A * int32

A type capable of rectangular variable length arrays of integers
can be written as two free type vars::

    A * B * int32

Sums
----

A **sum type** is a type representing a collection of heterogeneously
typed values.

* :ref:`option`

.. _option:

Option
~~~~~~

A Option is a tagged union representing with the left projection being
the presence of a value while the right projection being the absence of
a values. For example in C, all types can be nulled by using ``NULL``
reference.

For example a optional int field::

    option[int32]

Indicates the presense or absense of a integer. For example a
(``5 * option[int32]``) array could be model the Python data structure:

::

    [1, 2, 3, None, None, 4]

Option types are only defined for type arguments of unit dtypes and
records.

FAQ
---

* How do I convert from DataShape to NumPy shape and
  dtype?:

.. doctest::

    >>> from datashape import dshape, to_numpy
    >>> ds = dshape("5 * 5 * int32")
    >>> to_numpy(ds)
    ((5, 5), dtype('int32'))

* How do I convert from Numpy Dtype to Datashape?:

.. doctest::

    >>> from datashape import dshape, from_numpy
    >>> from numpy import dtype
    >>> from_numpy((5, 5), dtype('int32'))
    dshape("5 * 5 * int32")

