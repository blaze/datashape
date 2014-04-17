=================================
Data Attributes and Data Locators
=================================

Datashape types provide a way to give an array-oriented schema for
data to conform to, but do not specify how that data is laid out
in memory or on disk. To be useful for exchanging data in memory
between disparate systems, datashape additional defines data
attributes and data locators (dattr and dloc) to define this precisely.

Data Attributes
===============

Data attributes are specified as JSON objects associated with the type
constructors in a datashape type. Let's start with some simple examples
for various datashape dtypes.

Simple integer and floating point types have the endianness of their
storage specified::

    # datashape
    "int32", "float64", etc.

    # dattr
    {
        "endian" : "big"
    }

For the pointer type, the pointer could be pointing to a structure and
we're only interested in a field of it, so it's useful to provide
attributes like ``offset``::

    # datashape
    "pointer[int32]"

    # dattr
    {
        "offset" : 32,
        "target" : { "endian" : "little" }
    }

Dimension types follow a nested structure as follows::

    # datashape
    "3 * 4 * int32"

    # dattr
    {
        "stride" : 16,
        "element" : {
            "stride" : 4,
            "element" : {
                "endian" : "little"
            }
        }
    }

Data Locators
=============

Together, dshape and dattr can describe precisely how an array looks in memory or
on disk. The last piece of the puzzle is to describe where that data is. Data
locators fulfill this role.
