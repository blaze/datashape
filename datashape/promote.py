from __future__ import absolute_import

import numpy as np
import datashape


__all__ = ['promote', 'optionify']


def promote(lhs, rhs):
    """Promote two scalar dshapes to a possibly larger, but compatibile type



    Examples
    --------
    >>> from datashape import int32, int64, Option
    >>> x = Option(int32)
    >>> y = int64
    >>> promote(x, y)
    ?int64

    Notes
    ----
    This uses ``numpy.promote_types`` for type promotion logic.  See the numpy
    documentation at
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.promote_types.html
    """
    left, right = getattr(lhs, 'ty', lhs), getattr(rhs, 'ty', rhs)
    dtype = np.promote_types(datashape.to_numpy_dtype(left),
                             datashape.to_numpy_dtype(right))
    dshape = datashape.from_numpy((), dtype)
    return optionify(lhs, rhs, dshape)


def optionify(lhs, rhs, dshape):
    """Check whether a binary operation's dshape came from Option dshaped
    operands and construct an Option type accordingly

    Examples
    --------
    >>> from datashape import int32, int64, Option
    >>> x = Option(int32)
    >>> x
    ?int32
    >>> y = int64
    >>> y
    ctype("int64")
    >>> optionify(x, y, int64)
    ?int64
    """
    if hasattr(dshape.measure, 'ty'):
        return dshape
    if hasattr(lhs, 'ty') or hasattr(rhs, 'ty'):
        return datashape.Option(dshape)
    return dshape
