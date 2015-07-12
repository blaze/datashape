from __future__ import absolute_import

import numpy as np
import datashape


__all__ = ['promote', 'optionify']


def promote(lhs, rhs):
    """Promote two scalar dshapes to a possibly larger, but compatible type.

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
    if lhs == rhs:
        return lhs
    else:
        left, right = getattr(lhs, 'ty', lhs), getattr(rhs, 'ty', rhs)
        dtype = np.result_type(datashape.to_numpy_dtype(left),
                               datashape.to_numpy_dtype(right))
        return optionify(lhs, rhs, datashape.CType.from_numpy_dtype(dtype))


def optionify(lhs, rhs, dshape):
    """Check whether a binary operation's dshape came from
    :class:`~datashape.coretypes.Option` typed operands and construct an
    :class:`~datashape.coretypes.Option` type accordingly.

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


def broadcast_dims(dim1, dim2):
    """Broadcasts two dimension types or two lists of dimension types together.
    """
    if isinstance(dim1, list) and isinstance(dim2, list):
        # Broadcast a list of dimensions
        if len(dim1) > len(dim2):
            result = list(dim1)
            other = dim2
        else:
            result = list(dim2)
            other = dim1
        offset = len(result) - len(other)
        for i, dim in enumerate(other):
            result[offset + i] = broadcast_dims(result[offset + i], dim)
        return result
    else:
        # Broadcast a single dimension
        if isinstance(dim1, Fixed):
            if isinstance(dim2, Fixed):
                if dim1 == Fixed(1):
                    return dim2
                elif dim2 == Fixed(1):
                    return dim1
                else:
                    if dim1 == dim2:
                        return dim1
                    else:
                        raise TypeError(
                            "Cannot broadcast differing fixed dimensions "
                            "%s and %s" % (dim1, dim2))
            elif isinstance(dim2, Var):
                if dim1 == Fixed(1):
                    return dim2
                else:
                    return dim1
            else:
                raise TypeError("Unknown dim types, cannot broadcast: "
                                "%s and %s" % (dim1, dim2))
        elif isinstance(dim1, Var):
            if isinstance(dim2, Fixed):
                if dim2 == Fixed(1):
                    return dim1
                else:
                    return dim2
            elif isinstance(dim2, Var):
                return dim1
            else:
                raise TypeError(("Unknown dim types, cannot broadcast: " +
                                 "%s and %s") % (dim1, dim2))
