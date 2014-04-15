"""Type promotion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .error import UnificationError
from .coretypes import CType, Fixed, Var, to_numpy_dtype


def broadcast_dims(dim1, dim2):
    """
    Broadcasts two dimension types or two
    lists of dimension types together.
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
                        raise UnificationError(
                            "Cannot broadcast differing fixed dimensions "
                            "%s and %s" % (dim1, dim2))
            elif isinstance(dim2, Var):
                if dim1 == Fixed(1):
                    return dim2
                else:
                    return dim1
            else:
                raise TypeError(("Unknown dim types, cannot broadcast: " +
                                 "%s and %s") % (dim1, dim2))
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


def promote_dtypes(dt1, dt2):
    """

    >>> from datashape.coretypes import int32, int64, float32
    >>> promote_dtypes(int32, int64)
    ctype("int64")

    >>> promote_dtypes(int64, float32)
    ctype("float64")
    """
    if dt1 == dt2:
        return dt1
    elif isinstance(dt1, CType) and isinstance(dt2, CType):
        # Promote CTypes -- this should use coercion_cost()
        try:
            return CType.from_numpy_dtype(np.result_type(to_numpy_dtype(dt1),
                                                         to_numpy_dtype(dt2)))
        except TypeError as e:
            raise UnificationError("Cannot promote %s and %s: %s" % (dt1, dt2, e))
    else:
       raise TypeError(("Unknown data types, cannot promote: " +
                        "%s and %s") % (dt1, dt2))
