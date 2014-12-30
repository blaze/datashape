from __future__ import absolute_import

import numpy as np
import datashape


__all__ = ['promote', 'optionify']


def promote(lhs, rhs):
    """Promote two scalar dshapes to a possibly larger, but compatibile type

    Examples
    --------
    >>> from blaze import symbol
    >>> x = symbol('x', '?int32')
    >>> y = symbol('y', 'int64')
    >>> promote(x.schema.measure, y.schema.measure)
    ?int64
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
    >>> from blaze import symbol
    >>> from datashape import int64
    >>> x = symbol('x', '?int32')
    >>> y = symbol('y', 'int64')
    >>> optionify(x.schema.measure, y.schema.measure, int64)
    ?int64
    """
    if hasattr(lhs, 'ty') or hasattr(rhs, 'ty'):
        return datashape.Option(dshape)
    return dshape
