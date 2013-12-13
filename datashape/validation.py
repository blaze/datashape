# -*- coding: utf-8 -*-

"""
Datashape validation.
"""

from .error import DataShapeError
from .traversal import traverse
from . import coretypes as T


def validate(ds):
    """
    Validate a Blaze type to see whether it is well-formed.

        >>> import blaze
        >>> blaze.dshape('10, int32')
        dshape("10, int32")
        >>> blaze.dshape('..., int32')
        dshape("..., int32")
        >>> blaze.dshape('..., ..., int32')
        Traceback (most recent call last):
            ...
        DataShapeError: Can only use a single wildcard
        >>> blaze.dshape('T, ..., T2, ..., int32 -> T, X')
        Traceback (most recent call last):
            ...
        DataShapeError: Can only use a single wildcard
        >>> blaze.dshape('T, ...')
        Traceback (most recent call last):
            ...
        DataShapeError: Measure may not be an Ellipsis (...)
    """
    traverse(_validate, ds)

def _validate(ds, params):
    if isinstance(ds, T.DataShape):

        # -------------------------------------------------
        # Check ellipses
        ellipses = [x for x in ds.parameters if isinstance(x, T.Ellipsis)]
        if len(ellipses) > 1:
            raise DataShapeError("Can only use a single wildcard")
        elif isinstance(ds.parameters[-1], T.Ellipsis):
            raise DataShapeError("Measure may not be an Ellipsis (...)")

        # -------------------------------------------------
        # Check constraints
        for x in ds.parameters[:-1]:
            if isinstance(x, T.Implements):
                # TODO: What about further constaints on the dimensions?
                raise DataShapeError(
                    "Only the measure can have constraints")


if __name__ == '__main__':
    import doctest
    doctest.testmod()
