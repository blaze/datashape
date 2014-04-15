# -*- coding: utf-8 -*-

"""
Datashape validation.
"""

from .error import DataShapeError
from . import coretypes as T


def traverse(f, t):
    """
    Map f over t, calling `f` with type `t` and the map result of the mapping
    `f` over `t`s parameters.
    """
    if isinstance(t, T.Mono) and not isinstance(t, T.Unit):
        return f(t, [traverse(f, p) for p in t.parameters])
    return t


def validate(ds):
    """
    Validate a datashape to see whether it is well-formed.

        >>> from datashape import dshape
        >>> dshape('10 * int32')
        dshape("10 * int32")
        >>> dshape('... * int32')
        dshape("... * int32")
        >>> dshape('... * ... * int32') # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        DataShapeError: Can only use a single wildcard
        >>> dshape('T * ... * X * ... * X') # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        DataShapeError: Can only use a single wildcard
        >>> dshape('T * ...') # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        DataShapeSyntaxError: Expected a dtype
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
