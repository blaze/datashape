"""Type promotion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import product
from functools import reduce

import numpy as np

from .error import UnificationError
from .util import gensym, verify
from .coretypes import (DataShape, CType, Fixed, Var, to_numpy, to_numpy_dtype,
                        TypeVar)
from .typesets import TypeSet


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
    if dt1 == dt2:
        return dt1
    elif isinstance(dt1, CType) and isinstance(dt2, CType):
        # Promote CTypes -- this should use coercion_cost()
        try:
            return CType.from_numpy_dtype(np.result_type(to_numpy_dtype(dt1),
                                                         to_numpy_dtype(dt2)))
        except TypeError as e:
            raise TypeError("Cannot promote %s and %s: %s" % (dt1, dt2, e))
    else:
       raise TypeError(("Unknown data types, cannot promote: " +
                        "%s and %s") % (dt1, dt2))


def promote_units(*units):
    """
    Promote unit types, which are either CTypes or Constants.
    """
    return reduce(promote, units)


def promote(a, b):
    """Promote two blaze types"""

    if a == b:
        return a

    # -------------------------------------------------
    # Fixed

    elif isinstance(a, Fixed):
        if isinstance(b, Fixed):
            if a == Fixed(1):
                return b
            elif b == Fixed(1):
                return a
            else:
                if a != b:
                    raise UnificationError(
                        "Cannot unify differing fixed dimensions "
                        "%s and %s" % (a, b))
                return a
        elif isinstance(b, Var):
            if a == Fixed(1):
                return b
            else:
                return a
        else:
            raise TypeError("Unknown types, cannot promote: %s and %s" % (a, b))

    # -------------------------------------------------
    # Var

    elif isinstance(a, Var):
        if isinstance(b, Fixed):
            if b == Fixed(1):
                return a
            else:
                return b
        elif isinstance(b, Var):
            return a
        else:
            raise TypeError("Unknown types, cannot promote: %s and %s" % (a, b))

    # -------------------------------------------------
    # Typeset

    elif isinstance(a, TypeSet) and isinstance(b, TypeSet):
        # TODO: Find the join in the lattice with the below as a fallback ?
        return TypeSet(*set(promote(t1, t2)
                                for t1, t2 in product(a.types, b.types)))

    elif isinstance(a, TypeSet):
        if b not in a.types:
            raise UnificationError(
                "Type %s does not belong to typeset %s" % (b, a))
        return b

    elif isinstance(b, TypeSet):
        return promote(b, a)

    # -------------------------------------------------
    # Units

    elif isinstance(a, CType) and isinstance(b, CType):
        # Promote CTypes -- this should use coercion_cost()
        return promote_scalars(a, b)

    # -------------------------------------------------
    # DataShape

    elif isinstance(a, (DataShape, CType)) and isinstance(b, (DataShape, CType)):
        return promote_datashapes(a, b)

    else:
        raise TypeError("Unknown types, cannot promote: %s and %s" % (a, b))


def eq(a, b):
    if isinstance(a, TypeVar) and isinstance(b, TypeVar):
        return True
    return a == b


def promote_scalars(a, b):
    """Promote two CTypes"""
    try:
        return CType.from_numpy_dtype(np.result_type(to_numpy_dtype(a), to_numpy_dtype(b)))
    except TypeError as e:
        raise TypeError("Cannot promote %s and %s: %s" % (a, b, e))


def promote_datashapes(a, b):
    """Promote two DataShapes"""
    from .unification import unify
    from .normalization import normalize_simple

    # Normalize to determine parameters (eliminate broadcasting, etc)
    a, b = normalize_simple(a, b)
    n = len(a.parameters[:-1])

    # Allocate dummy result type for unification
    dst = DataShape(*[TypeVar(gensym()) for i in range(n + 1)])

    # Unify
    [result1, result2], _ = unify([(a, dst), (b, dst)])

    assert result1 == result2
    return result1


def promote_type_constructor(a, b):
    """Promote two generic type constructors"""
    # Verify type constructor equality
    verify(a, b)

    # Promote parameters according to flags
    args = []
    for flag, t1, t2 in zip(a.flags, a.parameters, b.parameters):
        if flag['coercible']:
            result = promote(t1, t2)
        else:
            if t1 != t2:
                raise UnificationError(
                    "Got differing types %s and %s for unpromotable type "
                    "parameter in constructors %s and %s" % (t1, t2, a, b))
            result = t1

        args.append(result)

    return type(a)(*args)
