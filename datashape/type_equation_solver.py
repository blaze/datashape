"""
A solver for some simple equations involving datashape types, that
occur when doing multiple dispatch.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['match_argtypes_to_signature', 'explode_coercion_eqns']

from collections import defaultdict
from functools import reduce

from . import coretypes
from . import error
from . import coercion
from . import promotion

inf = float('inf')


class PrunedMatchProcessing(Exception):
    """
    An exception thrown when a possible match is pruned because
    its cost is higher than the threshold.
    """


def match_argtypes_to_signature(argtypes, signature, cutoff_cost=inf):
    """
    Performs a pattern matching of the argument types against the
    function signature. Raises an exception if it cannot be matched,
    and returns a tuple (matched_signature, cost).

    Parameters
    ----------
    argtypes : Tuple datashape object
        A datashape tuple type composed of all the input parameter types.
    signature : Function signature datashape object
        A datashape function signature type against which to match.
    cutoff_cost : float
        If the cost of this matching is higher than the cutoff,
        a PrunedMatchProcessing exception is raised.
    """
    # Pull the Tuple and Function out of the DataShape wrappers
    if isinstance(argtypes, coretypes.DataShape) and len(argtypes) == 1:
        argtypes = argtypes[0]
    else:
        raise TypeError('invalid argtypes %s' % argtypes)
    if isinstance(signature, coretypes.DataShape) and len(signature) == 1:
        signature = signature[0]
    else:
        raise TypeError('invalid argtypes %s' % argtypes)

    # Validate the argument types
    if not isinstance(argtypes, coretypes.Tuple):
        raise TypeError('argtypes must be a datashape.Tuple')
    if not isinstance(signature, coretypes.Function):
        raise TypeError('signature must be a datashape.Function')

    # The number of arguments must match
    if len(argtypes.dshapes) != len(signature.argtypes):
        raise TypeError(('Cannot match signature, expected ' +
                         '%d arguments, got %d') %
                        (len(signature.argtypes), len(argtypes)))

    # Build a system of coercion equations
    eqns = zip(argtypes.dshapes, signature.argtypes)
    # Break down each equation down into a series of equations
    # with the same structure as the 'dst' datashape
    eqns = [_match_equation(src, dst) for src, dst in eqns]

    # Validate the broadcastiong/coercion and collect the typevar values
    dim_tv, dtype_tv = defaultdict(lambda: []), defaultdict(lambda: [])
    max_cost = 0
    for eqn in eqns:
        cost = _process_equation(eqn, dim_tv, dtype_tv)
        if cost == inf:
            raise TypeError(('Cannot coerce and broadcast arg types %d ' +
                             'to signature %d') % (argtypes, signature))
        elif cost > max_cost:
            if cost > cutoff_cost:
                raise PrunedMatchProcessing()
            else:
                max_cost = cost
    # Ensure that no TypeVar symbol has been used in multiple ways
    for tv in dim_tv:
        if isinstance(tv, coretypes.Ellipsis):
            s = tv.typevar
            if s in dim_tv:
                raise TypeError(('DataShape typevar %s has been ' +
                                 'used as both a dim and an ellipsis') % (s))
            elif s in dtype_tv:
                raise TypeError(('DataShape typevar %s has been ' +
                                 'used as both a dtype and an ellipsis') % (s))
        else:
            if tv in dtype_tv:
                raise TypeError(('DataShape typevar %s has been ' +
                                 'used as both a dtype and a dim') % (tv))
    # Promote all the TypeVars tegether, and merge into one dict
    # since we've ensured there are no name collisions
    tv = _promote_dim_typevars(dim_tv)
    tv.update(_promote_dtype_typevars(dtype_tv))

    # Process all the argument types
    params = []
    for ds, eqn in zip(signature.argtypes, eqns):
        params.append(_substitute_typevars_with_matching(ds, eqn, tv))
    # Create the output type
    params.append(_substitute_typevars(signature.restype, tv))

    # Return the resulting function signature and cost
    return (coretypes.Function(*params), max_cost)


def _match_equation(src, dst):
    """
    Matches a single src datashape against a dst datashape, building
    a nested structure of equations matching the individual terms
    in 'dst' to the corresponding parts in 'src'.
    """
    if isinstance(src, coretypes.DataShape) and isinstance(dst,
                                                           coretypes.DataShape):
        # Start off with a list matching the length of 'dst'
        eqns = [None] * len(dst)
        # Add all the broadcasting operations
        src_i, dst_i = 0, 0
        src_j, dst_j = len(src) - 2, len(dst) - 2
        # First match from the left, until we hit an ellipsis
        while src_i <= src_j and dst_i <= dst_j:
            if not isinstance(dst[dst_i], coretypes.Ellipsis):
                eqns[dst_i] = (src[src_i], dst[dst_i])
                src_i, dst_i = src_i + 1, dst_i + 1
            else:
                break
        # Then match from the right, until we hit an ellipsis
        while src_i <= src_j and dst_i <= dst_j:
            if not isinstance(dst[dst_j], coretypes.Ellipsis):
                eqns[dst_j] = (src[src_j], dst[dst_j])
                src_j, dst_j = src_j - 1, dst_j - 1
            else:
                break
        # Handle the ellipsis, if we hit one
        if dst_i == dst_j and isinstance(dst[dst_i], coretypes.Ellipsis):
            eqns[dst_i] = ([src[i] for i in range(src_i, src_j + 1)],
                           dst[dst_i])
            src_i, dst_i = src_j + 1, dst_i + 1
        # If we didn't match all the dimensions together, it's an error
        if src_i <= src_j or dst_i <= dst_j:
            raise error.CoercionError(src, dst)
        # Match the data type
        # TODO: For struct and similar types, make a recursive call here
        eqns[-1] = (src[-1], dst[-1])
        return eqns
    else:
        raise TypeError(('Only DataShape matching is implemented, ' +
                         'not yet %s or %s') % (src, dst))


def _process_equation(eqn, dim_tv, dtype_tv):
    """
    Collects all of the type variable values into the dim_tv and dtype_tv
    dicts, and returns the sum of all the broadcasting and coercion costs.
    """
    cost = 0
    for src, dst in eqn:
        if not isinstance(src, list) and src.cls == coretypes.MEASURE:
            if isinstance(dst, coretypes.TypeVar):
                # Add to the dtype typevar dict
                dtype_tv[dst].append(src)
            else:
                # This is a concrete coerciion, evaluate its cost
                cost += coercion.dtype_coercion_cost(src, dst)
        else:
            if isinstance(dst, (coretypes.Ellipsis, coretypes.TypeVar)):
                # Add to the dim typevar dict
                dim_tv[dst].append(src)
            else:
                # This is a concrete broadcasting, evaluate its cost
                cost += coercion.dim_coercion_cost(src, dst)
    return cost


def _promote_dim_typevars(dim_tv):
    """
    Takes a dict which contains a list of all the dim values
    each typevar takes on, and return a dict which contains
    the broadcasted result.
    """
    result = {}
    for tv in dim_tv:
        if isinstance(tv, coretypes.Ellipsis):
            result[tv] = reduce(promotion.broadcast_dims, dim_tv[tv])
        else:
            vals = dim_tv[tv]
            if not all(x == vals[0] for x in vals[1:]):
                raise error.UnificationError(("All typevar dims for %s must " +
                                              "match: %s") % (tv, vals))
            result[tv] = vals[0]
    return result


def _promote_dtype_typevars(dtype_tv):
    """
    Takes a dict which contains a list of all the dtype values
    each typevar takes on, and return a dict which contains
    the promoted result.
    """
    result = {}
    for tv in dtype_tv:
        result[tv] = reduce(promotion.promote_dtypes, dtype_tv[tv])
    return result


def _substitute_typevars(ds, tv):
    """
    Substitutes the type variables in 'ds' using the
    typevar dictionary. Returns the substituted datashape.
    """
    if isinstance(ds, coretypes.DataShape):
        params = []
        for x in ds.parameters:
            params.extend(_substitute_typevars(x, tv))
        return coretypes.DataShape(*params)
    elif isinstance(ds, coretypes.Ellipsis):
        # Substitute the typevar, leaving it as is if it's not
        # in the dict
        return tv.get(ds, ds)
    elif isinstance(ds, coretypes.TypeVar):
        # Substitute the typevar, leaving it as is if it's not
        # in the dict
        return [tv.get(ds, ds)]
    else:
        # TODO: recursively handle structs and similar types
        return [ds]


def _substitute_typevars_with_matching(ds, eqn, tv):
    """
    Substitutes the type variables in 'ds' using the
    typevar dictionary, except where it's a typevar which
    contributes to the typevar value without taking it on,
    such as a broadcasting ellipsis.
    """
    if isinstance(ds, coretypes.DataShape):
        params = []
        for x, y in zip(ds.parameters, eqn):
            params.extend(_substitute_typevars_with_matching(x, y, tv))
        return coretypes.DataShape(*params)
    elif isinstance(ds, coretypes.Ellipsis):
        # Grab the type from the matching equation
        return eqn[0]
    elif isinstance(ds, coretypes.TypeVar):
        # Substitute the typevar, leaving it as is if it's not
        # in the dict
        return [tv.get(ds, ds)]
    else:
        # TODO: recursively handle structs and similar types
        return [ds]
