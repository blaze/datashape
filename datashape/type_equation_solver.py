"""
A solver for some simple equations involving datashape types, that
occur when doing multiple dispatch.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['matches_datashape_pattern', 'match_argtypes_to_signature',
           'explode_coercion_eqns']

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


def matches_datashape_pattern(concrete, symbolic):
    """
    Performs a pattern matching of a concrete datashape against
    a symbolic one which may include type variables. Returns
    True if it matches, False otherwise.
    """
    if not isinstance(concrete, coretypes.DataShape):
        raise TypeError("Expected a datashape for 'concrete', got %s" % concrete)
    if not isinstance(symbolic, coretypes.DataShape):
        raise TypeError("Expected a datashape for 'symbolic', got %s" % symbolic)

    # Match the concrete against the symbolic datashape
    try:
        eqn = _match_equation(concrete, symbolic)
    except error.CoercionError:
        return False
    dim_tv, dtype_tv = defaultdict(lambda: []), defaultdict(lambda: [])
    if not _process_equation_with_equality(eqn, dim_tv, dtype_tv):
        return False
    # Ensure that no TypeVar symbol has been used in multiple ways
    _check_inconsistent_tv_usage(dim_tv, dtype_tv)

    # Promote all the TypeVars together, to validate that their
    # usage is self-consistent. In this function, we don't need
    # to use the promoted values.
    try:
        _promote_dim_typevars(dim_tv)
        _promote_dtype_typevars(dtype_tv)
    except error.UnificationError:
        return False

    return True


def match_argtypes_to_signature(argtypes, signature, resolver=None,
                                cutoff_cost=inf):
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
    resolver : callable, optional
        A callable that can resolve typevars in the output
        type which are not resolved by the pattern matching
        of the inputs. It is called as resolver(sym, tvdict),
        where sym is the unresolved symbol and tvdict is a
        dictionary of all the matched symbols.
    cutoff_cost : float
        If the cost of this matching is higher than the cutoff,
        a PrunedMatchProcessing exception is raised.
    """
    # Pull the Tuple and Function out of the DataShape wrappers
    if isinstance(argtypes, coretypes.DataShape) and len(argtypes) == 1:
        argtypes = argtypes[0]
    if isinstance(signature, coretypes.DataShape) and len(signature) == 1:
        signature = signature[0]

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
        cost = _process_equation_with_coercion(eqn, dim_tv, dtype_tv)
        if cost == inf:
            raise error.CoercionError(argtypes, signature)
        elif cost > max_cost:
            if cost > cutoff_cost:
                raise PrunedMatchProcessing()
            else:
                max_cost = cost
    # Ensure that no TypeVar symbol has been used in multiple ways
    _check_inconsistent_tv_usage(dim_tv, dtype_tv)

    # Promote all the TypeVars tegether, and merge into one dict
    # since we've ensured there are no name collisions
    tv = _promote_dim_typevars(dim_tv)
    tv.update(_promote_dtype_typevars(dtype_tv))

    # Process all the argument types
    params = []
    for ds, eqn in zip(signature.argtypes, eqns):
        params.append(_substitute_typevars_with_matching(ds, eqn, tv))
    # Create the output type
    params.append(_substitute_typevars(signature.restype, tv, resolver))

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


def _process_equation_with_coercion(eqn, dim_tv, dtype_tv):
    """
    Collects all of the type variable values into the dim_tv and dtype_tv
    dicts, and returns the sum of all the broadcasting and coercion costs.

    dim_tv and dtype_tv should be initialized prior to calling the function
    like this::

        dim_tv, dtype_tv = defaultdict(lambda: []), defaultdict(lambda: [])
    """
    cost = 0
    for src, dst in eqn:
        if not isinstance(src, list) and getattr(src, 'cls', None) == coretypes.MEASURE:
            if isinstance(dst, coretypes.TypeVar):
                # Add to the dtype typevar dict
                dtype_tv[dst].append(src)
                # Cost of broadcasting to a typevar
                cost += 0.125
            else:
                # This is a concrete coerciion, evaluate its cost
                cost += coercion.dtype_coercion_cost(src, dst)
        else:
            if isinstance(dst, coretypes.TypeVar):
                # Add to the dim typevar dict
                dim_tv[dst].append(src)
                # Cost of broadcasting to an ellipsis
                cost += 0.125
            if isinstance(dst, coretypes.Ellipsis):
                # Add to the dim typevar dict
                dim_tv[dst].append(src)
                # Cost of broadcasting to an ellipsis
                cost += 0.25
            else:
                # This is a concrete broadcasting, evaluate its cost
                cost += coercion.dim_coercion_cost(src, dst)
    return cost


def _process_equation_with_equality(eqn, dim_tv, dtype_tv):
    """
    Collects all of the type variable values into the dim_tv and dtype_tv
    dicts. Returns True if all the concrete types matched, False if
    there was a mismatch (and does not complete the matching in that case).

    dim_tv and dtype_tv should be initialized prior to calling the function
    like this::

        dim_tv, dtype_tv = defaultdict(lambda: []), defaultdict(lambda: [])
    """
    for src, dst in eqn:
        if not isinstance(src, list) and getattr(src, 'cls', None) == coretypes.MEASURE:
            if isinstance(dst, coretypes.TypeVar):
                # Add to the dtype typevar dict
                dtype_tv[dst].append(src)
            elif src != dst:
                return False
        else:
            if isinstance(dst, (coretypes.TypeVar, coretypes.Ellipsis)):
                # Add to the dim typevar dict
                dim_tv[dst].append(src)
            elif src != dst:
                return False
    return True


def _check_inconsistent_tv_usage(dim_tv, dtype_tv):
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


def _substitute_typevars(ds, tv, resolver):
    """
    Substitutes the type variables in 'ds' using the
    typevar dictionary and the resolver function for
    typevars not in 'tv'. Returns the substituted datashape.
    """
    if isinstance(ds, coretypes.DataShape):
        params = []
        for x in ds.parameters:
            params.extend(_substitute_typevars(x, tv, resolver))
        return coretypes.DataShape(*params)
    elif isinstance(ds, (coretypes.Ellipsis, coretypes.TypeVar)):
        # Substitute the typevar, first trying 'tv', then
        # using the 'resolvets' function
        result = tv.get(ds, None)
        if result is None:
            if resolver is not None:
                result = resolver(ds, tv)
            if result is None:
                raise TypeError(('Could not resolve typevar %s in' +
                                 ' function signature output') % ds)
            elif isinstance(ds, coretypes.TypeVar):
                result = [result]
            elif not isinstance(result, list):
                raise TypeError(('When resolving ellipsis typevar %s,' +
                                 ' %s was returned but a list is required') %
                                (ds, result))
        elif isinstance(ds, coretypes.TypeVar):
            result = [result]
        return result
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
