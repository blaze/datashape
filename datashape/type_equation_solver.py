"""
A solver for some simple equations involving datashape types, that
occur when doing multiple dispatch.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['match_argtypes_to_signature', 'explode_coercion_eqns']

from . import coretypes
from . import error
from . import coercion

inf = float('inf')


class _PruneProcessing(Exception):
    """An exception thrown when a possible match is pruned because its cost is higher than the threshold."""


def match_argtypes_to_signature(argtypes, signature, cutoff_cost):
    """
    Performs a pattern matching of the argument types against the
    function signature. Raises an exception if it cannot be matched,
    and returns a tuple (matched_signature, typevar_val_dict).

    Parameters
    ----------
    argtypes : Tuple datashape object
        A datashape tuple type composed of all the input parameter types.
    signature : Function signature datashape object
        A datashape function signature type against which to match.
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
    # First build a system of coercion equations
    eqns = zip(argtypes.dshapes, signature.argtypes)
    # Break down each equation down into a series of equations
    # with the same structure as the 'dst' datashape
    eqns = [_match_equation(src, dst) for src, dst in eqns]
    dim_tv, dtype_tv = {}

    # Ensure that no TypeVar symbol has been used in multiple ways
    for tv in bcast_eqns:
        if isinstance(tv, coretypes.Ellipsis):
            s = tv.typevar
            if s in bcast_eqns:
                raise TypeError(('DataShape typevar %s has been ' +
                                 'used as both a dim and an ellipsis') % (s))
            elif s in coerce_eqns:
                raise TypeError(('DataShape typevar %s has been ' +
                                 'used as both a dtype and an ellipsis') % (s))
        else:
            if s in coerce_eqns:
                raise TypeError(('DataShape typevar %s has been ' +
                                 'used as both a dtype and a dim') % (s))
    # Do all the promotions of the


def _match_equation(src, dst):
    """
    Matches a single src datashape against a dst datashape, building
    a nested structure of equations matching the individual terms
    in 'dst' to the corresponding parts in 'src'.

    As part of the matching, values taken on by type variables are
    accumulated in dim_tv and dtype_tv.
    """
    if isinstance(src, coretypes.DataShape) and isinstance(dst, coretypes.DataShape):
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


def process_broadcast_equations(bcast_eqns):
    """
    Collects all of the type variable values into a dictionary,
    and returns it with the maximum broadcasting cost.
    """
    vals = {}
    cost = 0
    for src, dst, eqn_idx in bcast_eqns:
        if isinstance(dst, coretypes.Ellipsis):
            # Indicate that the src value must be broadcastable to the type var value
            # TODO: If there is an "exact" version of the ellipsis typevar added, this would be '==' for it
            v = vals.get(dst, [])
            v.append(('>=', src, eqn_idx))
            vals[dst] = v
        elif isinstance(dst, coretypes.TypeVar):
            # Indicate that the src value must be equal to the type var value
            # TODO: If there is an "inexact" version of the typevar added, this would be '>=' for it
            v = vals.get(dst, [])
            v.append(('==', src, eqn_idx))
            vals[dst] = v
        elif isinstance(src, coretypes.TypeVar):
            # Indicate that the type var value must be equal to the dst value
            v = vals.get(src, [])
            v.append(('==', dst, eqn_idx))
            vals[src] = v
        else:
            # This is a concrete broadcasting, evaluate its cost
            c = coercion.dim_coercion_cost(src, dst)
            if c == inf:
                raise error.CoercionError(src, dst)
            cost = max(cost, c)
    return (vals, cost)

def process_coercion_equations(coerce_eqns):
    """
    Collects all of the type variable values into a dictionary,
    and returns the maximum coercion cost.
    """
    vals = {}
    cost = 0
    for src, dst, eqn_idx in coerce_eqns:
        if isinstance(dst, coretypes.TypeVar):
            # Indicate that the src value must be coercible to the type var value
            # TODO: If there is an "exact" version of the typevar added, this would be '==' for it
            v = vals.get(dst, [])
            v.append(('>=', src, eqn_idx))
            vals[dst] = v
        elif isinstance(src, coretypes.TypeVar):
            v = vals.get(src, [])
            v.append(('<=', dst, eqn_idx))
            vals[src] = v
        else:
            # This is a concrete coerciion, evaluate its cost
            c = coercion.dtype_coercion_cost(src, dst)
            if c == inf:
                raise error.CoercionError(src, dst)
            cost = max(cost, c)
    return (vals, cost)
