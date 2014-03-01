"""
A solver for some simple equations involving datashape types, that
occur when doing multiple dispatch.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['match_argtypes_to_signature', 'explode_coercion_eqns']

from . import coretypes as T
from . import error

def match_argtypes_to_signature(argtypes, signature):
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
    if isinstance(argtypes, T.DataShape) and len(argtypes) == 1:
        argtypes = argtypes[0]
    else:
        raise TypeError('invalid argtypes %s' % argtypes)
    if isinstance(signature, T.DataShape) and len(signature) == 1:
        signature = signature[0]
    else:
        raise TypeError('invalid argtypes %s' % argtypes)
    if not isinstance(argtypes, T.Tuple):
        raise TypeError('argtypes must be a datashape.Tuple')
    if not isinstance(signature, T.Function):
        raise TypeError('signature must be a datashape.Function')
    # The number of arguments must match
    if len(argtypes.dshapes) != len(signature.argtypes):
        raise TypeError(('Cannot match signature, expected ' +
                         '%d arguments, got %d') %
                        (len(signature.argtypes), len(argtypes)))
    # First build a system of coercion equations
    eqns = zip(argtypes.dshapes, signature.argtypes)
    # Break it down into a system of separate broadcasting and coercion equations
    bcast_eqns, coerce_eqns = explode_coercion_eqns(eqns)

def explode_coercion_eqns(eqns):
    """
    Breaks a system of datashape coercion equations into simpler
    dimension type broadcast equations and data type coercion equations.

    The resulting equations are in the form (src, dst, equation_index),
    including the index where they came from to help with substituting
    values for type variables later.
    """
    bcast_eqns = []
    coerce_eqns = []
    for eqn_idx, (src_ds, dst_ds) in enumerate(eqns):
        print('processing equation %s ==> %s' % (src_ds, dst_ds))
        # Add the data type coercion
        coerce_eqns.append((src_ds[-1], dst_ds[-1], eqn_idx))
        # Add all the broadcasting operations, starting from the right
        src_i, dst_i = 0, 0
        src_j, dst_j = len(src_ds) - 2, len(dst_ds) - 2
        while src_j >= src_i and dst_j >= dst_i:
            src = src_ds[src_j]
            dst = dst_ds[dst_j]
            print("processing term %s ==> %s" % (src, dst))
            if isinstance(dst, T.Ellipsis):
                # Since we hit an ellipsis, we need to now process
                # the dims from the left to drill down on the part
                # which broadcasts via adding dimensions
                pre_eqn = []
                print('leading test', src_i, dst_i, src_i, dst_i)
                while src_i <= src_j and dst_i < dst_j:
                    pre_eqn.append((src_ds[src_i], dst_ds[dst_i], eqn_idx))
                    print('added to pre_eqn')
                    print(pre_eqn)
                    src_i, dst_i = src_i + 1, dst_i + 1
                # When broadcasting against an ellipsis, the src side gets a
                # list of dimensions, instead of just one
                src = [src_ds[i] for i in range(src_i, src_j + 1)]
                pre_eqn.append((src, dst, eqn_idx))
                pre_eqn.extend(bcast_eqns)
                bcast_eqns = pre_eqn
                src_j, dst_j = src_i - 1, dst_j - 1
            else:
                bcast_eqns.insert(0, (src, dst, eqn_idx))
                src_j, dst_j = src_j - 1, dst_j - 1
        # As a special case, an ellipsis can match against nothing
        print('ellipsis test', dst_i, dst_j, dst_ds[dst_i])
        print(bcast_eqns)
        if src_i > src_j and dst_i == dst_j and isinstance(dst_ds[dst_i], T.Ellipsis):
            bcast_eqns.insert(0, ([], dst_ds[dst_j], eqn_idx))
            dst_j = dst_j - 1
        # If we didn't match all the dimensions together, it's an error
        if src_i <= src_j or dst_i <= dst_j:
            print('indexes', src_i, dst_i, src_j, dst_j)
            raise error.UnificationError(('Could not coerce dimensions of ' +
                                          '%s into %s') % (src_ds, dst_ds))
    return (bcast_eqns, coerce_eqns)
