from __future__ import print_function, division, absolute_import

from . import coretypes, util
from .error import UnificationError, CoercionError, OverloadError
from .type_equation_solver import (match_argtypes_to_signature,
                                   PrunedMatchProcessing)

__all__ = ['OverloadResolver']

inf = float('inf')


class OverloadResolver(object):
    """
    An object which encapsulates multiple dispatch for a set of
    overloads, all of which are function signatures. This object
    is designed so that an acceleration structure could be added
    to it for speeding up overload resolution, without breaking
    the interface.

    Parameters
    ----------
    name : str
        This is the name of the function the overloader is for,
        for error messages to provide some more context.
    """
    def __init__(self, name):
        self.__overloads = []
        self.name = name

    def extend_overloads(self, overloads):
        """
        Extend the overload resolver's list of overloads by the
        provided list. All items in the overloads argument must
        be datashape function signatures, either as strings or
        as datashape type objects.
        """
        # Parse any strings in the input into dshape objects
        overloads = [util.dshape(ds) for ds in overloads]
        # Strip off the outer DataShape object
        overloads = [ds[0] if isinstance(ds, coretypes.DataShape)
                     and len(ds) == 1 else ds
                     for ds in overloads]
        # Make sure all the overloads are function signatures
        for ds in overloads:
            if not isinstance(ds, coretypes.Function):
                raise TypeError(('Only function signatures allowed as' +
                                'overloads, not %s') % ds)
        # Add the overloads to the end of the overloads list
        self.__overloads.extend(overloads)
        self._rebuild_overload_resolution_accel()

    def __getitem__(self, item):
        # Provide access to the overload signatures through [] operator
        return self.__overloads[item]

    def _rebuild_overload_resolution_accel(self):
        # TODO Create an accelerator data structure here
        pass

    def resolve_overload(self, argstype, resolver=None):
        """
        Given a tuple type representing input arguments, finds
        the best match of all the overloads and returns a tuple
        of its index and a function signature with all type
        variables resolved.

        Parameters
        ----------
        argstype : datashape tuple type
            A tuple type of all the input arguments.
        resolver : callable, optional
            A callable that can resolve typevars in the output
            type which are not resolved by the pattern matching
            of the inputs. It is called as resolver(sym, tvdict),
            where sym is the unresolved symbol and tvdict is a
            dictionary of all the matched symbols.
        """
        # TODO Use accelerator data structure instead of naive loop
        nargs = len(argstype.dshapes)
        result = []
        min_cost = inf
        err = None
        for i, sig in enumerate(self.__overloads):
            # If it's the right number of args, try matching it
            if nargs == len(sig.argtypes):
                try:
                    matched_sig, cost = match_argtypes_to_signature(argstype,
                                                                    sig,
                                                                    resolver,
                                                                    min_cost)
                except PrunedMatchProcessing:
                    pass
                except UnificationError as e:
                    err = e
                    pass
                except CoercionError as e:
                    err = e
                    pass
                else:
                    if cost <= min_cost:
                        if cost < min_cost:
                            result = []
                        min_cost = cost
                        result.append((i, matched_sig))
        if len(result) == 0:
            # If a coercion error was caught while matching,
            # reraise it.
            # TODO: The particular error we raise is arbitrary, make it better!
            if err is not None:
                raise err
            else:
                raise OverloadError(("%s: no overload matches" +
                                     " for argtypes %s") % (self.name,
                                                            argstype))
        elif len(result) > 1:
            raise OverloadError(("%s: ambiguous overload for" +
                                 " argtypes %s\nambiguous candidates:\n%s") %
                                (self.name, argstype,
                                 "\n".join("    %s" % x[1] for x in result)))
        return result[0]
