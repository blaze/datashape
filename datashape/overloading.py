from __future__ import print_function, division, absolute_import

from collections import namedtuple

from .error import UnificationError, CoercionError, OverloadError
from . import coretypes as T
from .util import dshape, dummy_signature
from .overload_resolver import OverloadResolver
from .type_equation_solver import (match_argtypes_to_signature,
                                   PrunedMatchProcessing)

inf = float('inf')

Overload = namedtuple('Overload', 'resolved_sig, sig, func, kwds')


class Dispatcher(object):
    # TODO: Remove this class/file in favour of OverloadResolver
    """
    Dispatcher for overloaded functions

    Attributes
    ==========

    f: FunctionType
        Initial python function that got overloaded

    overloads: (FunctionType, str, dict)
        A three-tuple of (py_func, signature, kwds)
    """

    def __init__(self):
        self.overloader = OverloadResolver('<unnamed>')
        self.funcs_list = []
        self.kwds_list = []
        self.f = None

    def add_overload(self, f, signature, kwds, argspec=None):
        # TODO: assert signature is "compatible" with current signatures
        if self.f is None:
            self.f = f
        if hasattr(f, '__name__'):
            self.overloader.name = f.__name__

        # Add the signature and its data
        self.overloader.extend_overloads([signature])
        self.funcs_list.append(f)
        self.kwds_list.append(kwds)

    def lookup_dispatcher(self, args, kwargs):
        if len(kwargs) != 0:
            raise TypeError('Keyword arguments are not supported in this call')
        types = list(map(T.typeof, args))
        idx, match = self.overloader.resolve_overload(T.Tuple(types))
        return Overload(match, self.overloader[idx], self.funcs_list[idx],
                        self.kwds_list[idx]), args

    def best_match(self, argtypes):
        idx, match = self.overloader.resolve_overload(argtypes)
        return Overload(match, self.overloader[idx], self.funcs_list[idx],
                        self.kwds_list[idx])

    def __repr__(self):
        signatures = [sig for f, sig, _ in self.overloads]
        return '<%s: \n%s>' % (self.f and self.f.__name__,
                               "\n".join("    %s" % (s,) for s in signatures))


def overload(signature, dispatcher=None, **kwds):
    """
    Overload `func` with new signature, or find this function in the local
    scope with the same name.

        @overload('Array[dtype, ndim] -> dtype')
        def myfunc(...):
            ...
    """
    def decorator(f, signature=signature):
        if signature is None:
            signature = dummy_signature(f)
        else:
            signature = dshape(signature)

        disp = dispatcher or f.__globals__.get(f.__name__) or Dispatcher()
        disp.add_overload(f, signature, kwds)
        return disp

    return decorator
