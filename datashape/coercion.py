"""Implements type coercion rules for data shapes.

Note that transitive coercions could be supported, but we decide not to since
it may involve calling a whole bunch of functions with a whole bunch of types
to figure out whether this is possible in the face of polymorphic overloads.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from itertools import chain, product

import datashape
from .error import CoercionError
from .coretypes import CType, TypeVar, Mono
from .typesets import boolean, complexes, floating, integral, signed, unsigned
from . import verify, normalize, Implements, Fixed, Var, Ellipsis, DataShape


class CoercionTable(object):
    """Table to hold coercion rules"""

    def __init__(self):
        self.table = {}
        self.srcs = defaultdict(set)
        self.dsts = defaultdict(set)

    def _reflexivity(self, a):
        if (a, a) not in self.table:
            self.table[a, a] = 0

    def add_coercion(self, src, dst, cost, transitive=True):
        """
        Add a coercion rule
        """
        assert cost >= 0, 'Raw coercion costs must be nonnegative'
        if (src, dst) not in self.table:
            self.srcs[dst].add(src)
            self.dsts[src].add(dst)
            self._reflexivity(src)
            self._reflexivity(dst)
            if src != dst:
                self.table[src, dst] = cost
                if transitive:
                    transitivity(src, dst, self)
        else:
            # Add the cost for the shortest path for the coercion
            self.table[src, dst] = min(self.table[src, dst], cost)

    def coercion_cost(self, src, dst):
        """
        Determine a coercion cost for coercing type `a` to type `b`
        """
        return self.table[src, dst]


_table = CoercionTable()
add_coercion = _table.add_coercion
coercion_cost_table = _table.coercion_cost

#------------------------------------------------------------------------
# Coercion invariants
#------------------------------------------------------------------------

def transitivity(a, b, table=_table):
    """Enforce coercion rule transitivity"""
    # (src, a) in R and (a, b) in R => (src, b) in R
    for src in table.srcs[a]:
        table.add_coercion(src, b, table.coercion_cost(src, a) +
                                   table.coercion_cost(a, b))

    # (a, b) in R and (b, dst) in R => (a, dst) in R
    for dst in table.dsts[b]:
        table.add_coercion(a, dst, table.coercion_cost(a, b) +
                                   table.coercion_cost(b, dst))

#------------------------------------------------------------------------
# Coercion function
#------------------------------------------------------------------------


def coercion_cost(a, b, seen=None):
    """
    Determine a coercion cost from type `a` to type `b`.

    Type `a` and `b'` must be unifiable and normalized.
    """
    # Determine the approximate cost and subtract the term size of the
    # right hand side: the more complicated the RHS, the more specific
    # the match should be
    return _coercion_cost(a, b, seen) - (termsize(b) / 100.0)


def _coercion_cost(a, b, seen=None):
    # TODO: Cost functions for conversion between type constructors in the
    # lattice (implement a "type join")

    if seen is None:
        seen = set()

    if a == b or isinstance(a, TypeVar):
        return 0
    elif isinstance(a, CType) and isinstance(b, CType):
        try:
            return coercion_cost_table(a, b)
        except KeyError:
            raise CoercionError(a, b)
    elif isinstance(b, TypeVar):
        visited = b not in seen
        seen.add(b)
        return 0.1 * visited
    elif isinstance(b, Implements):
        if a in b.typeset:
            return 0.1 - (0.1 / len(b.typeset.types))
        else:
            raise CoercionError(a, b)
    elif isinstance(b, Fixed):
        if isinstance(a, Var):
            return 0.1 # broadcasting penalty

        assert isinstance(a, Fixed)
        if a.val != b.val:
            assert a.val == 1 or b.val == 1
            return 0.1 # broadcasting penalty
        return 0
    elif isinstance(b, Var):
        assert type(a) in [Var, Fixed]
        if isinstance(a, Fixed):
            return 0.1 # broadcasting penalty
        return 0
    elif isinstance(a, DataShape) and isinstance(b, DataShape):
        return coerce_datashape(a, b, seen)
    else:
        verify(a, b)
        return sum([_coercion_cost(x, y, seen) for x, y in zip(a.parameters,
                                                               b.parameters)])


def termsize(term):
    """Determine the size of a type term"""
    if isinstance(term, Mono):
        return sum(termsize(p) for p in term.parameters) + 1
    return 0


def coerce_datashape(a, b, seen):
    # Penalize broadcasting
    broadcast_penalty = abs(len(a.parameters) - len(b.parameters))

    # Penalize ellipsis if one side has it but not the other
    ellipses_a = sum(isinstance(p, Ellipsis) for p in a.parameters)
    ellipses_b = sum(isinstance(p, Ellipsis) for p in b.parameters)
    ellipsis_penalty = ellipses_a ^ ellipses_b

    penalty = broadcast_penalty + ellipsis_penalty

    # Process rest of parameters
    [(a, b)], _ = normalize([(a, b)], [True])
    verify(a, b)
    for x, y in zip(a.parameters, b.parameters):
        penalty += coercion_cost(x, y, seen)

    return penalty


#------------------------------------------------------------------------
# Default coercion rules
#------------------------------------------------------------------------

def add_numeric_rule(types, distance=1):
    types = list(types)
    for src, dst in zip(types[:-1], types[1:]):
        add_coercion(src, dst, distance)

promotable_unsigned = [datashape.uint8, datashape.uint16, datashape.uint32]
promoted_signed     = [datashape.int16, datashape.int32, datashape.int64]

add_numeric_rule(signed)
add_numeric_rule(unsigned)
add_numeric_rule(floating)
add_numeric_rule(complexes)

add_numeric_rule([datashape.bool_, datashape.int8])
add_numeric_rule([datashape.uint8, datashape.int16])
add_numeric_rule([datashape.uint16, datashape.int32])
add_numeric_rule([datashape.uint32, datashape.int64])

add_numeric_rule([datashape.int16, datashape.float32])
add_numeric_rule([datashape.int32, datashape.float64])
add_numeric_rule([datashape.float32, datashape.complex64])
add_numeric_rule([datashape.float64, datashape.complex128])

# Potentially lossy conversions

# unsigned -> signed
add_numeric_rule([datashape.uint8, datashape.int8], 1.5)
add_numeric_rule([datashape.uint16, datashape.int16], 1.5)
add_numeric_rule([datashape.uint32, datashape.int32], 1.5)
add_numeric_rule([datashape.uint64, datashape.int64], 1.5)

# signed -> unsigned
add_numeric_rule([datashape.int8, datashape.uint8], 1.5)
add_numeric_rule([datashape.int16, datashape.uint16], 1.5)
add_numeric_rule([datashape.int32, datashape.uint32], 1.5)
add_numeric_rule([datashape.int64, datashape.uint64], 1.5)

# int -> float
add_numeric_rule([datashape.int32, datashape.float32], 1.5)
add_numeric_rule([datashape.int64, datashape.float64], 1.5)

# float -> complex
add_numeric_rule([datashape.float64, datashape.complex64], 1.5)
