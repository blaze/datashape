"""Implements type coercion rules for data shapes.

Note that transitive coercions could be supported, but we decide not to since
it may involve calling a whole bunch of functions with a whole bunch of types
to figure out whether this is possible in the face of polymorphic overloads.
"""

from __future__ import absolute_import, division, print_function

from collections import defaultdict

from .error import CoercionError
from .coretypes import CType, TypeVar, Mono
from .typesets import complexes, floating, signed, unsigned
from .coretypes import Implements, Fixed, Var, DataShape
from . import coretypes

inf = float('inf')


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

def dimlist_coercion_cost(src, dst):
    """
    Cost of broadcasting one list of dimensions to another
    """
    # TODO: This is not handling ellipsis
    if len(src) > len(dst):
        return inf
    # Cost for adding dimensions is 0.1 for a size-one Fixed
    # dim, 0.2 for anything else
    leading = len(dst) - len(src)
    cost = sum(0.1 if x == Fixed(1) else 0.2 for x in dst[:leading])
    return cost + sum(dim_coercion_cost(x, y)
                      for x, y in zip(src, dst[leading:]))


def dim_coercion_cost(src, dst):
    """
    Cost of coercing one dimension type to another.
    """
    if isinstance(dst, Fixed):
        if isinstance(src, Var):
            return 0.1 # broadcasting penalty
        elif not isinstance(src, Fixed):
            return inf

        if src.val != dst.val:
            # broadcasting penalty
            return 0.1 if src.val == 1 else inf
        return 0
    elif isinstance(dst, Var):
        assert type(src) in [Var, Fixed]
        if isinstance(src, Fixed):
            return 0.1 # broadcasting penalty
        return 0
    elif isinstance(dst, TypeVar):
        return 0
    else:
        return inf


def dtype_coercion_cost(src, dst):
    """
    Cost of coercing one data type to another
    """
    if src == dst:
        return 0
    elif isinstance(src, CType) and isinstance(dst, CType):
        try:
            return coercion_cost_table(src, dst)
        except KeyError:
            return inf
    else:
        return inf


def _strip_datashape(a):
    """Strips off the outer DataShape(...) if a is zero-dimensional."""
    if isinstance(a, DataShape) and len(a) == 1:
        a = a[0]
    return a


def coercion_cost(a, b, seen=None):
    """
    Determine a coercion cost from type `a` to type `b`.

    Type `a` and `b'` must be unifiable and normalized.
    """
    return _coercion_cost(_strip_datashape(a), _strip_datashape(b), seen)


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
        return (dimlist_coercion_cost(a[:-1], b[:-1]) +
                dtype_coercion_cost(a[-1], b[-1]))
    else:
        raise TypeError(('Unhandled coercion cost case of ' +
                         '%s and %s') % (a, b))


def termsize(term):
    """Determine the size of a type term"""
    if isinstance(term, Mono):
        return sum(termsize(p) for p in term.parameters) + 1
    return 0


#------------------------------------------------------------------------
# Default coercion rules
#------------------------------------------------------------------------

def add_numeric_rule(types, cost=1):
    types = list(types)
    for src, dst in zip(types[:-1], types[1:]):
        add_coercion(src, dst, cost)

add_numeric_rule(signed)
add_numeric_rule(unsigned)
add_numeric_rule(floating)
add_numeric_rule(complexes)

add_numeric_rule([coretypes.uint8, coretypes.int16])
add_numeric_rule([coretypes.uint16, coretypes.int32])
add_numeric_rule([coretypes.uint32, coretypes.int64])

add_numeric_rule([coretypes.int16, coretypes.float32], 1.2)
add_numeric_rule([coretypes.int32, coretypes.float64], 1.2)
add_numeric_rule([coretypes.float32, coretypes.complex_float32], 1.2)
add_numeric_rule([coretypes.float64, coretypes.complex_float64], 1.2)

# Potentially lossy conversions

# unsigned -> signed
add_numeric_rule([coretypes.uint8, coretypes.int8], 1.5)
add_numeric_rule([coretypes.uint16, coretypes.int16], 1.5)
add_numeric_rule([coretypes.uint32, coretypes.int32], 1.5)
add_numeric_rule([coretypes.uint64, coretypes.int64], 1.5)

# signed -> unsigned
add_numeric_rule([coretypes.int8, coretypes.uint8], 1.5)
add_numeric_rule([coretypes.int16, coretypes.uint16], 1.5)
add_numeric_rule([coretypes.int32, coretypes.uint32], 1.5)
add_numeric_rule([coretypes.int64, coretypes.uint64], 1.5)

# int -> float
add_numeric_rule([coretypes.int32, coretypes.float32], 1.5)
add_numeric_rule([coretypes.int64, coretypes.float64], 1.5)

# float -> complex
add_numeric_rule([coretypes.float64, coretypes.complex_float32], 1.5)

# Anything -> bool
for tp in (list(signed) + list(unsigned) + list(floating) + list(complexes)):
    add_numeric_rule([tp, coretypes.bool_], 1000.)
