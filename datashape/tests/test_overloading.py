from __future__ import print_function, division, absolute_import

import unittest

from datashape.py2help import skip

from datashape import dshape, dshapes, unify_simple
from datashape import coretypes

from datashape.overloading import best_match, overload


#f

@overload('(X * Y * float32, X * Y * float32) -> X * Y * float32')
def f(a, b):
    return a

@overload('(X * Y * complex[float32], X * Y * complex[float32]) -> X * Y * complex[float32]')
def f(a, b):
    return a

@overload('(X * Y * complex[float64], X * Y * complex[float64]) -> X * Y * complex[float64]')
def f(a, b):
    return a

# g

@overload('(X * Y * float32, X * Y * float32) -> X * int32')
def g(a, b):
    return a

@overload('(X * Y * float32, ... * float32) -> X * int32')
def g(a, b):
    return a

# h

@overload('(A... * int64) -> A... * int64')
def h(a):
    return a

@overload('(A... * uint64) -> A... * uint64')
def h(a):
    return a

# j

@overload('(A... * float64, A... * float64) -> A... * float64')
def j(a, b):
    return a

@overload('(A... * complex[float32], A... * complex[float32]) -> A... * complex[float32]')
def j(a, b):
    return a

# k

@overload('(A... * float32) -> A... * float32')
def k(a):
    return a

@overload('(A... * float64) -> A... * float64')
def k(a):
    return a

# Define the test function 'm'
@overload('(A... * int32, A... * int32) -> A... * int32')
def m(a):
    return a
@overload('(A... * float32, A... * float32) -> A... * float32')
def m(a):
    return a
@overload('(A... * float64, A... * float64) -> A... * float64')
def m(a):
    return a

class TestOverloading(unittest.TestCase):
    @skip('function signatures with typevars are not supported presently')
    def test_best_match(self):
        d1 = dshape('10 * T1 * int32')
        d2 = dshape('T2 * T2 * float32')
        match = best_match(f, coretypes.Tuple([d1, d2]))
        self.assertEqual(str(match.sig),
                         '(X * Y * float32, X * Y * float32) -> X * Y * float32')

        input = dshape('(S * 1 * float32, T * 1 * float32) -> R')
        print('\n\n\n\n')
        unified = unify_simple(input, match.resolved_sig)
        self.assertEqual(str(unified),
                         '(10 * 1 * float32, 10 * 1 * float32) -> 10 * 1 * float32')

    def test_best_match_typevar_dims(self):
        d1 = dshape('3 * 10 * complex[float32]')
        d2 = dshape('3 * 10 * float32')
        match = best_match(f, coretypes.Tuple([d1, d2]))
        self.assertEqual(str(match.sig),
                         '(X * Y * complex[float32], X * Y * complex[float32]) -> X * Y * complex[float32]')
        self.assertEqual(str(match.resolved_sig),
                         '(3 * 10 * complex[float32], 3 * 10 * complex[float32]) -> 3 * 10 * complex[float32]')

    def test_best_match_ellipses(self):
        d1 = dshape('10 * var * int32')
        d2 = dshape('... * float32')
        match = best_match(g, coretypes.Tuple([d1, d2]))
        self.assertEqual(str(match.sig),
                         '(X * Y * float32, ... * float32) -> X * int32')
        self.assertEqual(str(match.resolved_sig),
                         '(10 * var * float32, ... * float32) -> 10 * int32')

    def test_best_match_signed_vs_unsigned(self):
        d1 = dshape('10 * 3 * int64')
        match = best_match(h, coretypes.Tuple([d1]))
        self.assertEqual(str(match.sig), '(A... * int64) -> A... * int64')
        self.assertEqual(str(match.resolved_sig),
                         '(10 * 3 * int64) -> 10 * 3 * int64')
        d1 = dshape('4 * 5 * uint64')
        match = best_match(h, coretypes.Tuple([d1]))
        self.assertEqual(str(match.sig), '(A... * uint64) -> A... * uint64')
        self.assertEqual(str(match.resolved_sig),
                         '(4 * 5 * uint64) -> 4 * 5 * uint64')

    def test_best_match_float_int_complex(self):
        d1, d2 = dshapes('3 * float64', 'int32')
        match = best_match(j, coretypes.Tuple([d1, d2]))
        self.assertEqual(str(match.sig), '(A... * float64, A... * float64) -> A... * float64')
        self.assertEqual(str(match.resolved_sig),
                         '(3 * float64, float64) -> 3 * float64')

    def test_best_match_int_float32_vs_float64(self):
        d1 = dshape('3 * int32')
        match = best_match(k, coretypes.Tuple([d1]))
        self.assertEqual(str(match.sig), '(A... * float64) -> A... * float64')
        self.assertEqual(str(match.resolved_sig),
                         '(3 * float64) -> 3 * float64')

    def test_best_match_int32_float32_ufunc_promotion(self):
        d1, d2 = dshapes('3 * int32', '3 * float32')
        match = best_match(m, coretypes.Tuple([d1, d2]))
        self.assertEqual(str(match.sig), '(A... * float64, A... * float64) -> A... * float64')
        self.assertEqual(str(match.resolved_sig),
                         '(3 * float64, 3 * float64) -> 3 * float64')

if __name__ == '__main__':
    #TestOverloading('test_best_match_broadcasting').debug()
    unittest.main()
