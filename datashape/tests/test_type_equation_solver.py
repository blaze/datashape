from __future__ import absolute_import, division, print_function

import unittest

from datashape import coretypes as T
from datashape.type_equation_solver import (matches_datashape_pattern,
                                            match_argtypes_to_signature,
                                            _match_equation)
from datashape import dshape
from datashape import error
from datashape.coercion import dim_coercion_cost, dtype_coercion_cost


class TestPatternMatch(unittest.TestCase):
    def test_simple_matches(self):
        self.assertTrue(matches_datashape_pattern(dshape('int32'),
                                                  dshape('int32')))
        self.assertTrue(matches_datashape_pattern(dshape('int32'),
                                                  dshape('M')))
        self.assertTrue(matches_datashape_pattern(dshape('int32'),
                                                  dshape('A... * int32')))
        self.assertTrue(matches_datashape_pattern(dshape('int32'),
                                                  dshape('A... * M')))
        self.assertFalse(matches_datashape_pattern(dshape('int32'),
                                                  dshape('int64')))
        self.assertFalse(matches_datashape_pattern(dshape('3 * int32'),
                                                  dshape('M')))
        self.assertFalse(matches_datashape_pattern(dshape('int16'),
                                                  dshape('A... * int32')))
        self.assertFalse(matches_datashape_pattern(dshape('4 * int32'),
                                                  dshape('A... * 3 * M')))


class TestSignatureArgMatching(unittest.TestCase):
    def test_nargs_mismatch(self):
        # Make sure an error is rased when the # of arguments is wrong
        self.assertRaises(TypeError, match_argtypes_to_signature,
                          dshape('(int32, float64)'),
                          dshape('(int32) -> int32'))
        self.assertRaises(TypeError, match_argtypes_to_signature,
                          dshape('(int32, float64)'),
                          dshape('(int32, float64, int16) -> int32'))

    def test_dtype_matches_concrete(self):
        # Exact match, same signature and zero cost
        at = dshape('(int32, float64)')
        sig = dshape('(int32, float64) -> int16')
        self.assertEqual(match_argtypes_to_signature(at, sig),
                         (sig[0], 0))
        # Requires a coercion, cost is that of the coercion
        at = dshape('(int32, int32)')
        sig = dshape('(int32, float64) -> int16')
        self.assertEqual(match_argtypes_to_signature(at, sig),
                         (sig[0], dtype_coercion_cost(T.int32, T.float64)))
        # Requires two coercions, cost is maximum of the two
        at = dshape('(int16, int32)')
        sig = dshape('(int32, float64) -> int16')
        self.assertEqual(match_argtypes_to_signature(at, sig),
                         (sig[0], dtype_coercion_cost(T.int32, T.float64)))

    def test_dtype_coerce_error(self):
        at = dshape('(int32, float64)')
        sig = dshape('(int32, int32) -> int16')
        self.assertRaises(error.CoercionError, match_argtypes_to_signature,
                          at, sig)

    def test_dtype_matches_typevar(self):
        # Exact match, and zero cost
        at = dshape('(int32, float64)')
        sig = dshape('(int32, T) -> T')
        self.assertEqual(match_argtypes_to_signature(at, sig),
                         (dshape('(int32, float64) -> float64')[0], 0.125))
        # Type promotion between the inputs
        at = dshape('(int32, float64)')
        sig = dshape('(T, T) -> T')
        self.assertEqual(match_argtypes_to_signature(at, sig),
                         (dshape('(float64, float64) -> float64')[0], 0.125))
        # Type promotion between the inputs
        at = dshape('(int32, bool, float64)')
        sig = dshape('(T, S, T) -> S')
        self.assertEqual(match_argtypes_to_signature(at, sig),
                         (dshape('(float64, bool, float64) -> bool')[0], 0.125))

    def test_dshape_matches_concrete(self):
        # Exact match, same signature and zero cost
        at = dshape('(3 * int32, 2 * var * float64)')
        sig = dshape('(3 * int32, 2 * var * float64) -> 4 * int16')
        self.assertEqual(match_argtypes_to_signature(at, sig),
                         (sig[0], 0))
        # Requires broadcasting
        at = dshape('(1 * int32, 2 * 4 * float64)')
        sig = dshape('(3 * int32, 2 * var * float64) -> 4 * int16')
        self.assertEqual(match_argtypes_to_signature(at, sig),
                         (sig[0], max(dim_coercion_cost(T.Fixed(1), T.Fixed(3)),
                                      dim_coercion_cost(T.Fixed(4), T.Var()))))

    def test_dshape_matches_typevar(self):
        # Arrays with matching size
        at = dshape('(5 * int32, 5 * float64)')
        sig = dshape('(N * int32, N * float64) -> N * int16')
        self.assertEqual(match_argtypes_to_signature(at, sig),
                         (dshape('(5 * int32, 5 * float64) -> 5 * int16')[0],
                          0.125))
        # Matrix multiplication
        at = dshape('(3 * 5 * float64, 5 * 6 * float32)')
        sig = dshape('(M * N * A, N * R * A) -> M * R * A')
        self.assertEqual(match_argtypes_to_signature(at, sig),
                         (dshape('(3 * 5 * float64, 5 * 6 * float64) ->' +
                                 ' 3 * 6 * float64')[0], 0.375))
        # Broadcasted matrix multiplication
        at = dshape('(20 * 3 * 5 * float64, 3 * 1 * 5 * 6 * float32)')
        sig = dshape('(Dims... * M * N * A, Dims... * N * R * A) ->' +
                     ' Dims... * M * R * A')
        self.assertEqual(match_argtypes_to_signature(at, sig),
                         (dshape('(20 * 3 * 5 * float64,' +
                                 ' 3 * 1 * 5 * 6 * float64) ->' +
                                 ' 3 * 20 * 3 * 6 * float64')[0], 0.625))

    def test_dshape_dim_mismatch_error(self):
        # Single dimension type variables must match up exactly
        at = dshape('(1 * int32, 3 * float64)')
        sig = dshape('(M * int32, M * int32) -> M * int16')
        self.assertRaises(error.CoercionError, match_argtypes_to_signature,
                          at, sig)
        # Ellipsis typevars must broadcast
        at = dshape('(2 * int32, 3 * float64)')
        sig = dshape('(Dims... * int32, Dims... * int32) -> Dims... * int16')
        self.assertRaises(error.CoercionError, match_argtypes_to_signature,
                          at, sig)

    def test_broadcast_vs_not(self):
        # Single dimension type variables must match up exactly
        at = dshape('(int32, float64)')
        sig_scalar = dshape('(float64, float64) -> int16')
        sig_bcast = dshape('(A... * float64, A... * float64) -> A... * int16')
        match_scalar = match_argtypes_to_signature(at, sig_scalar)
        match_bcast = match_argtypes_to_signature(at, sig_bcast)
        self.assertEqual(match_scalar[0],
                         dshape('(float64, float64) -> int16')[0])
        self.assertEqual(match_bcast[0],
                         dshape('(float64, float64) -> int16')[0])
        # Should be cheaper to match without the broadcasting
        self.assertTrue(match_scalar[1] < match_bcast[1])

    def test_tv_matches_struct(self):
        at = dshape('(3 * {x: int, y: string}, 3 * bool)')
        sig = dshape('(M * T, M * bool) -> var * T')
        match = match_argtypes_to_signature(at, sig)
        self.assertEqual(match[0],
                         dshape('(3 * {x: int, y: string}, 3 * bool) -> var * {x: int, y: string}')[0])

    def test_match_with_resolver(self):
        # Test matching with a resolver function
        # This is a contrived resolver which combines the A... and
        # B typevars in a way that cannot be done with simple pattern
        # matching. While not a useful example in and of itself, it
        # exhibits the needed behavior in reduction function signature
        # matching.
        def resolver(tvar, tvdict):
            if tvar == T.Ellipsis(T.TypeVar('R')):
                a = tvdict[T.Ellipsis(T.TypeVar('A'))]
                b = tvdict[T.TypeVar('B')]
                result = [b]
                for x in a:
                    result.extend([x, b])
                return result
            elif tvar == T.TypeVar('T'):
                return T.int16
        at = dshape('(5 * int32, 4 * float64)')
        sig = dshape('(B * int32, A... * float64) -> R... * T')
        self.assertEqual(match_argtypes_to_signature(at, sig, resolver),
                         (dshape('(5 * int32, 4 * float64) -> 5 * 4 * 5 * int16')[0],
                          0.25))
        at = dshape('(5 * var * 2 * int32, 4 * float64)')
        sig = dshape('(A... * int32, B * float64) -> R... * 2 * T')
        self.assertEqual(match_argtypes_to_signature(at, sig, resolver),
                         (dshape('(5 * var * 2 * int32, 4 * float64) ->' +
                                 ' 4 * 5 * 4 * var * 4 * 2 * 4 * 2 * int16')[0],
                          0.25))


class TestEquationMatching(unittest.TestCase):
    def test_match_equation_dtype(self):
        # A simple coercion
        eqns = _match_equation(dshape('int32'), dshape('int64'))
        self.assertEqual(eqns, [(T.int32, T.int64)])
        # Matching a data type variable
        eqns = _match_equation(dshape('int32'), dshape('D'))
        self.assertEqual(eqns, [(T.int32, T.TypeVar('D'))])

    def test_match_equation_dim(self):
        # Broadcasting a single dimension
        eqns = _match_equation(dshape('1 * int32'), dshape('10 * int32'))
        self.assertEqual(eqns, [(T.Fixed(1), T.Fixed(10)),
                                (T.int32, T.int32)])
        # Matching a dim type variable
        eqns = _match_equation(dshape('3 * int32'), dshape('M * int32'))
        self.assertEqual(eqns, [(T.Fixed(3), T.TypeVar('M')),
                                (T.int32, T.int32)])

    def test_match_equation_ellipsis(self):
        # Matching an ellipsis
        eqns = _match_equation(dshape('int32'), dshape('... * int32'))
        self.assertEqual(eqns, [([], T.Ellipsis()),
                                (T.int32, T.int32)])
        eqns = _match_equation(dshape('3 * int32'), dshape('... * int32'))
        self.assertEqual(eqns, [([T.Fixed(3)], T.Ellipsis()),
                                (T.int32, T.int32)])
        eqns = _match_equation(dshape('3 * var * int32'), dshape('... * int32'))
        self.assertEqual(eqns, [([T.Fixed(3), T.Var()], T.Ellipsis()),
                                (T.int32, T.int32)])
        # Matching an ellipsis type variable
        eqns = _match_equation(dshape('int32'), dshape('A... * int32'))
        self.assertEqual(eqns, [([], T.Ellipsis(T.TypeVar('A'))),
                                (T.int32, T.int32)])
        eqns = _match_equation(dshape('3 * int32'), dshape('A... * int32'))
        self.assertEqual(eqns, [([T.Fixed(3)], T.Ellipsis(T.TypeVar('A'))),
                                (T.int32, T.int32)])
        eqns = _match_equation(dshape('3 * var * int32'), dshape('A... * int32'))
        self.assertEqual(eqns, [([T.Fixed(3), T.Var()], T.Ellipsis(T.TypeVar('A'))),
                                (T.int32, T.int32)])
        # Matching an ellipsis with a dim type variable on the left
        eqns = _match_equation(dshape('3 * var * int32'), dshape('A * B... * int32'))
        self.assertEqual(eqns, [(T.Fixed(3), T.TypeVar('A')),
                                ([T.Var()], T.Ellipsis(T.TypeVar('B'))),
                                (T.int32, T.int32)])
        # Matching an ellipsis with a dim type variable on the right
        eqns = _match_equation(dshape('3 * var * int32'), dshape('A... * B * int32'))
        self.assertEqual(eqns, [([T.Fixed(3)], T.Ellipsis(T.TypeVar('A'))),
                                (T.Var(), T.TypeVar('B')),
                                (T.int32, T.int32)])
        # Matching an ellipsis with a dim type variable on both sides
        eqns = _match_equation(dshape('3 * var * int32'), dshape('A * B... * C * int32'))
        self.assertEqual(eqns, [(T.Fixed(3), T.TypeVar('A')),
                                ([], T.Ellipsis(T.TypeVar('B'))),
                                (T.Var(), T.TypeVar('C')),
                                (T.int32, T.int32)])
        eqns = _match_equation(dshape('3 * var * 4 * M * int32'), dshape('A * B... * C * int32'))
        self.assertEqual(eqns, [(T.Fixed(3), T.TypeVar('A')),
                                ([T.Var(), T.Fixed(4)], T.Ellipsis(T.TypeVar('B'))),
                                (T.TypeVar('M'), T.TypeVar('C')),
                                (T.int32, T.int32)])
