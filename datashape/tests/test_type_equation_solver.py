from __future__ import absolute_import, division, print_function

import unittest

from datashape import coretypes as T
from datashape.type_equation_solver import match_argtypes_to_signature, explode_coercion_eqns
from datashape import dshape


class TestCoercion(unittest.TestCase):
    def test_nargs_mismatch(self):
        # Make sure an error is rased when the # of arguments is wrong
        self.assertRaises(TypeError, match_argtypes_to_signature,
                          dshape('(int32, float64)'),
                          dshape('(int32) -> int32'))
        self.assertRaises(TypeError, match_argtypes_to_signature,
                          dshape('(int32, float64)'),
                          dshape('(int32, float64, int16) -> int32'))

    def test_explode_coercion_eqns_dtype(self):
        # A simple coercion
        beqn, deqn = explode_coercion_eqns([(dshape('int32'), dshape('int64'))])
        self.assertEqual(beqn, [])
        self.assertEqual(deqn, [(T.int32, T.int64, 0)])
        # Matching a data type variable
        beqn, deqn = explode_coercion_eqns([(dshape('int32'), dshape('D'))])
        self.assertEqual(beqn, [])
        self.assertEqual(deqn, [(T.int32, T.TypeVar('D'), 0)])

    def test_explode_coercion_eqns_dim(self):
        # Broadcasting a single dimension
        beqn, deqn = explode_coercion_eqns([(dshape('1 * int32'), dshape('10 * int32'))])
        self.assertEqual(beqn, [(T.Fixed(1), T.Fixed(10), 0)])
        self.assertEqual(deqn, [(T.int32, T.int32, 0)])
        # Matching a dim type variable
        beqn, deqn = explode_coercion_eqns([(dshape('3 * int32'), dshape('M * int32'))])
        self.assertEqual(beqn, [(T.Fixed(3), T.TypeVar('M'), 0)])
        self.assertEqual(deqn, [(T.int32, T.int32, 0)])

    def test_explode_coercion_eqns_ellipsis(self):
        # Matching an ellipsis
        beqn, deqn = explode_coercion_eqns([(dshape('int32'), dshape('... * int32'))])
        self.assertEqual(beqn, [([], T.Ellipsis(), 0)])
        self.assertEqual(deqn, [(T.int32, T.int32, 0)])
        beqn, deqn = explode_coercion_eqns([(dshape('3 * int32'), dshape('... * int32'))])
        self.assertEqual(beqn, [([T.Fixed(3)], T.Ellipsis(), 0)])
        self.assertEqual(deqn, [(T.int32, T.int32, 0)])
        beqn, deqn = explode_coercion_eqns([(dshape('3 * var * int32'), dshape('... * int32'))])
        self.assertEqual(beqn, [([T.Fixed(3), T.Var()], T.Ellipsis(), 0)])
        self.assertEqual(deqn, [(T.int32, T.int32, 0)])
        # Matching an ellipsis type variable
        beqn, deqn = explode_coercion_eqns([(dshape('int32'), dshape('A... * int32'))])
        self.assertEqual(beqn, [([], T.Ellipsis(T.TypeVar('A')), 0)])
        self.assertEqual(deqn, [(T.int32, T.int32, 0)])
        beqn, deqn = explode_coercion_eqns([(dshape('3 * int32'), dshape('A... * int32'))])
        self.assertEqual(beqn, [([T.Fixed(3)], T.Ellipsis(T.TypeVar('A')), 0)])
        self.assertEqual(deqn, [(T.int32, T.int32, 0)])
        beqn, deqn = explode_coercion_eqns([(dshape('3 * var * int32'), dshape('A... * int32'))])
        self.assertEqual(beqn, [([T.Fixed(3), T.Var()], T.Ellipsis(T.TypeVar('A')), 0)])
        self.assertEqual(deqn, [(T.int32, T.int32, 0)])
        # Matching an ellipsis with a dim type variable on the left
        beqn, deqn = explode_coercion_eqns([(dshape('3 * var * int32'), dshape('A * B... * int32'))])
        self.assertEqual(beqn, [(T.Fixed(3), T.TypeVar('A'), 0),
                                ([T.Var()], T.Ellipsis(T.TypeVar('B')), 0)])
        self.assertEqual(deqn, [(T.int32, T.int32, 0)])
        # Matching an ellipsis with a dim type variable on the right
        beqn, deqn = explode_coercion_eqns([(dshape('3 * var * int32'), dshape('A... * B * int32'))])
        self.assertEqual(beqn, [([T.Fixed(3)], T.Ellipsis(T.TypeVar('A')), 0),
                                (T.Var(), T.TypeVar('B'), 0)])
        self.assertEqual(deqn, [(T.int32, T.int32, 0)])
        # Matching an ellipsis with a dim type variable on both sides
        beqn, deqn = explode_coercion_eqns([(dshape('3 * var * int32'), dshape('A * B... * C * int32'))])
        self.assertEqual(beqn, [(T.Fixed(3), T.TypeVar('A'), 0),
                                ([], T.Ellipsis(T.TypeVar('B')), 0),
                                (T.Var(), T.TypeVar('C'), 0)])
        self.assertEqual(deqn, [(T.int32, T.int32, 0)])
        beqn, deqn = explode_coercion_eqns([(dshape('3 * var * 4 * M * int32'), dshape('A * B... * C * int32'))])
        self.assertEqual(beqn, [(T.Fixed(3), T.TypeVar('A'), 0),
                                ([T.Var(), T.Fixed(4)], T.Ellipsis(T.TypeVar('B')), 0),
                                (T.TypeVar('M'), T.TypeVar('C'), 0)])
        self.assertEqual(deqn, [(T.int32, T.int32, 0)])
