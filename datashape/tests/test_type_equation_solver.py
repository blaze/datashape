from __future__ import absolute_import, division, print_function

import unittest

from datashape import coretypes as T
from datashape.type_equation_solver import match_argtypes_to_signature, _match_equation
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
        eqns = _match_equation(dshape('int32'), dshape('int64'))
        self.assertEqual(eqns, [(T.int32, T.int64)])
        # Matching a data type variable
        eqns = _match_equation(dshape('int32'), dshape('D'))
        self.assertEqual(eqns, [(T.int32, T.TypeVar('D'))])

    def test_explode_coercion_eqns_dim(self):
        # Broadcasting a single dimension
        eqns = _match_equation(dshape('1 * int32'), dshape('10 * int32'))
        self.assertEqual(eqns, [(T.Fixed(1), T.Fixed(10)),
                                (T.int32, T.int32)])
        # Matching a dim type variable
        eqns = _match_equation(dshape('3 * int32'), dshape('M * int32'))
        self.assertEqual(eqns, [(T.Fixed(3), T.TypeVar('M')),
                                (T.int32, T.int32)])

    def test_explode_coercion_eqns_ellipsis(self):
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
