# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from datashape import coercion_cost, dshapes, error
from datashape.tests import common


class TestCoercion(common.BTestCase):

    def test_coerce_ctype(self):
        a, b, c = dshapes('float32', 'float32', 'float64')
        self.assertLess(coercion_cost(a, b), coercion_cost(a, c))
        a, b, c = dshapes('uint64', 'uint64', 'int64')
        self.assertLess(coercion_cost(a, b), coercion_cost(a, c))
        a, b, c = dshapes('int64', 'int64', 'uint64')
        self.assertLess(coercion_cost(a, b), coercion_cost(a, c))
        a, b, c = dshapes('float64', 'float64', 'complex[float32]')
        self.assertLess(coercion_cost(a, b), coercion_cost(a, c))

    def test_coerce_ctype_float_vs_complex(self):
        # int -> float32 is preferred over int -> complex[float32]
        a, b, c = dshapes('int32', 'float32', 'complex[float32]')
        self.assertLess(coercion_cost(a, b), coercion_cost(a, c))
        # int -> float64 is preferred over int -> complex[float64]
        a, b, c = dshapes('int32', 'float64', 'complex[float64]')
        self.assertLess(coercion_cost(a, b), coercion_cost(a, c))
        # int -> float64 is preferred over int -> complex[float32]
        a, b, c = dshapes('int32', 'float64', 'complex[float32]')
        self.assertLess(coercion_cost(a, b), coercion_cost(a, c))

    def test_coerce_numeric(self):
        a, b = dshapes('float32', 'float64')
        self.assertGreater(coercion_cost(a, b), 0)

    def test_coercion_transitivity(self):
        a, b, c = dshapes('int8', 'complex128', 'float64')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    def test_coerce_typevars(self):
        a, b, c = dshapes('10, 11, float32', 'X, Y, float64', '10, Y, float64')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    def test_coerce_constrained_typevars(self):
        a, b, c = dshapes('10, 10, float32', 'X, Y, float64', 'X, X, float64')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    def test_coerce_broadcasting(self):
        a, b, c = dshapes('10, 10, float32', '10, Y, Z, float64', 'X, Y, float64')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    def test_coerce_broadcasting2(self):
        a, b, c = dshapes('10, 10, float32', '1, 10, 10, float32', '10, 10, float32')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    def test_coerce_broadcasting3(self):
        a, b, c = dshapes('10, 10, float32', '10, 10, 10, float32', '1, 10, 10, float32')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    def test_coerce_traits(self):
        a, b, c = dshapes('10, 10, float32', '10, X, A : floating', '10, X, float32')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    def test_coerce_dst_ellipsis(self):
        a, b, c = dshapes('10, 10, float32', 'X, ..., float64', 'X, Y, float64')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    def test_coerce_src_ellipsis(self):
        a, b, c = dshapes('10, ..., float32', 'X, Y, float64', 'X, ..., float64')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))


class TestCoercionErrors(unittest.TestCase):

    def test_downcast(self):
        a, b = dshapes('float32', 'int32')
        self.assertRaises(error.CoercionError, coercion_cost, a, b)


if __name__ == '__main__':
    unittest.main()
