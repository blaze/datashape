from __future__ import absolute_import, division, print_function

import unittest

from datashape import coercion_cost, dshape, dshapes, error
from datashape.tests import common
from datashape.py2help import xfail


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
        a, b, c = dshapes('int16', 'float64', 'complex[float32]')
        self.assertLess(coercion_cost(a, b), coercion_cost(a, c))
        a, b, c = dshapes('int8', 'float64', 'complex[float32]')
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

    @xfail(reason='This is something that needs to be handled by overloading')
    def test_coerce_typevars(self):
        a, b, c = dshapes('10 * 11 * float32', 'X * Y * float64',
                          '10 * Y * float64')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    @xfail(reason='This is something that needs to be handled by overloading')
    def test_coerce_constrained_typevars(self):
        a, b, c = dshapes('10 * 10 * float32', 'X * Y * float64',
                          'X * X * float64')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    def test_coerce_broadcasting(self):
        a, b, c = dshapes('10 * 10 * float32', '10 * Y * Z * float64',
                          'X * Y * float64')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    def test_coerce_broadcasting2(self):
        a, b, c = dshapes('10 * 10 * float32', '1 * 10 * 10 * float32',
                          '10 * 10 * float32')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    def test_coerce_broadcasting3(self):
        a, b, c = dshapes('10 * 10 * float32', '10 * 10 * 10 * float32',
                          '1 * 10 * 10 * float32')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    @xfail(reason='implements has not been implemented in the new parser')
    def test_coerce_traits(self):
        a, b, c = dshapes('10 * 10 * float32', '10 * X * A : floating',
                          '10 * X * float32')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    def test_coerce_dst_ellipsis(self):
        a, b, c = dshapes('10 * 10 * float32', 'X * ... * float64',
                          'X * Y * float64')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    @xfail(reason='not dealing with ellipsis in the src of a coercion')
    def test_coerce_src_ellipsis(self):
        a, b, c = dshapes('10 * ... * float32', 'X * Y * float64',
                          'X * ... * float64')
        self.assertGreater(coercion_cost(a, b), coercion_cost(a, c))

    def test_allow_anything_to_bool(self):
        # The cost should be large
        min_cost = coercion_cost(dshape('int8'), dshape('complex[float64]'))
        for ds in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                   'uint32', 'uint64', 'float32', 'float64',
                   'complex[float32]', 'complex[float64]']:
            self.assertGreater(coercion_cost(dshape(ds), dshape('bool')),
                               min_cost)

class TestCoercionErrors(unittest.TestCase):

    def test_downcast(self):
        a, b = dshapes('float32', 'int32')
        self.assertRaises(error.CoercionError, coercion_cost, a, b)

    def test_disallow_bool_to_anything(self):
        for ds in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                   'uint32', 'uint64', 'float32', 'float64',
                   'complex[float32]', 'complex[float64]']:
            self.assertRaises(error.CoercionError, coercion_cost,
                              dshape('bool'), dshape(ds))


if __name__ == '__main__':
    unittest.main()
