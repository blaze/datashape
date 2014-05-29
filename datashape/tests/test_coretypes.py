from datashape.coretypes import Record
from datashape import dshape, to_numpy_dtype
import numpy as np
import unittest

class TestRecord(unittest.TestCase):
    def setUp(self):
        self.a = Record([('x', int), ('y', int)])
        self.b = Record([('y', int), ('x', int)])

    def test_respects_order(self):
        self.assertNotEqual(self.a, self.b)

class Test_to_numpy_dtype(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(to_numpy_dtype(dshape('2 * int32')), np.int32)
        self.assertEqual(to_numpy_dtype(dshape('2 * {x: int32, y: int32}')),
                         np.dtype([('x', '<i4'), ('y', '<i4')]))

    def test_datetime(self):
        self.assertEqual(to_numpy_dtype(dshape('2 * datetime')),
                         np.dtype('M8[us]'))

    def test_date(self):
        self.assertEqual(to_numpy_dtype(dshape('2 * date')),
                         np.dtype('M8[D]'))

    def test_string(self):
        self.assertEqual(to_numpy_dtype(dshape('2 * string')),
                         np.dtype('O'))

class TestOther(unittest.TestCase):
    def test_eq(self):
        self.assertEqual(dshape('int'), dshape('int'))
        self.assertNotEqual(dshape('int'), 'apple')
