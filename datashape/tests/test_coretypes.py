from datashape.coretypes import Record, real
from datashape import dshape, to_numpy_dtype, from_numpy
import numpy as np
import unittest

class TestRecord(unittest.TestCase):
    def setUp(self):
        self.a = Record([('x', int), ('y', int)])
        self.b = Record([('y', int), ('x', int)])

    def test_respects_order(self):
        self.assertNotEqual(self.a, self.b)

    def test_strings(self):
        self.assertEqual(Record([('x', 'real')]), Record([('x', real)]))


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


class TestFromNumPyDtype(object):

    def test_int32(self):
        assert from_numpy((2,), 'int32') == dshape('2 * int32')
        assert from_numpy((2,), 'i4') == dshape('2 * int32')

    def test_struct(self):
        dtype = np.dtype([('x', '<i4'), ('y', '<i4')])
        result = from_numpy((2,), dtype)
        assert result == dshape('2 * {x: int32, y: int32}')

    def test_datetime(self):
        keys = 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as'
        for k in keys:
            assert from_numpy((2,),
                              np.dtype('M8[%s]' % k)) == dshape('2 * datetime')

    def test_date(self):
        for d in ('D', 'M', 'Y', 'W'):
            assert from_numpy((2,),
                              np.dtype('M8[%s]' % d)) == dshape('2 * date')

    def test_ascii_string(self):
        assert (from_numpy((2,), np.dtype('S7')) ==
                dshape('2 * string[7, "ascii"]'))

    def test_string(self):
        assert from_numpy((2,), np.dtype('U7')) == dshape('2 * string[7]')


class TestOther(unittest.TestCase):
    def test_eq(self):
        self.assertEqual(dshape('int'), dshape('int'))
        self.assertNotEqual(dshape('int'), 'apple')

    def test_serializable(self):
        import pickle
        ds = dshape('''{id: int64,
                        name: string,
                        amount: float32,
                        arr: 3 * (int32, string)}''')
        ds2 = pickle.loads(pickle.dumps(ds))

        assert ds == ds2

        assert str(ds) == str(ds2)
