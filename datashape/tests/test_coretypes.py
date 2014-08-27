from datashape.coretypes import Record, real, int64, datetime, float64
from datashape import dshape, to_numpy_dtype
import numpy as np
import unittest

class TestRecord(unittest.TestCase):
    def setUp(self):
        self.a = Record([('x', int), ('y', int)])
        self.b = Record([('y', int), ('x', int)])
        #Record.__init__ should escape all of the quotes, spaces, and other special characters, turning them into "_"s
        self.c = Record([('Unique Key', int64), ('Creation Date', datetime)])
        self.d = Record([('Unique Field(foo)', float64), ("Unique Field (foo2)", int)])
        self.e = Record([('"Unique Field(foo)"', float64), ("'Unique Field (foo2)'", int)])

    def test_respects_order(self):
        self.assertNotEqual(self.a, self.b)

    def test_strings(self):
        self.assertEqual(Record([('x', 'real')]), Record([('x', real)]))

    def test_name_normalization(self):
        assert("Unique_Key" in self.c.names)
        assert("Creation_Date" in self.c.names)
        assert("Unique_Field_foo_" in self.d.names)
        assert("Unique_Field__foo2_" in self.d.names)
        assert("_Unique_Field_foo__" in self.e.names)
        assert("_Unique_Field__foo2__" in self.e.names)
                        

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

    def test_serializable(self):
        import pickle
        ds = dshape('''{id: int64,
                        name: string,
                        amount: float32,
                        arr: 3 * (int32, string)}''')
        ds2 = pickle.loads(pickle.dumps(ds))

        assert ds == ds2

        assert str(ds) == str(ds2)
