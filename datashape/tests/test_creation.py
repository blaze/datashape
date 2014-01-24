import ctypes
import unittest

import datashape
from datashape import dshape, error


class TestDatashapeCreation(unittest.TestCase):

    def test_raise_on_bad_input(self):
        # Make sure it raises exceptions on a few nonsense inputs
        self.assertRaises(TypeError, dshape, None)
        self.assertRaises(TypeError, dshape, lambda x: x+1)
        # Check issue 11
        self.assertRaises(datashape.parser.DatashapeSyntaxError, dshape, '1,')

    def test_reserved_future_int(self):
        # The "int" datashape is reserved for a future big integer type
        self.assertRaises(Exception, dshape, "int")

    def test_atom_shapes(self):
        self.assertEqual(dshape('bool'), dshape(datashape.bool_))
        self.assertEqual(dshape('int8'), dshape(datashape.int8))
        self.assertEqual(dshape('int16'), dshape(datashape.int16))
        self.assertEqual(dshape('int32'), dshape(datashape.int32))
        self.assertEqual(dshape('int64'), dshape(datashape.int64))
        self.assertEqual(dshape('uint8'), dshape(datashape.uint8))
        self.assertEqual(dshape('uint16'), dshape(datashape.uint16))
        self.assertEqual(dshape('uint32'), dshape(datashape.uint32))
        self.assertEqual(dshape('uint64'), dshape(datashape.uint64))
        self.assertEqual(dshape('float32'), dshape(datashape.float32))
        self.assertEqual(dshape('float64'), dshape(datashape.float64))
        self.assertEqual(dshape('complex64'), dshape(datashape.complex64))
        self.assertEqual(dshape('complex128'), dshape(datashape.complex128))
        self.assertEqual(dshape("string"), datashape.string)
        self.assertEqual(dshape("json"), datashape.json)
        if ctypes.sizeof(ctypes.c_void_p) == 4:
            self.assertEqual(dshape('intptr'), dshape(datashape.int32))
            self.assertEqual(dshape('uintptr'), dshape(datashape.uint32))
        else:
            self.assertEqual(dshape('intptr'), dshape(datashape.int64))
            self.assertEqual(dshape('uintptr'), dshape(datashape.uint64))

    @unittest.skip("undefined string reverting to typevars. TODO revert to ocaml ' or single char ")
    def test_atom_shape_errors(self):
        self.assertRaises(TypeError, dshape, 'boot')
        self.assertRaises(TypeError, dshape, 'int33')
        self.assertRaises(TypeError, dshape, '12')
        self.assertRaises(TypeError, dshape, 'var')
        self.assertRaises(TypeError, dshape, 'N')

    def test_constraints_error(self):
        self.assertRaises(error.DataShapeTypeError, dshape,
                          'A : integral, B : numeric')

    def test_ellipsis_error(self):
        self.assertRaises(error.DataShapeTypeError, dshape, 'T, ...')

    def test_type_decl(self):
        self.assertRaises(TypeError, dshape, 'type X T = 3, T')
        self.assertEqual(dshape('3, int32'), dshape('type X = 3, int32'))

    def test_string_atom(self):
        self.assertEqual(dshape('string'), dshape("string('U8')"))
        self.assertEqual(dshape("string('ascii')").encoding, 'A')
        self.assertEqual(dshape("string('A')").encoding, 'A')
        self.assertEqual(dshape("string('utf-8')").encoding, 'U8')
        self.assertEqual(dshape("string('U8')").encoding, 'U8')
        self.assertEqual(dshape("string('utf-16')").encoding, 'U16')
        self.assertEqual(dshape("string('U16')").encoding, 'U16')
        self.assertEqual(dshape("string('utf-32')").encoding, 'U32')
        self.assertEqual(dshape("string('U32')").encoding, 'U32')

    def test_struct_of_array(self):
        self.assertEqual(str(dshape('5, int32')), '5, int32')
        self.assertEqual(str(dshape('{field: 5, int32}')),
                                    '{ field : 5, int32 }')
        self.assertEqual(str(dshape('{field: M, int32}')),
                                    '{ field : M, int32 }')

    def test_ragged_array(self):
        self.assertTrue(isinstance(dshape('3, var, int32')[1], datashape.Var))

    def test_numpy_fields(self):
        import numpy as np
        dt = np.dtype('i4,i8,f8')
        ds = datashape.from_numpy((), dt)
        self.assertEqual(ds.names, ['f0', 'f1', 'f2'])
        self.assertEqual(ds.types,
                         [datashape.int32, datashape.int64, datashape.float64])

if __name__ == '__main__':
    unittest.main()
