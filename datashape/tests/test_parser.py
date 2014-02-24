"""
Test the DataShape parser.
"""

from __future__ import absolute_import, division, print_function

import unittest

import datashape
from datashape import parser_redo as parser
from datashape import coretypes

class TestDataShapeParser(unittest.TestCase):
    def setUp(self):
        # Create a default symbol table for the parser to use
        self.sym = datashape.TypeSymbolTable(bare=False)

    def test_basic_bool(self):
        self.assertEqual(parser.parse('bool', self.sym),
                         coretypes.DataShape(coretypes.bool_))

    def test_basic_signed_integers(self):
        self.assertEqual(parser.parse('int8', self.sym),
                         coretypes.DataShape(coretypes.int8))
        self.assertEqual(parser.parse('int16', self.sym),
                         coretypes.DataShape(coretypes.int16))
        self.assertEqual(parser.parse('int32', self.sym),
                         coretypes.DataShape(coretypes.int32))
        self.assertEqual(parser.parse('int64', self.sym),
                         coretypes.DataShape(coretypes.int64))
        #self.assertEqual(parser.parse('int128', self.sym),
        #                 coretypes.DataShape(coretypes.int128))
        self.assertEqual(parser.parse('int', self.sym),
                         coretypes.DataShape(coretypes.int_))
        # 'int' is an alias for 'int32'
        self.assertEqual(parser.parse('int', self.sym),
                         parser.parse('int32', self.sym))
        self.assertEqual(parser.parse('intptr', self.sym),
                         coretypes.DataShape(coretypes.intptr))

    def test_basic_unsigned_integers(self):
        self.assertEqual(parser.parse('uint8', self.sym),
                         coretypes.DataShape(coretypes.uint8))
        self.assertEqual(parser.parse('uint16', self.sym),
                         coretypes.DataShape(coretypes.uint16))
        self.assertEqual(parser.parse('uint32', self.sym),
                         coretypes.DataShape(coretypes.uint32))
        self.assertEqual(parser.parse('uint64', self.sym),
                         coretypes.DataShape(coretypes.uint64))
        #self.assertEqual(parser.parse('uint128', self.sym),
        #                 coretypes.DataShape(coretypes.uint128))
        self.assertEqual(parser.parse('uintptr', self.sym),
                         coretypes.DataShape(coretypes.uintptr))

    def test_basic_float(self):
        #self.assertEqual(parser.parse('float16', self.sym),
        #                 coretypes.DataShape(coretypes.float16))
        self.assertEqual(parser.parse('float32', self.sym),
                         coretypes.DataShape(coretypes.float32))
        self.assertEqual(parser.parse('float64', self.sym),
                         coretypes.DataShape(coretypes.float64))
        #self.assertEqual(parser.parse('float128', self.sym),
        #                 coretypes.DataShape(coretypes.float128))
        self.assertEqual(parser.parse('real', self.sym),
                         coretypes.DataShape(coretypes.real))
        # 'real' is an alias for 'float64'
        self.assertEqual(parser.parse('real', self.sym),
                         parser.parse('float64', self.sym))

    def test_basic_complex(self):
        self.assertEqual(parser.parse('complex[float32]', self.sym),
                         coretypes.DataShape(coretypes.complex_float32))
        self.assertEqual(parser.parse('complex[float64]', self.sym),
                         coretypes.DataShape(coretypes.complex_float64))
        self.assertEqual(parser.parse('complex', self.sym),
                         coretypes.DataShape(coretypes.complex_))
        # 'complex' is an alias for 'complex[float64]'
        self.assertEqual(parser.parse('complex', self.sym),
                         parser.parse('complex[float64]', self.sym))

    def test_basic_raise(self):
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parser.parse, '', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parser.parse, 'boot', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parser.parse, 'int33', self.sym)

if __name__ == '__main__':
    unittest.main()

