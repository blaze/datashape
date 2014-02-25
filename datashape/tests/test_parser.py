"""
Test the DataShape parser.
"""

from __future__ import absolute_import, division, print_function

import unittest

import datashape
from datashape import parser_redo as parser
from datashape import coretypes as T
from datashape import DataShapeSyntaxError

class TestDataShapeParserBasicDType(unittest.TestCase):
    def setUp(self):
        # Create a default symbol table for the parser to use
        self.sym = datashape.TypeSymbolTable()

    def test_bool(self):
        self.assertEqual(parser.parse('bool', self.sym),
                         T.DataShape(T.bool_))

    def test_signed_integers(self):
        self.assertEqual(parser.parse('int8', self.sym),
                         T.DataShape(T.int8))
        self.assertEqual(parser.parse('int16', self.sym),
                         T.DataShape(T.int16))
        self.assertEqual(parser.parse('int32', self.sym),
                         T.DataShape(T.int32))
        self.assertEqual(parser.parse('int64', self.sym),
                         T.DataShape(T.int64))
        #self.assertEqual(parser.parse('int128', self.sym),
        #                 T.DataShape(T.int128))
        self.assertEqual(parser.parse('int', self.sym),
                         T.DataShape(T.int_))
        # 'int' is an alias for 'int32'
        self.assertEqual(parser.parse('int', self.sym),
                         parser.parse('int32', self.sym))
        self.assertEqual(parser.parse('intptr', self.sym),
                         T.DataShape(T.intptr))

    def test_unsigned_integers(self):
        self.assertEqual(parser.parse('uint8', self.sym),
                         T.DataShape(T.uint8))
        self.assertEqual(parser.parse('uint16', self.sym),
                         T.DataShape(T.uint16))
        self.assertEqual(parser.parse('uint32', self.sym),
                         T.DataShape(T.uint32))
        self.assertEqual(parser.parse('uint64', self.sym),
                         T.DataShape(T.uint64))
        #self.assertEqual(parser.parse('uint128', self.sym),
        #                 T.DataShape(T.uint128))
        self.assertEqual(parser.parse('uintptr', self.sym),
                         T.DataShape(T.uintptr))

    def test_float(self):
        #self.assertEqual(parser.parse('float16', self.sym),
        #                 T.DataShape(T.float16))
        self.assertEqual(parser.parse('float32', self.sym),
                         T.DataShape(T.float32))
        self.assertEqual(parser.parse('float64', self.sym),
                         T.DataShape(T.float64))
        #self.assertEqual(parser.parse('float128', self.sym),
        #                 T.DataShape(T.float128))
        self.assertEqual(parser.parse('real', self.sym),
                         T.DataShape(T.real))
        # 'real' is an alias for 'float64'
        self.assertEqual(parser.parse('real', self.sym),
                         parser.parse('float64', self.sym))

    def test_complex(self):
        self.assertEqual(parser.parse('complex[float32]', self.sym),
                         T.DataShape(T.complex_float32))
        self.assertEqual(parser.parse('complex[float64]', self.sym),
                         T.DataShape(T.complex_float64))
        self.assertEqual(parser.parse('complex', self.sym),
                         T.DataShape(T.complex_))
        # 'complex' is an alias for 'complex[float64]'
        self.assertEqual(parser.parse('complex', self.sym),
                         parser.parse('complex[float64]', self.sym))

    def test_raise(self):
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parser.parse, '', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parser.parse, 'boot', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parser.parse, 'int33', self.sym)

class TestDataShapeParserDTypeConstr(unittest.TestCase):
    def test_unary_dtype_constr(self):
        # Create a symbol table with no types in it, so we can
        # make some isolated type constructors for testing
        sym = datashape.TypeSymbolTable(bare=True)
        # A limited set of dtypes for testing
        sym.dtype['int8'] = T.int8
        sym.dtype['uint16'] = T.uint16
        sym.dtype['float64'] = T.float64
        # Unary dtype constructor that asserts on the argument value
        expected_blah = [None]
        def _unary_type_constr(blah):
            self.assertEqual(blah, expected_blah[0])
            expected_blah[0] = None
            return T.float32
        sym.dtype_constr['unary'] = _unary_type_constr

        def assertExpectedParse(ds_str, expected):
            # Set the expected value, and call the parser
            expected_blah[0] = expected
            self.assertEqual(parser.parse(ds_str, sym), T.DataShape(T.float32))
            # Make sure the expected value was actually run by
            # check that it reset the expected value to None
            self.assertEqual(expected_blah[0], None,
                             'The test unary type constructor did not run')

        # Integer parameter (positional)
        assertExpectedParse('unary[0]', 0)
        assertExpectedParse('unary[100000]', 100000)
        # String parameter (positional)
        assertExpectedParse('unary["test"]', 'test')
        assertExpectedParse("unary['test']", 'test')
        assertExpectedParse('unary["\\uc548\\ub155"]', u'\uc548\ub155')
        assertExpectedParse(u'unary["\uc548\ub155"]', u'\uc548\ub155')
        # DataShape parameter (positional)
        assertExpectedParse('unary[int8]', T.DataShape(T.int8))
        assertExpectedParse('unary[X]', T.DataShape(T.TypeVar('X')))
        # Empty list parameter (positional)
        assertExpectedParse('unary[[]]', [])
        # List of integers parameter (positional)
        assertExpectedParse('unary[[0, 3, 12]]', [0, 3, 12])
        # List of strings parameter (positional)
        assertExpectedParse('unary[["test", "one", "two"]]',
                            ["test", "one", "two"])
        # List of datashapes parameter (positional)
        assertExpectedParse('unary[[float64, int8, uint16]]',
                            [T.DataShape(T.float64), T.DataShape(T.int8),
                             T.DataShape(T.uint16)])

        # Integer parameter (keyword)
        assertExpectedParse('unary[blah=0]', 0)
        assertExpectedParse('unary[blah=100000]', 100000)
        # String parameter (keyword)
        assertExpectedParse('unary[blah="test"]', 'test')
        assertExpectedParse("unary[blah='test']", 'test')
        assertExpectedParse('unary[blah="\\uc548\\ub155"]', u'\uc548\ub155')
        assertExpectedParse(u'unary[blah="\uc548\ub155"]', u'\uc548\ub155')
        # DataShape parameter (keyword)
        assertExpectedParse('unary[blah=int8]', T.DataShape(T.int8))
        assertExpectedParse('unary[blah=X]', T.DataShape(T.TypeVar('X')))
        # Empty list parameter (keyword)
        assertExpectedParse('unary[blah=[]]', [])
        # List of integers parameter (keyword)
        assertExpectedParse('unary[blah=[0, 3, 12]]', [0, 3, 12])
        # List of strings parameter (keyword)
        assertExpectedParse('unary[blah=["test", "one", "two"]]',
                            ["test", "one", "two"])
        # List of datashapes parameter (keyword)
        assertExpectedParse('unary[blah=[float64, int8, uint16]]',
                            [T.DataShape(T.float64), T.DataShape(T.int8),
                             T.DataShape(T.uint16)])

    def test_dtype_constr_errors(self):
        # Create a symbol table with no types in it, so we can
        # make some isolated type constructors for testing
        sym = datashape.TypeSymbolTable(bare=True)
        # A limited set of dtypes for testing
        sym.dtype['int8'] = T.int8
        sym.dtype['uint16'] = T.uint16
        sym.dtype['float64'] = T.float64
        # Arbitrary dtype constructor that does nothing
        def _type_constr(*args, **kwargs):
            return T.float32
        sym.dtype_constr['tcon'] = _type_constr

        # Require closeing "]"
        self.assertRaises(DataShapeSyntaxError,
                          parser.parse, 'tcon[', sym)
        # Type constructors should always have an argument
        self.assertRaises(DataShapeSyntaxError,
                          parser.parse, 'tcon[]', sym)
        # Unknown type
        self.assertRaises(DataShapeSyntaxError,
                          parser.parse, 'tcon[unknown]', sym)
        # Missing parameter value
        self.assertRaises(DataShapeSyntaxError,
                          parser.parse, 'tcon[x=', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parser.parse, 'tcon[x=]', sym)
        # A positional arg cannot be after a keyword arg
        self.assertRaises(DataShapeSyntaxError,
                          parser.parse, 'tcon[x=A, B]', sym)
        # List args must be homogeneous
        self.assertRaises(DataShapeSyntaxError,
                          parser.parse, 'tcon[[0, "x"]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parser.parse, 'tcon[[0, X]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parser.parse, 'tcon[["x", 0]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parser.parse, 'tcon[["x", X]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parser.parse, 'tcon[[X, 0]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parser.parse, 'tcon[[X, "x"]]', sym)

if __name__ == '__main__':
    unittest.main()
