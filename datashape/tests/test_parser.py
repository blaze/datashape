"""
Test the DataShape parser.
"""

from __future__ import absolute_import, division, print_function

import unittest

import datashape
from datashape.parser_redo import parse
from datashape import coretypes as T
from datashape import DataShapeSyntaxError

class TestDataShapeParseBasicDType(unittest.TestCase):
    def setUp(self):
        # Create a default symbol table for the parser to use
        self.sym = datashape.TypeSymbolTable()

    def test_bool(self):
        self.assertEqual(parse('bool', self.sym),
                         T.DataShape(T.bool_))

    def test_signed_integers(self):
        self.assertEqual(parse('int8', self.sym),
                         T.DataShape(T.int8))
        self.assertEqual(parse('int16', self.sym),
                         T.DataShape(T.int16))
        self.assertEqual(parse('int32', self.sym),
                         T.DataShape(T.int32))
        self.assertEqual(parse('int64', self.sym),
                         T.DataShape(T.int64))
        #self.assertEqual(parse('int128', self.sym),
        #                 T.DataShape(T.int128))
        self.assertEqual(parse('int', self.sym),
                         T.DataShape(T.int_))
        # 'int' is an alias for 'int32'
        self.assertEqual(parse('int', self.sym),
                         parse('int32', self.sym))
        self.assertEqual(parse('intptr', self.sym),
                         T.DataShape(T.intptr))

    def test_unsigned_integers(self):
        self.assertEqual(parse('uint8', self.sym),
                         T.DataShape(T.uint8))
        self.assertEqual(parse('uint16', self.sym),
                         T.DataShape(T.uint16))
        self.assertEqual(parse('uint32', self.sym),
                         T.DataShape(T.uint32))
        self.assertEqual(parse('uint64', self.sym),
                         T.DataShape(T.uint64))
        #self.assertEqual(parse('uint128', self.sym),
        #                 T.DataShape(T.uint128))
        self.assertEqual(parse('uintptr', self.sym),
                         T.DataShape(T.uintptr))

    def test_float(self):
        #self.assertEqual(parse('float16', self.sym),
        #                 T.DataShape(T.float16))
        self.assertEqual(parse('float32', self.sym),
                         T.DataShape(T.float32))
        self.assertEqual(parse('float64', self.sym),
                         T.DataShape(T.float64))
        #self.assertEqual(parse('float128', self.sym),
        #                 T.DataShape(T.float128))
        self.assertEqual(parse('real', self.sym),
                         T.DataShape(T.real))
        # 'real' is an alias for 'float64'
        self.assertEqual(parse('real', self.sym),
                         parse('float64', self.sym))

    def test_complex(self):
        self.assertEqual(parse('complex[float32]', self.sym),
                         T.DataShape(T.complex_float32))
        self.assertEqual(parse('complex[float64]', self.sym),
                         T.DataShape(T.complex_float64))
        self.assertEqual(parse('complex', self.sym),
                         T.DataShape(T.complex_))
        # 'complex' is an alias for 'complex[float64]'
        self.assertEqual(parse('complex', self.sym),
                         parse('complex[float64]', self.sym))

    def test_raise(self):
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse, '', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse, 'boot', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse, 'int33', self.sym)

class TestDataShapeParserDTypeConstr(unittest.TestCase):
    def test_unary_dtype_constr(self):
        # Create a symbol table with no types in it, so we can
        # make some isolated type constructors for testing
        sym = datashape.TypeSymbolTable(bare=True)
        # A limited set of dtypes for testing
        sym.dtype['int8'] = T.int8
        sym.dtype['uint16'] = T.uint16
        sym.dtype['float64'] = T.float64
        # TypeVar type constructor
        sym.dtype_constr['typevar'] = T.TypeVar
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
            self.assertEqual(parse(ds_str, sym), T.DataShape(T.float32))
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

    def test_binary_dtype_constr(self):
        # Create a symbol table with no types in it, so we can
        # make some isolated type constructors for testing
        sym = datashape.TypeSymbolTable(bare=True)
        # A limited set of dtypes for testing
        sym.dtype['int8'] = T.int8
        sym.dtype['uint16'] = T.uint16
        sym.dtype['float64'] = T.float64
        # TypeVar type constructor
        sym.dtype_constr['typevar'] = T.TypeVar
        # Binary dtype constructor that asserts on the argument values
        expected_arg = [None, None]
        def _binary_type_constr(a, b):
            self.assertEqual(a, expected_arg[0])
            self.assertEqual(b, expected_arg[1])
            expected_arg[0] = None
            expected_arg[1] = None
            return T.float32
        sym.dtype_constr['binary'] = _binary_type_constr

        def assertExpectedParse(ds_str, expected_a, expected_b):
            # Set the expected value, and call the parser
            expected_arg[0] = expected_a
            expected_arg[1] = expected_b
            self.assertEqual(parse(ds_str, sym), T.DataShape(T.float32))
            # Make sure the expected value was actually run by
            # check that it reset the expected value to None
            self.assertEqual(expected_arg, [None, None],
                             'The test binary type constructor did not run')

        # Positional args
        assertExpectedParse('binary[1, 0]', 1, 0)
        assertExpectedParse('binary[0, "test"]', 0, 'test')
        assertExpectedParse('binary[int8, "test"]',
                            T.DataShape(T.int8), 'test')
        assertExpectedParse('binary[[1,3,5], "test"]', [1, 3, 5], 'test')
        # Positional and keyword args
        assertExpectedParse('binary[0, b=1]', 0, 1)
        assertExpectedParse('binary["test", b=A]', 'test',
                            T.DataShape(T.TypeVar('A')))
        assertExpectedParse('binary[[3, 6], b=int8]', [3, 6],
                            T.DataShape(T.int8))
        assertExpectedParse('binary[Arg, b=["x", "test"]]',
                            T.DataShape(T.TypeVar('Arg')), ['x', 'test'])
        # Keyword args
        assertExpectedParse('binary[a=1, b=0]', 1, 0)
        assertExpectedParse('binary[a=[int8, A, uint16], b="x"]',
                            [T.DataShape(T.int8),
                             T.DataShape(T.TypeVar('A')),
                             T.DataShape(T.uint16)],
                            'x')

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

        # Require closing "]"
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[', sym)
        # Type constructors should always have an argument
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[]', sym)
        # Unknown type
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[unknown]', sym)
        # Missing parameter value
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[x=', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[x=]', sym)
        # A positional arg cannot be after a keyword arg
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[x=A, B]', sym)
        # List args must be homogeneous
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[[0, "x"]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[[0, X]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[["x", 0]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[["x", X]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[[X, 0]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[[X, "x"]]', sym)

class TestDataShapeParserDims(unittest.TestCase):
    def setUp(self):
        # Create a default symbol table for the parser to use
        self.sym = datashape.TypeSymbolTable()

    def test_fixed_dims(self):
        self.assertEqual(parse('3 * bool', self.sym),
                         T.DataShape(T.Fixed(3), T.bool_))
        self.assertEqual(parse('7 * 3 * bool', self.sym),
                         T.DataShape(T.Fixed(7), T.Fixed(3), T.bool_))
        self.assertEqual(parse('5 * 3 * 12 * bool', self.sym),
                         T.DataShape(T.Fixed(5), T.Fixed(3),
                                     T.Fixed(12), T.bool_))
        self.assertEqual(parse('2 * 3 * 4 * 5 * bool', self.sym),
                         T.DataShape(T.Fixed(2), T.Fixed(3),
                                     T.Fixed(4), T.Fixed(5), T.bool_))

    def test_typevar_dims(self):
        self.assertEqual(parse('M * bool', self.sym),
                         T.DataShape(T.TypeVar('M'), T.bool_))
        self.assertEqual(parse('A * B * bool', self.sym),
                         T.DataShape(T.TypeVar('A'), T.TypeVar('B'), T.bool_))
        self.assertEqual(parse('A... * X * 3 * bool', self.sym),
                         T.DataShape(T.Ellipsis(T.TypeVar('A')), T.TypeVar('X'),
                                     T.Fixed(3), T.bool_))

    def test_var_dims(self):
        self.assertEqual(parse('var * bool', self.sym),
                         T.DataShape(T.Var(), T.bool_))
        self.assertEqual(parse('var * var * bool', self.sym),
                         T.DataShape(T.Var(), T.Var(), T.bool_))
        self.assertEqual(parse('M * 5 * var * bool', self.sym),
                         T.DataShape(T.TypeVar('M'), T.Fixed(5), T.Var(), T.bool_))


class TestDataShapeParseStruct(unittest.TestCase):
    def setUp(self):
        # Create a default symbol table for the parser to use
        self.sym = datashape.TypeSymbolTable()

    def test_struct(self):
        # Simple struct
        self.assertEqual(parse('{x: int16, y: int32}', self.sym),
                         T.DataShape(T.Record([('x', T.DataShape(T.int16)),
                                               ('y', T.DataShape(T.int32))])))
        # A trailing comma is ok
        self.assertEqual(parse('{x: int16, y: int32,}', self.sym),
                         T.DataShape(T.Record([('x', T.DataShape(T.int16)),
                                               ('y', T.DataShape(T.int32))])))
        # Field names starting with _ and caps
        self.assertEqual(parse('{_x: int16, Zed: int32,}', self.sym),
                         T.DataShape(T.Record([('_x', T.DataShape(T.int16)),
                                               ('Zed', T.DataShape(T.int32))])))
        # A slightly bigger example
        ds_str = """3 * var * {
                        id : int32,
                        name : string,
                        description : {
                            language : string,
                            text : string
                        },
                        entries : var * {
                            date : date,
                            text : string
                        }
                    }"""
        int32 = T.DataShape(T.int32)
        string = T.DataShape(T.string)
        date = T.DataShape(T.date)
        ds = (T.Fixed(3), T.Var(),
              T.Record([('id', int32),
                        ('name', string),
                        ('description', T.DataShape(T.Record([('language', string),
                                                              ('text', string)]))),
                        ('entries', T.DataShape(T.Var(),
                                                T.Record([('date', date),
                                                          ('text', string)])))]))
        self.assertEqual(parse(ds_str, self.sym), T.DataShape(*ds))

    def test_fields_with_dshape_names(self):
        # Should be able to name a field 'type', 'int64', etc
        ds = parse("""{
                type: bool,
                data: bool,
                blob: bool,
                bool: bool,
                int: int32,
                float: float32,
                double: float64,
                int8: int8,
                int16: int16,
                int32: int32,
                int64: int64,
                uint8: uint8,
                uint16: uint16,
                uint32: uint32,
                uint64: uint64,
                float16: float32,
                float32: float32,
                float64: float64,
                float128: float64,
                complex: float32,
                complex64: float32,
                complex128: float64,
                string: string,
                object: string,
                datetime: string,
                datetime64: string,
                timedelta: string,
                timedelta64: string,
                json: string,
                var: string,
            }""", self.sym)
        self.assertEqual(type(ds[-1]), T.Record)
        self.assertEqual(len(ds[-1].names), 30)

    def test_kiva_datashape(self):
        # A slightly more complicated datashape which should parse
        ds = parse("""5 * var * {
              id: int64,
              name: string,
              description: {
                languages: var * string[2],
                texts: json,
              },
              status: string,
              funded_amount: float64,
              basket_amount: json,
              paid_amount: json,
              image: {
                id: int64,
                template_id: int64,
              },
              video: json,
              activity: string,
              sector: string,
              use: string,
              delinquent: bool,
              location: {
                country_code: string[2],
                country: string,
                town: json,
                geo: {
                  level: string,
                  pairs: string,
                  type: string,
                },
              },
              partner_id: int64,
              posted_date: json,
              planned_expiration_date: json,
              loan_amount: float64,
              currency_exchange_loss_amount: json,
              borrowers: var * {
                first_name: string,
                last_name: string,
                gender: string[1],
                pictured: bool,
              },
              terms: {
                disbursal_date: json,
                disbursal_currency: string[3,'A'],
                disbursal_amount: float64,
                loan_amount: float64,
                local_payments: var * {
                  due_date: json,
                  amount: float64,
                },
                scheduled_payments: var * {
                  due_date: json,
                  amount: float64,
                },
                loss_liability: {
                  nonpayment: string,
                  currency_exchange: string,
                  currency_exchange_coverage_rate: json,
                },
              },
              payments: var * {
                amount: float64,
                local_amount: float64,
                processed_date: json,
                settlement_date: json,
                rounded_local_amount: float64,
                currency_exchange_loss_amount: float64,
                payment_id: int64,
                comment: json,
              },
              funded_date: json,
              paid_date: json,
              journal_totals: {
                entries: int64,
                bulkEntries: int64,
              },
            }
        """, self.sym)
        self.assertEqual(type(ds[-1]), T.Record)
        self.assertEqual(len(ds[-1].names), 25)

    def test_struct_errors(self):
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse, '{\n}\n', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse,
                          '{id: int64, name: string amount: invalidtype}',
                          self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse,
                          '{id: int64, name: string, amount: invalidtype}',
                          self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse,
                          '{id: int64, name: string, amount: %}',
                          self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse,
                          "{\n" +
                          "   id: int64;\n" +
                          "   name: string;\n" +
                          "   amount+ float32;\n" +
                          "}\n",
                          self.sym)


class TestDataShapeParseTuple(unittest.TestCase):
    def setUp(self):
        # Create a default symbol table for the parser to use
        self.sym = datashape.TypeSymbolTable()

    def test_tuple(self):
        # Simple tuple
        self.assertEqual(parse('(float32)', self.sym),
                         T.DataShape(T.Tuple([T.DataShape(T.float32)])))
        self.assertEqual(parse('(int16, int32)', self.sym),
                         T.DataShape(T.Tuple([T.DataShape(T.int16),
                                              T.DataShape(T.int32)])))
        # A trailing comma is ok
        self.assertEqual(parse('(float32,)', self.sym),
                         T.DataShape(T.Tuple([T.DataShape(T.float32)])))
        self.assertEqual(parse('(int16, int32,)', self.sym),
                         T.DataShape(T.Tuple([T.DataShape(T.int16),
                                              T.DataShape(T.int32)])))


class TestDataShapeParseFuncProto(unittest.TestCase):
    def setUp(self):
        # Create a default symbol table for the parser to use
        self.sym = datashape.TypeSymbolTable()

    def test_funcproto(self):
        # Simple funcproto
        self.assertEqual(parse('(float32) -> float64', self.sym),
                         T.DataShape(T.Function(T.DataShape(T.float32),
                                                T.DataShape(T.float64))))
        self.assertEqual(parse('(int16, int32) -> bool', self.sym),
                         T.DataShape(T.Function(T.DataShape(T.int16),
                                                T.DataShape(T.int32),
                                                T.DataShape(T.bool_))))
        # A trailing comma is ok
        self.assertEqual(parse('(float32,) -> float64', self.sym),
                         T.DataShape(T.Function(T.DataShape(T.float32),
                                                T.DataShape(T.float64))))
        self.assertEqual(parse('(int16, int32,) -> bool', self.sym),
                         T.DataShape(T.Function(T.DataShape(T.int16),
                                                T.DataShape(T.int32),
                                                T.DataShape(T.bool_))))

if __name__ == '__main__':
    unittest.main()
