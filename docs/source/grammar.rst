DataShape Grammar
=================

The datashape language is a DSL which describes the structure of data, abstracted from
a particular implementation in a language or file format. Compared to the Python
library NumPy, it combines `shape` and `dtype` together, and introduces a
syntax for describing structured data.

Some of the basic features include:

* Dimensions are separated by asterisks.

* Lists of types are separated by commas.

* Types and Typevars are distinguished by the capitalization of the leading
  character. Lowercase for types, and uppercase for typevars.

* Type constructors operate using square brackets.

* Type constructors accept positional and keyword arguments,
  and their arguments may be:

  * datashape, string, integer, list of datashape, list of string,
    list of integer

* In multi-line datashape strings or files, comments start from #
  characters to the end of the line.

Some Simple Examples
--------------------

Here are some simple examples to motivate the idea::

    # Scalar types
    bool
    int32
    float64

    # Scalar types with missing data/NA support
    ?bool
    ?float32
    ?complex

    # Arrays
    3 * 4 * int32
    3 * 4 * int32
    10 * var * float64
    3 * complex[float64]

    # Array of Structures
    100 * {
        name: string,
        birthday: date,
        address: {
            street: string,
            city: string,
            postalcode: string,
            country: string
        }
    }

    # Structure of Arrays
    {
        x: 100 * 100 * float32,
        y: 100 * 100 * float32,
        u: 100 * 100 * float32,
        v: 100 * 100 * float32,
    }

    # Structure with strings for field names
    {
        'field 0': 100 * float32,
        'field 1': float32,
        'field 2': float32,
    }

    # Array of Tuples
    20 * (int32, float64)

    # Function prototype
    (3 * int32, float64) -> 3 * float64

    # Function prototype with broadcasting dimensions
    (A... * int32, A... * int32) -> A... * int32

Syntactic Sugar
---------------

Many syntax elements in datashape are syntax sugar for particular
type constructors. For dtypes, this is::

    {x : int32, y : int16}    =>   struct[['x', 'y'], [int32, int16]]
    (int64, float32)          =>   tuple[[int64, float32]]
    (int64, float32) -> bool  =>   funcproto[[int64, float32], bool]
    DTypeVar                  =>   typevar['DTypeVar']
    ?int32                    =>   option[int32]
    2 * ?3 * int32            =>   2 * option[3 * int32]

For dims, this is::

    3 * int32                 =>   fixed[3] * int32
    DimVar * int32            =>   typevar['DimVar'] * int32
    ... * int32               =>   ellipsis * int32
    DimVar... * int32         =>   ellipsis['DimVar'] * int32

The DataShape Grammar
---------------------

Dimension Type Symbol Table::

    # Variable-sized dimension
    var

Dimension Type Constructor Symbol Table::

    # Arrays which are either missing or fully there
    # option[3 * int32]
    option

Data Type Symbol Table::

    # Numeric
    bool
    # Two's complement binary integers
    int8
    int16
    int32
    int64
    int128
    # Unsigned binary integers
    uint8
    uint16
    uint32
    uint64
    uint128
    # IEEE 754-2008 binary### floating point binary numbers
    float16
    float32
    float64
    float128
    # IEEE 754-2008 decimal### floating point decimal numbers
    decimal32
    decimal64
    decimal128
    # Arbitrary precision integer
    bignum
    # Alias for int32
    int
    # Alias for float64
    real
    # Alias for complex[float64]
    complex
    # Alias for int32 or int64 depending on platform
    intptr
    # Alias for uint32 or uint64 depending on platform
    uintptr

    # A unicode string
    string
    # A single unicode code point
    char
    # A blob of bytes
    bytes
    # A date
    date
    # A string containing JSON
    json
    # No data
    void

Data Type Constructor Symbol Table::

    # complex[float32], complex[type=float64]
    complex
    # string['ascii'], string[enc='cp949']
    string
    # bytes[size=4,align=2]
    bytes
    # datetime[unit='minutes',tz='CST']
    datetime
    # categorical[type=string, values=['low', 'medium', 'high']]
    categorical
    # option[float64]
    option
    # pointer[target=2 * 3 * int32]
    pointer

Tokens::

    NAME_LOWER : [a-z][a-zA-Z0-9_]*
    NAME_UPPER : [A-Z][a-zA-Z0-9_]*
    NAME_OTHER : _[a-zA-Z0-9_]*
    ASTERISK : \*
    COMMA : ,
    EQUAL : =
    COLON : :
    LBRACKET : \[
    RBRACKET : \]
    LBRACE : \{
    RBRACE : \}
    LPAREN : \(
    RPAREN : \)
    ELLIPSIS : \.\.\.
    RARROW : ->
    QUESTIONMARK : ?
    INTEGER : 0(?![0-9])|[1-9][0-9]*
    STRING : (?:"(?:[^"\n\r\\]|(?:\\u[0-9a-fA-F]{4})|(?:\\["bfnrt]))*")|(?:\'(?:[^\'\n\r\\]|(?:\\u[0-9a-fA-F]{4})|(?:\\['bfnrt]))*"))*\')


Grammar::

    # Datashape may start with a '?' or not to signal optionality
    datashape : datashape_nooption
              | QUESTIONMARK datashape_nooption

    # Asterisk-separated list of dimensions, followed by data type
    datashape_nooption : dim ASTERISK datashape
                       | dtype

    # Dimension Type (from the dimension type symbol table)
    dim : typevar
        | ellipsis_typevar
        | type
        | type_constr
        | INTEGER
        | ELLIPSIS

    # Data Type (from the data type symbol table)
    dtype : typevar
          | type
          | type_constr
          | struct_type
          | funcproto_or_tuple_type

    # A type variable
    typevar : NAME_UPPER

    # A type variable with ellipsis
    ellipsis_typevar : NAME_UPPER ELLIPSIS

    # A bare type (from the data type symbol table)
    type : NAME_LOWER

    # Type Constructor (from the data type constructor symbol table)
    type_constr : NAME_LOWER LBRACKET type_arg_list RBRACKET

    # Type Constructor: list of arguments
    type_arg_list : type_arg COMMA type_arg_list
                  | type_kwarg_list
                  | type_arg

    # Type Constructor: list of arguments
    type_kwarg_list : type_kwarg COMMA type_kwarg_list
                    | type_kwarg

    # Type Constructor : single argument
    type_arg : datashape
             | INTEGER
             | STRING
             | list_type_arg

    # Type Constructor : single keyword argument
    type_kwarg : NAME_LOWER EQUAL type_arg

    # Type Constructor : single list argument
    list_type_arg : LBRACKET RBRACKET
                  | LBRACKET datashape_list RBRACKET
                  | LBRACKET integer_list RBRACKET
                  | LBRACKET string_list RBRACKET

    datashape_list : datashape COMMA datashape_list
                   | datashape

    integer_list : INTEGER COMMA integer_list
                 | INTEGER

    string_list : STRING COMMA string_list
                | STRING


    # Struct/Record type (allowing for a trailing comma)
    struct_type : LBRACE struct_field_list RBRACE
                | LBRACE struct_field_list COMMA RBRACE

    struct_field_list : struct_field COMMA struct_field_list
                      | struct_field

    struct_field : struct_field_name COLON datashape

    struct_field_name : NAME_LOWER
                      | NAME_UPPER
                      | NAME_OTHER
                      | STRING

    # Function prototype is a tuple with an arrow to the output type
    funcproto_or_tuple_type : tuple_type RARROW datashape
                            | tuple_type
    
    # Tuple type (allowing for a trailing comma)
    tuple_type : LPAREN tuple_item_list RPAREN
               | LPAREN tuple_item_list COMMA RPAREN

    tuple_item_list : datashape COMMA tuple_item_list
                    | datashape

