DataShape Grammar
=================

The datashape language is a DSL which describes the structure of data, abstracted from
a particular implementation in a language or file format. Compared to the Python
library NumPy, it combines `shape` and `dtype` together, and introduces a
syntax for describing structured data.

Some of the basic features include:

* Dimensions are separated by commas.
* Lists of types are separated by semicolons.
* Types and Typevars are distinguished by the capitalization of the leading
  character. Lowercase for types, and uppercase for typevars.
* Type constructors operate using square brackets.

Some Simple Examples
--------------------

Here are some simple examples to motivate the idea::

    # Scalar types
    bool
    int32
    float64

    # Arrays
    3, 4, int32
    10, var, float64
    3, complex[float64]

    # Array of Structures
    100, {
        name: string;
        birthday: date;
        address: {
            street: string;
            city: string;
            postalcode: string;
            country: string;
        };
    }


The DataShape Grammar
---------------------

Dimension Type Symbol Table::

    var : variable dimension

Data Type Symbol Table::

    # Numeric
    bool
    int8
    int16
    int32
    int64
    int128
    uint8
    uint16
    uint32
    uint64
    uint128
    float16
    float32
    float64
    float128

    # Other
    string
    bytes
    date
    json

Data Type Constructor Symbol Table::

    # complex[float32], complex[float64]
    complex
    # string[ascii], string[cp949]
    string
    # bytes[4;2]
    bytes
    # datetime[minutes;CST]
    datetime
    # categorical[string; ['low', 'medium', 'high']]
    categorical

Tokens::

    NAME_LOWER : [a-z][a-zA-Z0-9_]*
    NAME_UPPER : [A-Z][a-zA-Z0-9_]*
    NAME_OTHER : _[a-zA-Z0-9_]*
    COMMA : ,
    SEMI : ;
    ELLIPSIS : \.\.\.
    LBRACKET : \[
    RBRACKET : \]
    LBRACE : \{
    RBRACE : \}
    INTEGER : 0|[1-9][0-9]*
    STRING : (?:"(?:[^"\n\r\\]|(?:\\x[0-9a-fA-F]{2})|(?:\\u[0-9a-fA-F]{4})|(?:\\.))*")|(?:\'(?:[^\'\n\r\\]|(?:\\x[0-9a-fA-F]+)|(?:\\u[0-9a-fA-F]{4})|(?:\\.))*\')


Grammar::

    # Comma-separated list of dimensions, followed by data type
    datashape : dim COMMA datashape
              | dtype

    # Dimension Type (from the dimension type symbol table)
    dim : symbol_type
        | ellipsis_typevar
        | INTEGER


    # Data Type (from the data type symbol table)
    dtype : symbol_type
          | struct_type

    # A type defined by a symbol
    symbol_type : typevar
                | type
                | type_constr

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
                  | type_arg

    # Type Constructor : single argument
    type_arg : datashape
             | INTEGER
             | STRING
             | list_type_arg

    # Type Constructor : single list argument
    list_type_arg : empty_list
                  | LBRACKET datashape_list RBRACKET
                  | LBRACKET integer_list RBRACKET
                  | LBRACKET string_list RBRACKET

    empty_list : LBRACKET RBRACKET

    datashape_list : datashape SEMI datashape_list
                   | datashape

    integer_list : INTEGER SEMI integer_list
                 | INTEGER

    string_list : STRING SEMI string_list
                | STRING


    # Struct/Record type
    struct_type : LBRACE struct_field_list RBRACE

    struct_field_list : struct_field SEMI struct_field_list
                      | struct_field

    struct_field : struct_field_name COLON datashape

    struct_field_name : NAME_LOWER
                      | NAME_UPPER
                      | NAME_OTHER
    

