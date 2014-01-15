DataShape Grammar
=================

The DataShape grammar.

Grammar::

    top : mod

    mod : mod mod
        | stmt

    stmt : TYPE lhs_expression EQUALS rhs_expression
         | rhs_expression

    lhs_expression : lhs_expression lhs_expression
                   | NAME

    rhs_expression : rhs_expression_list

    rhs_expression_list : rhs_expression_list COMMA rhs_expression_list
                   | appl
                   | record
                   | BIT
                   | NAME
                   | NUMBER

    appl_args : appl_args COMMA appl_args
              | appl
              | record
              | '(' rhs_expression ')'
              | BIT
              | NAME
              | NUMBER
              | STRING

    appl : NAME '(' appl_args ')'

    record : LBRACE record_opt RBRACE
    record_opt : record_opt SEMI record_opt
               | record_item
               | empty
    record_name : NAME
                | BIT
                | TYPE
    record_item : record_name COLON '(' rhs_expression ')'
                | record_name COLON rhs_expression'

    empty :
