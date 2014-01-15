# dlex.py. This file automatically created by PLY (version 3.4). Don't edit!
_tabversion   = '3.4'
_lextokens    = {'LBRACE': 1, 'STRING': 1, 'SEMI': 1, 'DATA': 1, 'RBRACK': 1, 'EQUALS': 1, 'RBRACE': 1, 'NUMBER': 1, 'LBRACK': 1, 'JSON': 1, 'COLON': 1, 'COMMA': 1, 'ARROW': 1, 'VAR': 1, 'ELLIPSIS': 1, 'BIT': 1, 'TYPE': 1, 'NAME': 1}
_lexreflags   = 0
_lexliterals  = '=|,():{}*\\!'
_lexstateinfo = {'INITIAL': 'inclusive'}
_lexstatere   = {'INITIAL': [('(?P<t_TYPE>type)|(?P<t_newline>\\n+)|(?P<t_JSON>json)|(?P<t_VAR>var)|(?P<t_NAME>[a-zA-Z_][a-zA-Z0-9_]*)|(?P<t_COMMENT>\\#.*)|(?P<t_NUMBER>\\d+)|(?P<t_STRING>(?:"(?:[^"\\n\\r\\\\]|(?:\\\\x[0-9a-fA-F]{2})|(?:\\\\u[0-9a-fA-F]{4})|(?:\\\\.))*")|(?:\\\'(?:[^\\\'\\n\\r\\\\]|(?:\\\\x[0-9a-fA-F]+)|(?:\\\\u[0-9a-fA-F]{4})|(?:\\\\.))*\\\'))|(?P<t_ELLIPSIS>\\.\\.\\.)|(?P<t_LBRACE>\\{)|(?P<t_ARROW>->)|(?P<t_RBRACE>\\})|(?P<t_RBRACK>\\])|(?P<t_LBRACK>\\[)|(?P<t_EQUALS>=)|(?P<t_COMMA>,)|(?P<t_SEMI>;)|(?P<t_COLON>:)', [None, ('t_TYPE', 'TYPE'), ('t_newline', 'newline'), ('t_JSON', 'JSON'), ('t_VAR', 'VAR'), ('t_NAME', 'NAME'), ('t_COMMENT', 'COMMENT'), ('t_NUMBER', 'NUMBER'), ('t_STRING', 'STRING'), (None, 'ELLIPSIS'), (None, 'LBRACE'), (None, 'ARROW'), (None, 'RBRACE'), (None, 'RBRACK'), (None, 'LBRACK'), (None, 'EQUALS'), (None, 'COMMA'), (None, 'SEMI'), (None, 'COLON')])]}
_lexstateignore = {'INITIAL': ' '}
_lexstateerrorf = {'INITIAL': 't_error'}
