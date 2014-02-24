"""
Parser for the datashape grammar.
"""

from __future__ import absolute_import, division, print_function

from .. import error, coretypes, lexer

__all__ = ['parse']

class DataShapeParser(object):
    """A DataShape parser object."""
    def __init__(self, ds_str, sym):
        # The datashape string being parsed
        self.ds_str = ds_str
        # Symbol tables for dimensions, dtypes, and type constructors for each
        self.sym = sym
        # The lexer
        self.lex = lexer.lex(ds_str)
        # The array of tokens self.lex has already produced
        self.tokens = []
        # The token currently being examined, and
        # the end position, set when self.lex is exhausted
        self.pos = -1
        self.end_pos = None
        # Advance to the first token
        self.advance_tok()

    def advance_tok(self):
        """Advances self.pos by one, if it is not already at the end."""
        if self.pos != self.end_pos:
            self.pos = self.pos + 1
            try:
                # If self.pos has not been backtracked,
                # we need to request a new token from the lexer
                if self.pos >= len(self.tokens):
                    self.tokens.append(next(self.lex))
            except StopIteration:
                # Create an EOF token, whose span starts at the
                # end of the last token to use for error messages
                if len(self.tokens) > 0:
                    span = (self.tokens[self.pos-1].span[1],)*2
                else:
                    span = (0, 0)
                self.tokens.append(lexer.Token(None, None, span, None))
                self.end_pos = self.pos

    @property
    def tok(self):
        return self.tokens[self.pos]

    def parse_datashape(self):
        """
        datashape : dim ASTERISK datashape
                  | dtype

        Returns a datashape object or None.
        """
        print('parse_datashape', self.tok)
        saved_pos = self.pos
        # Parse zero or more "dim ASTERISK" repetitions
        dims = []
        dim = True
        while dim is not None:
            dim_saved_pos = self.pos
            # Parse the dim
            dim = self.parse_dim()
            if dim is not None:
                if self.tok.id == lexer.ASTERISK:
                    # If an asterisk is next, we're good
                    self.advance_tok()
                    dims.append(dim)
                else:
                    # Otherwise try a dtype
                    dim = None
                    self.pos = dim_saved_pos
        # Parse the dtype
        dtype = self.parse_dtype()
        if dtype:
            return coretypes.DataShape(*(dims + [dtype]))
        else:
            if len(dims) > 0:
                # If we already saw "dim ASTERISK" at least once,
                # we can point at the more specific position within
                # the datashape where the error occurred
                raise error.DataShapeSyntaxError(self.tok.span[0], '<nofile>',
                                                 self.ds_str,
                                                 'Expected a dim or a dtype')
            else:
                self.pos = saved_pos
                return None

    def parse_dim(self):
        """
        dim : typevar
            | ellipsis_typevar
            | type
            | type_constr
            | INTEGER
        typevar : NAME_UPPER
        ellipsis_typevar : NAME_UPPER ELLIPSIS
        type : NAME_LOWER
        type_constr : NAME_LOWER LBRACKET type_arg_list RBRACKET

        Returns a the dim object, or None.
        TODO: Support type constructors
        """
        print('parse_dim', self.tok)
        tok = self.tok
        if tok.id == lexer.NAME_UPPER:
            tvar = coretypes.TypeVar(tok.val)
            self.advance_tok()
            if self.tok.id == lexer.ELLIPSIS:
                self.advance_tok()
                return coretypes.Ellipsis(tvar)
            else:
                return tvar
        elif tok.id == lexer.NAME_LOWER:
            saved_pos = self.pos
            name = tok.val
            self.advance_tok()
            if self.tok.id == lexer.LBRACKET:
                dim_constr = self.sym.dim_constr.get(name)
                if dim_constr is None:
                    self.pos = saved_pos
                    return None
                self.advance_tok()
                args = self.parse_type_arg_list()
                if self.tok.id == lexer.RBRACKET:
                    self.advance_tok()
                    raise RuntimeError('dim type constructors not actually supported yet')
                else:
                    raise error.DataShapeSyntaxError(self.tok.span[0], '<nofile>',
                                                     ds_str,
                                                     'Expected closing ]')
            else:
                dim = self.sym.dim.get(name)
                if dim is not None:
                    return dim
                else:
                    self.pos = saved_pos
                    return None
        elif tok.id == lexer.INTEGER:
            return coretypes.Fixed(tok.val)
        else:
            return None

    def parse_dtype(self):
        """
        dtype : typevar
              | type
              | type_constr
              | struct_type
              | funcproto_or_tuple_type
        typevar : NAME_UPPER
        ellipsis_typevar : NAME_UPPER ELLIPSIS
        type : NAME_LOWER
        type_constr : NAME_LOWER LBRACKET type_arg_list RBRACKET
        struct_type : LBRACE ...
        funcproto_or_tuple_type : LPAREN ...

        Returns a the dtype object, or None.
        """
        print('parse_dtype', self.tok)
        tok = self.tok
        if tok.id == lexer.NAME_UPPER:
            tvar = coretypes.TypeVar(tok.val)
            self.advance_tok()
            return tvar
        elif tok.id == lexer.NAME_LOWER:
            saved_pos = self.pos
            name = tok.val
            self.advance_tok()
            if self.tok.id == lexer.LBRACKET:
                dtype_constr = self.sym.dtype_constr.get(name)
                if dtype_constr is None:
                    self.pos = saved_pos
                    return None
                self.advance_tok()
                args, kwargs = self.parse_type_arg_list()
                if self.tok.id == lexer.RBRACKET:
                    self.advance_tok()
                    return dtype_constr(*args, **kwargs)
                else:
                    raise error.DataShapeSyntaxError(self.tok.span[0], '<nofile>',
                                                     ds_str,
                                                     'Expected an argument or a closing ]')
            else:
                dtype = self.sym.dtype.get(name)
                if dtype is not None:
                    return dtype
                else:
                    self.pos = saved_pos
                    return None
        else:
            return None

    def parse_type_arg_list(self):
        """
        type_arg_list : type_arg COMMA type_arg_list
                      | type_kwarg_list
                      | type_arg
        type_kwarg_list : type_kwarg COMMA type_kwarg_list
                        | type_kwarg

        Returns a tuple (args, kwargs), or (None, None).
        """
        # Parse zero or more "type_arg COMMA" repetitions
        args = []
        arg = True
        while arg is not None:
            saved_pos = self.pos
            # Parse the type_arg
            arg = self.parse_type_arg()
            if arg is not None:
                if self.tok.id == lexer.COMMA:
                    # If a comma is next, there are more args
                    self.advance_tok()
                    args.append(arg)
                else:
                    # Otherwise we've reached the end, and there
                    # were no keyword args
                    args.append(arg)
                    return (args, {})
            else:
                break
        kwargs = self.parse_type_kwarg_list()
        return (args, kwargs)

    def parse_type_arg(self):
        """
        type_arg : datashape
                 | INTEGER
                 | STRING
                 | list_type_arg
        list_type_arg : LBRACKET RBRACKET
                      | LBRACKET datashape_list RBRACKET
                      | LBRACKET integer_list RBRACKET
                      | LBRACKET string_list RBRACKET

        Returns a type_arg value, or None.
        """
        ds = self.parse_datashape()
        if ds is not None:
            return ds
        if self.tok.id in [lexer.INTEGER, lexer.STRING]:
            val = self.tok.val
            self.advance_tok()
            return val
        elif self.tok.id == lexer.LBRACKET:
            saved_pos = self.pos
            self.advance_tok()
            val = self.parse_datashape_list()
            if val is None:
                val = self.parse_integer_list()
            if val is None:
                val = self.parse_string_list()
            if self.tok.id == lexer.RBRACKET:
                return [] if val is None else val
            else:
                raise error.DataShapeSyntaxError(self.tok.span[0], '<nofile>',
                                                 ds_str,
                                                 'Expected a closing ]')
        else:
            return None

def parse(ds_str, sym):
    """Parses a single datashape from a string.

    Parameters
    ----------
    ds_str : string
        The datashape string to parse.
    sym : TypeSymbolTable
        The symbol tables of dimensions, dtypes, and type constructors for each.

    """
    dsp = DataShapeParser(ds_str, sym)
    ds = dsp.parse_datashape()
    # If no datashape could be found
    if ds is None:
        raise error.DataShapeSyntaxError(dsp.tok.span[0], '<nofile>',
                                         ds_str,
                                         'Expected a dim or a dtype')

    # Make sure there's no garbage at the end
    if dsp.pos != dsp.end_pos:
        raise error.DataShapeSyntaxError(dsp.tok.span[0], '<nofile>',
                                         ds_str,
                                         'Unexpected token in datashape')
    return ds
