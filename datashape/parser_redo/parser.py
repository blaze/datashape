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

        Returns a datashape object.
        """
        print('parse_datashape', self.tok)
        tok = self.tok
        if tok.id:
            # Parse zero or more "dim ASTERISK" repetitions
            dims = []
            dim = True
            while dim:
                saved_pos = self.pos
                # Parse the dim
                dim = self.parse_dim()
                if dim:
                    if self.tok.id == lexer.ASTERISK:
                        # If an asterisk is next, we're good
                        self.advance_tok()
                        dims.append(dim)
                    else:
                        # Otherwise try a dtype
                        dim = None
                        self.pos = saved_pos
            # Parse the dtype
            dtype = self.parse_dtype()
            if dtype:
                return coretypes.DataShape(*(dims + [dtype]))
            else:
                raise error.DataShapeSyntaxError(self.tok.span[0], '<nofile>',
                                                 self.ds_str,
                                                 'Expected a dim or a dtype')
        else:
            raise error.DataShapeSyntaxError(self.tok.span[0], '<nofile>',
                                             self.ds_str,
                                             'Expected a datashape')

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
                self.advance_tok()
                args = self.parse_type_arg_list()
                if self.tok.id == lexer.RBRACKET:
                    self.advance_tok()
                else:
                    raise error.DataShapeSyntaxError(self.tok.span[0], '<nofile>',
                                                     ds_str,
                                                     'Expected closing ]')
                raise RuntimeError('dim type constructors not actually supported yet')
            else:
                dim = self.sym.dim.get(name)
                if dim:
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
                self.advance_tok()
                args = self.parse_type_arg_list()
                if self.tok.id == lexer.RBRACKET:
                    self.advance_tok()
                    dtype_constr = self.sym.dtype_constr.get(name)
                    if dtype_constr:
                        return dtype_constr(*args)
                    else:
                        self.pos = saved_pos
                        return None
                else:
                    raise error.DataShapeSyntaxError(self.tok.span[0], '<nofile>',
                                                     ds_str,
                                                     'Expected an argument or a closing ]')
            else:
                dtype = self.sym.dtype.get(name)
                if dtype:
                    return dtype
                else:
                    self.pos = saved_pos
                    return None
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
    # Make sure there's no garbage at the end
    if dsp.pos != dsp.end_pos:
        raise error.DataShapeSyntaxError(dsp.tok.span[0], '<nofile>',
                                         ds_str,
                                         'Unexpected token in datashape')
    return ds
