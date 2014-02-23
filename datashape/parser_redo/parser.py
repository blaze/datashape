"""
Parser for the datashape grammar.
"""

from __future__ import absolute_import, division, print_function

from .. import error
from . import lexer

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
        tok = self.tok
        if tok.id:
            return None
        else:
            raise error.DataShapeSyntaxError(self.tok.span[0], '<nofile>',
                                             self.ds_str,
                                             'Expected a datashape')

def parse(ds_str, sym):
    """Parses a single datashape from a string.

    Parameters
    ----------
    ds_str : string
        The datashape string to parse.
    sym : TypeSymTable
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
