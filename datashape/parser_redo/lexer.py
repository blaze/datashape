"""
Lexer for the datashape grammar.
"""

from __future__ import absolute_import, division, print_function

import re
from .. import error

# This is updated to include all the token names from _tokens,
# where e.g. _tokens[NAME_LOWER-1] is the entry for NAME_LOWER
__all__ = ['DataShapeLexer']

# A list of the token names and corresponding regex
_tokens = [
    ('NAME_LOWER', r'[a-z][a-zA-Z0-9_]*'),
    ('NAME_UPPER', r'[A-Z][a-zA-Z0-9_]*'),
    ('NAME_OTHER', r'_[a-zA-Z0-9_]*'),
    ('ASTERISK',   r'\*'),
    ('COMMA',      r','),
    ('EQUAL',      r'='),
    ('COLON',      r':'),
    ('LBRACKET',   r'\['),
    ('RBRACKET',   r'\]'),
    ('LBRACE',     r'\{'),
    ('RBRACE',     r'\}'),
    ('LPAREN',     r'\('),
    ('RPAREN',     r'\)'),
    ('ELLIPSIS',   r'\.\.\.'),
    ('RARROW',     r'->'),
    ('INTEGER',    r'0(?![0-9])|[1-9][0-9]*'),
    ('STRING',    (r'(?:"(?:[^"\n\r\\]|(?:\\x[0-9a-fA-F]{2})|(?:\\u[0-9a-fA-F]{4})|(?:\\.))*")' +
                   r'|(?:\'(?:[^\'\n\r\\]|(?:\\x[0-9a-fA-F]+)|(?:\\u[0-9a-fA-F]{4})|(?:\\.))*\')')),
]

# Dynamically add all the token indices to globals() and __all__
__all__.extend(tok[0] for tok in _tokens)
globals().update((tok[0], i) for i, tok in enumerate(_tokens, 1))

# Regex for skipping whitespace and comments
_whitespace = r'(?:\s|(?:#.*$))*'

# Compile the token-matching and whitespace-matching regular expressions
_tokens_re = re.compile('|'.join('(' + tok[1] + ')' for tok in _tokens),
                        re.MULTILINE)
_whitespace_re = re.compile(_whitespace, re.MULTILINE)

class DataShapeLexer(object):
    """A lexer which converts a string output into a stream
    of tokens.

    Example
    -------

        s = '   -> ... A... Blah _eil(#'
        print('lexing %r' % s)
        lex = DataShapeLexer(s)
        while lex.token:
            print(lex.token, lex.token_name, lex.token_range, lex.token_str)
            lex.advance()

    """
    def __init__(self, ds_str):
        self.ds_str = ds_str
        self.pos = 0
        self.advance()

    def advance(self):
        """Advances the lexer to the next token. This updates
        self.token, self.token_name, and self.token_range.
        """
        # Skip whitespace
        m = _whitespace_re.match(self.ds_str, self.pos)
        if m:
            self.pos = m.end()
        # Try to match a token
        m = _tokens_re.match(self.ds_str, self.pos)
        if m:
            # m.lastindex gives us which group was matched, which
            # is one greater than the index into the _tokens list.
            self.token = m.lastindex
            self.token_name = _tokens[self.token - 1][0]
            self.token_range = m.span()
            self.pos = m.end()
        else:
            if self.pos == len(self.ds_str):
                self.token = None
                self.token_name = None
                self.token_range = None
            else:
                raise error.DataShapeSyntaxError(self.pos, '<nofile>',
                                                 self.ds_str,
                                                 'Invalid DataShape token')

    @property
    def token_str(self):
        tr = self.token_range
        return self.ds_str[tr[0]:tr[1]]

if __name__ == '__main__':
    s = '   -> ... A... Blah _eil(#'
    print('lexing %r' % s)
    lex = DataShapeLexer(s)
    while lex.token:
        print(lex.token, lex.token_name, lex.token_range, lex.token_str)
        lex.advance()
