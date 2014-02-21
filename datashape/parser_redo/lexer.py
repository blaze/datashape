"""
Lexer for the datashape grammar.
"""

from __future__ import absolute_import, division, print_function

import re

# A list of the token names and corresponding regex
_tokens = [
    ('NAME_LOWER', r'[a-z][a-zA-Z0-9_]*'),
    ('NAME_UPPER', r'[A-Z][a-zA-Z0-9_]*'),
    ('NAME_OTHER', r'_[a-zA-Z0-9_]*'),
    ('ASTERISK',   r'\*'),
    ('COMMA',      r','),
    ('EQUAL',      r'='),
    ('LBRACKET',   r'\['),
    ('RBRACKET',   r'\]'),
    ('LBRACE',     r'\{'),
    ('RBRACE',     r'\}'),
    ('LPAREN',     r'\('),
    ('RPAREN',     r'\)'),
    ('ELLIPSIS',   r'\.\.\.'),
    ('RARROW',     r'->'),
    ('INTEGER',    r'0|[1-9][0-9]*'),
    ('STRING',     r'(?:"(?:[^"\n\r\\]|(?:\\x[0-9a-fA-F]{2})|(?:\\u[0-9a-fA-F]{4})|(?:\\.))*")|(?:\'(?:[^\'\n\r\\]|(?:\\x[0-9a-fA-F]+)|(?:\\u[0-9a-fA-F]{4})|(?:\\.))*\')'),
]

# Regex for skipping whitespace and comments
_whitespace = r'(?:\s|(?:#.*$))*'

# Compile the token-matching and whitespace-matching regular expressions
_tokens_re = re.compile('|'.join('(' + tok[1] + ')' for tok in _tokens),
                        re.MULTILINE)
_whitespace_re = re.compile(_whitespace, re.MULTILINE)

class DataShapeLexer(object):
    def __init__(self, ds_str):
        self.ds_str = ds_str
        self.pos = 0
        self.advance()

    def advance(self):
        # Skip whitespace
        m = _whitespace_re.match(self.ds_str, self.pos)
        if m:
            self.pos = m.end()
        # Try to match a token
        m = _tokens_re.match(self.ds_str, self.pos)
        if m:
            self.token_index = m.lastindex - 1
            self.token = _tokens[self.token_index][0]
            self.token_range = m.span()
            self.pos = m.end()
        else:
            if self.pos == len(self.ds_str):
                self.token_index = -1
                self.token = None
                self.token_range = None
            else:
                # TODO: custom lexing exception
                raise RuntimeError('Failed to lex at position %d' % self.pos)

    @property
    def token_str(self):
        tr = self.token_range
        return self.ds_str[tr[0]:tr[1]]

if __name__ == '__main__':
    s = '   -> ... A... Blah _eil(#'
    print('lexing %r' % s)
    lex = DataShapeLexer(s)
    while lex.token:
        print(lex.token, lex.token_index, lex.token_range, lex.token_str)
        lex.advance()
