"""
Test the DataShape lexer.
"""

from __future__ import absolute_import, division, print_function

import unittest

import datashape
from datashape import parser_redo as parser

class TestDataShapeLexer(unittest.TestCase):

    def check_isolated_token(self, tname, ds_str):
        # The token name should be a property in parser
        tid = getattr(parser, tname)
        # Create a lexer, and assert the token we expect
        lex = parser.DataShapeLexer(ds_str)
        self.assertEqual(lex.token, tid)
        self.assertEqual(lex.token_name, tname)
        self.assertEqual(lex.token_range, (0, len(ds_str)))
        # Advancing by one reaches the end
        lex.advance()
        self.assertEqual(lex.token, None)
        self.assertEqual(lex.token_name, None)
        self.assertEqual(lex.token_range, None)

    def test_isolated_tokens(self):
        self.check_isolated_token('NAME_LOWER', 'testing')
        self.check_isolated_token('NAME_UPPER', 'Testing')
        self.check_isolated_token('NAME_OTHER', '_testing')
        self.check_isolated_token('ASTERISK', '*')
        self.check_isolated_token('COMMA', ',')
        self.check_isolated_token('EQUAL', '=')
        self.check_isolated_token('COLON', ':')
        self.check_isolated_token('LBRACKET', '[')
        self.check_isolated_token('RBRACKET', ']')
        self.check_isolated_token('LBRACE', '{')
        self.check_isolated_token('RBRACE', '}')
        self.check_isolated_token('LPAREN', '(')
        self.check_isolated_token('RPAREN', ')')
        self.check_isolated_token('ELLIPSIS', '...')
        self.check_isolated_token('RARROW', '->')
        self.check_isolated_token('INTEGER', '0')
        self.check_isolated_token('INTEGER', '32102')
        self.check_isolated_token('RARROW', '->')
        self.check_isolated_token('STRING', '"testing"')

    def check_failing_token(self, ds_str):
        # Creating the lexer will fail, because the error is
        # in the first token.
        self.assertRaises(datashape.DataShapeSyntaxError, parser.DataShapeLexer, ds_str)

    def test_failing_tokens(self):
        self.check_failing_token('~')
        self.check_failing_token('`')
        self.check_failing_token('!')
        self.check_failing_token('@')
        self.check_failing_token('$')
        self.check_failing_token('%')
        self.check_failing_token('^')
        self.check_failing_token('&')
        self.check_failing_token('-')
        self.check_failing_token('+')
        self.check_failing_token(';')
        self.check_failing_token('<')
        self.check_failing_token('>')
        self.check_failing_token('.')
        self.check_failing_token('..')
        self.check_failing_token('?')
        self.check_failing_token('/')
        self.check_failing_token('|')
        self.check_failing_token('\\')

if __name__ == '__main__':
    unittest.main()
