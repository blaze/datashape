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

    def check_failing_token(self, ds_str):
        # Creating the lexer will fail, because the error is
        # in the first token.
        self.assertRaises(datashape.DataShapeSyntaxError, parser.DataShapeLexer, ds_str)

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
        self.check_isolated_token('INTEGER', '32102')
        self.check_isolated_token('RARROW', '->')
        self.check_isolated_token('STRING', '"testing"')
        self.check_isolated_token('STRING', '"testing"')

    def test_integer(self):
        # Digits
        self.check_isolated_token('INTEGER', '0')
        self.check_isolated_token('INTEGER', '1')
        self.check_isolated_token('INTEGER', '2')
        self.check_isolated_token('INTEGER', '3')
        self.check_isolated_token('INTEGER', '4')
        self.check_isolated_token('INTEGER', '5')
        self.check_isolated_token('INTEGER', '6')
        self.check_isolated_token('INTEGER', '7')
        self.check_isolated_token('INTEGER', '8')
        self.check_isolated_token('INTEGER', '9')
        # Various-sized numbers
        self.check_isolated_token('INTEGER', '10')
        self.check_isolated_token('INTEGER', '102')
        self.check_isolated_token('INTEGER', '1024')
        self.check_isolated_token('INTEGER', '10246')
        self.check_isolated_token('INTEGER', '102468')
        self.check_isolated_token('INTEGER', '1024683')
        self.check_isolated_token('INTEGER', '10246835')
        self.check_isolated_token('INTEGER', '102468357')
        self.check_isolated_token('INTEGER', '1024683579')
        # Leading zeros are not allowed
        self.check_failing_token('00')
        self.check_failing_token('01')
        self.check_failing_token('090')

    def test_string(self):
        # Trivial strings
        self.check_isolated_token('STRING', '""')
        self.check_isolated_token('STRING', "''")
        self.check_isolated_token('STRING', '"test"')
        self.check_isolated_token('STRING', "'test'")
        # Valid escaped characters
        self.check_isolated_token('STRING', r'"\"\b\f\n\r\t\ub155"')
        self.check_isolated_token('STRING', r"'\'\b\f\n\r\t\ub155'")
        # A sampling of invalid escaped characters
        self.check_failing_token(r'''"\'"''')
        self.check_failing_token(r"""'\"'""")
        self.check_failing_token(r"'\a'")
        self.check_failing_token(r"'\s'")
        self.check_failing_token(r"'\R'")
        self.check_failing_token(r"'\N'")
        self.check_failing_token(r"'\U'")
        self.check_failing_token(r"'\u123g'")
        self.check_failing_token(r"'\u123'")
        # Some unescaped unicode characters
        self.check_isolated_token('STRING', u'"\uc548\ub155"')

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

    def check_whitespace_token_sequence(self, ds_str):
        # Check that a particular token sequence is lexed
        lex = parser.DataShapeLexer(ds_str)
        self.assertEqual(lex.token, parser.COLON)
        lex.advance()
        self.assertEqual(lex.token, parser.STRING)
        lex.advance()
        self.assertEqual(lex.token, parser.INTEGER)
        lex.advance()
        self.assertEqual(lex.token, parser.RARROW)
        lex.advance()
        self.assertEqual(lex.token, parser.EQUAL)
        lex.advance()
        self.assertEqual(lex.token, parser.ASTERISK)
        lex.advance()
        self.assertEqual(lex.token, parser.NAME_OTHER)
        lex.advance()
        self.assertEqual(lex.token, None)

    def test_whitespace(self):
        # With minimal whitespace
        self.check_whitespace_token_sequence(':"x"3->=*_')
        # With spaces
        self.check_whitespace_token_sequence(' : "a" 0 -> = * _b ')
        # With tabs
        self.check_whitespace_token_sequence('\t:\t"a"\t0\t->\t=\t*\t_b\t')
        # With newlines
        self.check_whitespace_token_sequence('\n:\n"a"\n0\n->\n=\n*\n_b\n')
        # With spaces, tabs, newlines and comments
        self.check_whitespace_token_sequence('# comment\n' +
                                             ': # X\n' +
                                             ' "a" # "b"\t\n' +
                                             '\t12345\n\n' +
                                             '->\n' +
                                             '=\n' +
                                             '*\n' +
                                             '_b # comment\n' +
                                             ' \t # end')

if __name__ == '__main__':
    unittest.main()
