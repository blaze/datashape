from __future__ import absolute_import, division, print_function

import unittest

from datashape.type_equation_solver import match_argtypes_to_signature
from datashape import dshape

class TestCoercion(unittest.TestCase):
    def test_nargs_mismatch(self):
        self.assertRaises(TypeError, match_argtypes_to_signature,
                          dshape('(int32, float64)'),
                          dshape('(int32) -> int32'))
        self.assertRaises(TypeError, match_argtypes_to_signature,
                          dshape('(int32, float64)'),
                          dshape('(int32, float64, int16) -> int32'))

