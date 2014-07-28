from __future__ import print_function, division, absolute_import

import unittest

from datashape.py2help import xfail

from datashape import dshape, dshapes
from datashape import coretypes

from datashape.overload_resolver import OverloadResolver


class TestOverloading(unittest.TestCase):
    @xfail(reason='function signatures with typevars are not supported presently')
    def test_best_match(self):
        d1 = dshape('10 * T1 * int32')
        d2 = dshape('T2 * T2 * float32')
        match = best_match(f, coretypes.Tuple([d1, d2]))
        self.assertEqual(str(match.sig),
                         '(X * Y * float32, X * Y * float32) -> X * Y * float32')

    def test_best_match_typevar_dims(self):
        ores = OverloadResolver('f')
        ores.extend_overloads(['(X * Y * float32, X * Y * float32) -> X * Y * float32',
                               '(X * Y * complex[float32], X * Y * complex[float32]) -> X * Y * complex[float32]',
                               '(X * Y * complex[float64], X * Y * complex[float64]) -> X * Y * complex[float64]'])
        d1 = dshape('3 * 10 * complex[float32]')
        d2 = dshape('3 * 10 * float32')
        idx, match = ores.resolve_overload(coretypes.Tuple([d1, d2]))
        self.assertEqual(idx, 1)
        self.assertEqual(match,
                         dshape('(3 * 10 * complex[float32], 3 * 10 * complex[float32]) -> 3 * 10 * complex[float32]')[0])

    def test_best_match_ellipses(self):
        ores = OverloadResolver('g')
        ores.extend_overloads(['(X * Y * float32, X * Y * float32) -> X * int32',
                               '(X * Y * float32, ... * float32) -> X * int32'])
        d1 = dshape('10 * var * int32')
        d2 = dshape('3 * float32')
        idx, match = ores.resolve_overload(coretypes.Tuple([d1, d2]))
        self.assertEqual(idx, 1)
        self.assertEqual(match,
                         dshape('(10 * var * float32, 3 * float32) -> 10 * int32')[0])

    def test_best_match_signed_vs_unsigned(self):
        ores = OverloadResolver('h')
        ores.extend_overloads(['(A... * int64) -> A... * int64',
                               '(A... * uint64) -> A... * uint64'])
        d1 = dshape('10 * 3 * int64')
        idx, match = ores.resolve_overload(coretypes.Tuple([d1]))
        self.assertEqual(idx, 0)
        self.assertEqual(match,
                         dshape('(10 * 3 * int64) -> 10 * 3 * int64')[0])
        d1 = dshape('4 * 5 * uint64')
        idx, match = ores.resolve_overload(coretypes.Tuple([d1]))
        self.assertEqual(idx, 1)
        self.assertEqual(match,
                         dshape('(4 * 5 * uint64) -> 4 * 5 * uint64')[0])

    def test_best_match_float_int_complex(self):
        ores = OverloadResolver('j')
        ores.extend_overloads(['(A... * float64, A... * float64) -> A... * float64',
                               '(A... * complex[float32], A... * complex[float32]) -> A... * complex[float32]'])
        d1, d2 = dshapes('3 * float64', 'int32')
        idx, match = ores.resolve_overload(coretypes.Tuple([d1, d2]))
        self.assertEqual(idx, 0)
        self.assertEqual(match,
                         dshape('(3 * float64, float64) -> 3 * float64')[0])

    def test_best_match_int_float32_vs_float64(self):
        ores = OverloadResolver('k')
        ores.extend_overloads(['(A... * float32) -> A... * float32',
                               '(A... * float64) -> A... * float64'])
        d1 = dshape('3 * int32')
        idx, match = ores.resolve_overload(coretypes.Tuple([d1]))
        self.assertEqual(idx, 1)
        self.assertEqual(match,
                         dshape('(3 * float64) -> 3 * float64')[0])

    def test_best_match_int32_float32_ufunc_promotion(self):
        ores = OverloadResolver('m')
        ores.extend_overloads(['(A... * int32, A... * int32) -> A... * int32',
                               '(A... * float32, A... * float32) -> A... * float32',
                               '(A... * float64, A... * float64) -> A... * float64'])
        d1, d2 = dshapes('3 * int32', '3 * float32')
        idx, match = ores.resolve_overload(coretypes.Tuple([d1, d2]))
        self.assertEqual(idx, 2)
        self.assertEqual(match,
                         dshape('(3 * float64, 3 * float64) -> 3 * float64')[0])

if __name__ == '__main__':
    #TestOverloading('test_best_match_broadcasting').debug()
    unittest.main()
