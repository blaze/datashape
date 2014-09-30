from datashape.predicates import *
from datashape.predicates import _dimensions
from datashape.coretypes import TypeVar, int32
from unittest import TestCase

class Test_All(TestCase):
    def test_isfixed(self):
        assert not isfixed(TypeVar('M') * int32)

    def test_option(self):
        assert _dimensions('?int') == _dimensions('int')
        assert _dimensions('3 * ?int') == _dimensions('3 * int')

