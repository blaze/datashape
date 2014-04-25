from datashape.predicates import *
from datashape.coretypes import TypeVar, int32
from unittest import TestCase

class Test_All(TestCase):
    def test_isfixed(self):
        assert not isfixed(TypeVar('M') * int32)
