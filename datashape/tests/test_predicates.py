from datashape.predicates import isfixed, _dimensions
from datashape.predicates import _dimensions
from datashape.coretypes import TypeVar, int32


def test_isfixed():
    assert not isfixed(TypeVar('M') * int32)


def test_option():
    assert _dimensions('?int') == _dimensions('int')
    assert _dimensions('3 * ?int') == _dimensions('3 * int')


def test_tuple():
    assert _dimensions('1 * (int, string)') == 2
    assert _dimensions('3 * (int, string)') == 2
    assert _dimensions('(int, string)') == 1
