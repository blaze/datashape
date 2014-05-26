from datashape.discovery import discover
from datashape.coretypes import *
from datashape.py2help import skip

def test_simple():
    assert discover(3) == int64
    assert discover(3.0) == float64
    assert discover('Hello') == string


def test_list():
    assert discover([1, 2, 3]) == 3 * discover(1)
    assert discover([1.0, 2.0, 3.0]) == 3 * discover(1.0)


@skip("We don't have logic in datashape for this yet")
def test_list_many_types():
    assert discover([1, 1.0]) == 2 * discover(1.0)
