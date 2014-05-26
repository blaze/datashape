from datashape.discovery import discover
from datashape.coretypes import *
from datashape.py2help import skip
from datashape import dshape

def test_simple():
    assert discover(3) == int64
    assert discover(3.0) == float64
    assert discover('Hello') == string
    assert discover(True) == bool


def test_list():
    assert discover([1, 2, 3]) == 3 * discover(1)
    assert discover([1.0, 2.0, 3.0]) == 3 * discover(1.0)


def test_heterogeneous_ordered_container():
    assert discover(('Hello', 1)) == Tuple([discover('Hello'), discover(1)])



def test_string():
    assert discover('1') == discover(1)
    assert discover('1.0') == discover(1.0)
    assert discover('True') == discover(True)
    assert discover('true') == discover(True)


def test_record():
    assert discover({'name': 'Alice', 'amount': 100}) == \
            Record([['amount', discover(100)],
                    ['name', discover('Alice')]])


@skip("We don't have logic in datashape for this yet")
def test_list_many_types():
    assert discover([1, 1.0]*100) == 200 * discover(1.0)


def test_integrative():
    data = [{'name': 'Alice', 'amount': '100'},
            {'name': 'Bob', 'amount': '200'},
            {'name': 'Charlie', 'amount': '300'}]

    assert dshape(discover(data)) == \
            dshape('3 * {amount: int64, name: string}')
