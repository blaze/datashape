import numpy as np

from datashape.discovery import discover, unite
from datashape.coretypes import *
from datashape.py2help import skip
from datashape import dshape

def test_simple():
    assert discover(3) == int64
    assert discover(3.0) == float64
    assert discover(3.0 + 1j) == complex128
    assert discover('Hello') == string
    assert discover(True) == bool_


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


def test_datetime():
    inputs = ["1991-02-03 04:05:06",
              "11/12/1822 06:47:26.00",
              "1822-11-12T06:47:26",
              "Fri Dec 19 15:10:11 1997",
              "Friday, November 11, 2005 17:56:21",
              "1982-2-20 5:02:00",
              "20030331 05:59:59.9",
              "Jul  6 2030  5:55PM",
              "1994-10-20 T 11:15",
              "2013-03-04T14:38:05.123",
              # "15MAR1985:14:15:22",
              # "201303041438"
              ]
    for dt in inputs:
        assert discover(dt) == datetime_


@skip("We don't have logic in datashape for this yet")
def test_list_many_types():
    assert discover([1, 1.0]*100) == 200 * discover(1.0)


def test_integrative():
    data = [{'name': 'Alice', 'amount': '100'},
            {'name': 'Bob', 'amount': '200'},
            {'name': 'Charlie', 'amount': '300'}]

    assert dshape(discover(data)) == \
            dshape('3 * {amount: int64, name: string}')


def test_numpy_scalars():
    assert discover(np.int32(1)) == int32
    assert discover(np.float64(1)) == float64


def test_numpy_array():
    assert discover(np.ones((3, 2), dtype=np.int32)) == dshape('3 * 2 * int32')


def test_unite():
    assert unite([int32, int32, int32]) == int32
    assert unite([3 * int32, 2 * int32]) == var * int32
    assert unite([2 * int32, 2 * int32]) == 2 * int32
    assert unite([3 * (2 * int32), 2 * (2 * int32)]) == var * (2 * int32)

    assert unite([int32, None, int32]) == Option(int32)
    assert not unite([string, None, int32])

    assert unite((Tuple([int32, int32, string]),
                  Tuple([int32, None, None]),
                  Tuple([int32, int32, string]))) == \
                    Tuple([int32, Option(int32), Option(string)])


def test_dshape_missing_data():
    assert dshape(discover([1, 2, '', 3])) == dshape(4 * Option(discover(1)))
