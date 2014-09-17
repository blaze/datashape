import pickle

import numpy as np
import pytest

from datashape.coretypes import Record, real, String, CType
from datashape import dshape, to_numpy_dtype, from_numpy


@pytest.fixture
def a():
    return Record([('x', int), ('y', int)])


@pytest.fixture
def b():
    return Record([('y', int), ('x', int)])


def test_respects_order(a, b):
    assert a != b


def test_strings():
    assert Record([('x', 'real')]) == Record([('x', real)])


class TestToNumpyDtype(object):
    def test_simple(self):
        assert to_numpy_dtype(dshape('2 * int32')) == np.int32
        assert (to_numpy_dtype(dshape('2 * {x: int32, y: int32}')) ==
                np.dtype([('x', '<i4'), ('y', '<i4')]))

    def test_datetime(self):
        assert to_numpy_dtype(dshape('2 * datetime')) == np.dtype('M8[us]')

    def test_date(self):
        assert to_numpy_dtype(dshape('2 * date')) == np.dtype('M8[D]')

    def test_string(self):
        assert to_numpy_dtype(dshape('2 * string')) == np.dtype('O')

    def test_dimensions(self):
        return to_numpy_dtype(dshape('var * int32')) == np.int32


class TestFromNumPyDtype(object):

    def test_int32(self):
        assert from_numpy((2,), 'int32') == dshape('2 * int32')
        assert from_numpy((2,), 'i4') == dshape('2 * int32')

    def test_struct(self):
        dtype = np.dtype([('x', '<i4'), ('y', '<i4')])
        result = from_numpy((2,), dtype)
        assert result == dshape('2 * {x: int32, y: int32}')

    def test_datetime(self):
        keys = 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as'
        for k in keys:
            assert from_numpy((2,),
                              np.dtype('M8[%s]' % k)) == dshape('2 * datetime')

    def test_date(self):
        for d in ('D', 'M', 'Y', 'W'):
            assert from_numpy((2,),
                              np.dtype('M8[%s]' % d)) == dshape('2 * date')

    def test_ascii_string(self):
        assert (from_numpy((2,), np.dtype('S7')) ==
                dshape('2 * string[7, "ascii"]'))

    def test_string(self):
        assert (from_numpy((2,), np.dtype('U7')) ==
                dshape('2 * string[7, "U32"]'))

    def test_string_from_CType_classmethod(self):
        assert CType.from_numpy_dtype(np.dtype('S7')) == String(7, 'A')


def test_eq():
    assert dshape('int') == dshape('int')
    assert dshape('int') != 'apple'


def test_serializable():
    ds = dshape('''{id: int64,
                    name: string,
                    amount: float32,
                    arr: 3 * (int32, string)}''')
    ds2 = pickle.loads(pickle.dumps(ds))

    assert str(ds) == str(ds2)

def test_subshape():
    ds = dshape('5 * 3 * float32')
    assert ds.subshape[2:] == dshape('3 * 3 * float32')

    ds = dshape('5 * 3 * float32')
    assert ds.subshape[::2] == dshape('3 * 3 * float32')
