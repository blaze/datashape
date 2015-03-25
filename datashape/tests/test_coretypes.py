import pickle

import numpy as np
import pytest

from datashape.coretypes import (Record, real, String, CType, DataShape, int32,
                                 Fixed, Option, _units, _unit_aliases)
from datashape import dshape, to_numpy_dtype, from_numpy, error
from datashape.py2help import unicode


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


def test_integers():
    assert Record([(0, 'real')]) == Record([('0', real)])


def test_error_on_datashape_with_string_argument():
    with pytest.raises(TypeError):
        DataShape('5 * int32')


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

    def test_timedelta(self):
        assert to_numpy_dtype(dshape('2 * timedelta')) == np.dtype('m8[us]')
        assert to_numpy_dtype(dshape("2 * timedelta[unit='s']")) == \
            np.dtype('m8[s]')


def test_timedelta_repr():
    assert eval(repr(dshape('timedelta'))) == dshape('timedelta')
    assert eval(repr(dshape('timedelta[unit="ms"]'))) == \
        dshape('timedelta[unit="ms"]')


def test_timedelta_bad_unit():
    with pytest.raises(ValueError):
        dshape('timedelta[unit="foo"]')


def test_timedelta_nano():
    dshape('timedelta[unit="ns"]').measure.unit == 'ns'


def test_timedelta_aliases():
    for alias in _unit_aliases:
        a = alias + 's'
        assert (dshape('timedelta[unit=%r]' % a) ==
                dshape('timedelta[unit=%r]' % _unit_aliases[alias]))


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

    def test_timedelta(self):
        for d in _units:
            assert from_numpy((2,),
                              np.dtype('m8[%s]' % d)) == \
                dshape('2 * timedelta[unit=%r]' % d)

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

def test_negative_slicing():
    ds = dshape('10 * int')
    assert ds.subshape[-3:] == dshape('3 * int')

def test_newaxis_slicing():
    ds = dshape('10 * int')
    assert ds.subshape[None, :] == dshape('1 * 10 * int')
    assert ds.subshape[:, None] == dshape('10 * 1 * int')


def test_DataShape_coerces_ints():
    assert DataShape(5, 'int32')[0] == Fixed(5)
    assert DataShape(5, 'int32')[1] == int32


def test_shape():
    assert dshape('5 * 3 * float32').shape == (5, 3)
    assert dshape('float32').shape == ()
    assert dshape('float32').measure.shape == ()
    assert dshape('?float32').measure.shape == ()


def test_option_sanitizes_strings():
    assert Option('float32').ty == dshape('float32').measure

def test_option_passes_itemsize():
    assert dshape('?float32').measure.itemsize ==\
            dshape('float32').measure.itemsize


class TestComplexFieldNames(object):
    """
    The tests in this class should verify that the datashape parser can handle
    field names that contain strange characters like spaces, quotes, and
    backslashes

    The idea is that any given input datashape should be recoverable once we
    have created the actual dshape object.

    This test suite is by no means complete, but it does handle some of the
    more common special cases (common special? oxymoron?)
    """
    def test_spaces_01(self):
        space_dshape = "{'Unique Key': ?int64}"
        assert space_dshape == str(dshape(space_dshape))

    def test_spaces_02(self):
        big_space_dshape="""{ 'Unique Key' : ?int64, 'Created Date' : string,
'Closed Date' : string, Agency : string, 'Agency Name' : string,
'Complaint Type' : string, Descriptor : string, 'Location Type' : string,
'Incident Zip' : ?int64, 'Incident Address' : ?string, 'Street Name' : ?string,
'Cross Street 1' : ?string, 'Cross Street 2' : ?string,
'Intersection Street 1' : ?string, 'Intersection Street 2' : ?string,
'Address Type' : string, City : string, Landmark : string,
'Facility Type' : string, Status : string, 'Due Date' : string,
'Resolution Action Updated Date' : string, 'Community Board' : string,
Borough : string, 'X Coordinate (State Plane)' : ?int64,
'Y Coordinate (State Plane)' : ?int64, 'Park Facility Name' : string,
'Park Borough' : string, 'School Name' : string, 'School Number' : string,
'School Region' : string, 'School Code' : string,
'School Phone Number' : string, 'School Address' : string,
'School City' : string, 'School State' : string, 'School Zip' : string,
'School Not Found' : string, 'School or Citywide Complaint' : string,
'Vehicle Type' : string, 'Taxi Company Borough' : string,
'Taxi Pick Up Location' : string, 'Bridge Highway Name' : string,
'Bridge Highway Direction' : string, 'Road Ramp' : string,
'Bridge Highway Segment' : string, 'Garage Lot Name' : string,
'Ferry Direction' : string, 'Ferry Terminal Name' : string,
Latitude : ?float64, Longitude : ?float64, Location : string }"""

        ds1 = dshape(big_space_dshape)
        ds2 = dshape(str(ds1))

        assert str(ds1) == str(ds2)

    def test_single_quotes_01(self):

        quotes_dshape = """{ 'field \\' with \\' quotes' : string }"""

        ds1 = dshape(quotes_dshape)
        ds2 = dshape(str(ds1))

        assert str(ds1) == str(ds2)

    def test_double_quotes_01(self):
        quotes_dshape = """{ 'doublequote \" field \"' : int64 }"""
        ds1 = dshape(quotes_dshape)
        ds2 = dshape(str(ds1))

        assert str(ds1) == str(ds2)

    def test_multi_quotes_01(self):
        quotes_dshape = """{ 'field \\' with \\' quotes' : string, 'doublequote \" field \"' : int64 }"""

        ds1 = dshape(quotes_dshape)
        ds2 = dshape(str(ds1))

        assert str(ds1) == str(ds2)

    def test_mixed_quotes_01(self):
        quotes_dshape = """{ 'field \" with \\' quotes' : string, 'doublequote \" field \\'' : int64 }"""

        ds1 = dshape(quotes_dshape)
        ds2 = dshape(str(ds1))

        assert str(ds1) == str(ds2)

    def test_bad_02(self):
        bad_dshape = """{ Unique Key : int64}"""
        with pytest.raises(error.DataShapeSyntaxError):
            dshape(bad_dshape)

    def test_bad_backslashes_01(self):
        """backslashes aren't allowed in datashapes according to the definitions
        in lexer.py as of 2014-10-02. This is probably an oversight that should
        be fixed.
        """
        backslash_dshape = """{ 'field with \\\\   backslashes' : int64 }"""

        with pytest.raises(error.DataShapeSyntaxError):
            dshape(backslash_dshape)


def test_record_string():
    s = '{name_with_underscores: int32}'
    assert s.replace(' ', '') == str(dshape(s)).replace(' ', '')


def test_record_with_unicode_name_as_numpy_dtype():
    r = Record([(unicode('a'), 'int32')])
    assert r.to_numpy_dtype() == np.dtype([('a', 'i4')])


def test_tuple_datashape_to_numpy_dtype():
    ds = dshape('5 * (int32, float32)')
    assert to_numpy_dtype(ds) == [('f0', 'i4'), ('f1', 'f4')]
