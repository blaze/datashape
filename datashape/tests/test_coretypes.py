import pickle

import numpy as np
import pytest

from datashape.coretypes import Record, real, String, CType, error
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


class TestComplexFieldNames(unittest.TestCase):
    """
    The tests in this class should verify that the datashape parser can handle field names that contain
      strange characters like spaces, quotes, and backslashes
    The idea is that any given input datashape should be recoverable once we have created the actual dshape object.

    This test suite is by no means complete, but it does handle some of the more common special cases (common special? oxymoron?)
    """


    space_dshape="""{ 'Unique Key' : ?int64 }"""

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

    bad_dshape="""{ Unique Key : int64}"""

    quotes_dshape_01="""{ 'field \\' with \\' quotes' : string }"""
    quotes_dshape_02="""{ 'doublequote \" field \"' : int64 }"""
    quotes_dshape_03="""{ 'field \\' with \\' quotes' : string, 'doublequote \" field \"' : int64 }"""

    backslash_dshape="""{ 'field with    backslashes' : int64 }"""

    def test_spaces_01(self):
        ds1=dshape(self.space_dshape)
        self.assertEqual(self.space_dshape, str(ds1))

    def test_spaces_02(self):
        ds1=dshape(self.big_space_dshape)
        self.assertEqual(self.big_space_dshape.replace("\n", ""), str(ds1))

    def test_quotes_01(self):
        ds1=dshape(self.quotes_dshape_01)
        self.assertEqual(self.quotes_dshape_01, str(ds1))

    def test_quotes_02(self):
        ds1=dshape(self.quotes_dshape_02)
        self.assertEqual(self.quotes_dshape_02, str(ds1))

    def test_quotes_03(self):
        ds1=dshape(self.quotes_dshape_02)
        self.assertEqual(self.quotes_dshape_02, str(ds1))

    def test_backslash_01(self):
        ds1=dshape(self.backslash_dshape)
        self.assertEqual(self.backslash_dshape, str(ds1))

    def test_bad_01(self):
        self.assertRaises(error.DataShapeSyntaxError,dshape,self.bad_dshape)
        
