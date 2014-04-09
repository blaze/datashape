from datashape.coretypes import Record
import unittest

class TestRecord(unittest.TestCase):
    def setUp(self):
        self.a = Record([('x', int), ('y', int)])
        self.b = Record([('y', int), ('x', int)])

    def test_respects_order(self):
        self.assertNotEqual(self.a, self.b)
