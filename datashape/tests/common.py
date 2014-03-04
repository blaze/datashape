# Utilities for the high level Blaze test suite

import unittest

class BTestCase(unittest.TestCase):
    """
    TestCase that provides some stuff missing in 2.6.
    """

    def assertIsInstance(self, obj, cls, msg=None):
        self.assertTrue(isinstance(obj, cls),
                        msg or "%s is not an instance of %s" % (obj, cls))

    def assertGreater(self, a, b, msg=None):
        self.assertTrue(a > b, msg or "%s is not greater than %s" % (a, b))

    def assertLess(self, a, b, msg=None):
        self.assertTrue(a < b, msg or "%s is not less than %s" % (a, b))
