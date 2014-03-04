import unittest

import datashape
from datashape import dshape


class TestDataShapeUtil(unittest.TestCase):
    def test_cat_dshapes(self):
        # concatenating 1 dshape is a no-op
        dslist = [dshape('3 * 10 * int32')]
        self.assertEqual(datashape.cat_dshapes(dslist),
                        dslist[0])
        # two dshapes
        dslist = [dshape('3 * 10 * int32'),
                        dshape('7 * 10 * int32')]
        self.assertEqual(datashape.cat_dshapes(dslist),
                        dshape('10 * 10 * int32'))

    def test_cat_dshapes_errors(self):
        # need at least one dshape
        self.assertRaises(ValueError, datashape.cat_dshapes, [])
        # dshapes need to match after the first dimension
        self.assertRaises(ValueError, datashape.cat_dshapes,
                        [dshape('3 * 10 * int32'), dshape('3 * 1 * int32')])


if __name__ == '__main__':
    unittest.main()

