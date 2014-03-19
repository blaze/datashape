import unittest

import datashape
from datashape import dshape, has_var_dim, has_ellipsis


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

    def test_has_var_dim(self):
        msg = ""
        fail = False
        ds_pos = [dshape("... * float32"),
                  dshape("A... * float32"),
                  dshape("var * float32"),
                  dshape("10 * { f0: int32, f1: A... * float32 }"),
                  dshape("{ f0 : { g0 : var * int }, f1: int32 }"),
                  (dshape("var*int32"),),
                  ]
        ds_false_negatives = list(filter(lambda ds: not has_var_dim(ds), ds_pos))
        if len(ds_false_negatives) != 0:
            fail = True
            msg += "The following dshapes should have a var dim\n  %s\n" \
                   % "\n  ".join(map(str, ds_false_negatives))
        ds_neg = [dshape("float32"),
                  dshape("10 * float32"),
                  dshape("10 * { f0: int32, f1: 10 * float32 }"),
                  dshape("{ f0 : { g0 : 2 * int }, f1: int32 }"),
                  (dshape("int32"),),
                  ]
        ds_false_positives = list(filter(has_var_dim, ds_neg))
        if len(ds_false_positives) != 0:
            fail = True
            msg += "The following dshapes should not have a var dim\n  %s \n" \
                   % "\n  ".join(map(str, ds_false_positives))

        self.assertFalse(fail, msg)

    def test_has_ellipsis(self):
        msg = ""
        fail = False
        ds_pos = [dshape("... * float32"),
                  dshape("A... * float32"),
                  dshape("var * ... * float32"),
                  dshape("(int32, M... * int16) -> var * int8"),
                  dshape("(int32, var * int16) -> ... * int8"),
                  dshape("10 * { f0: int32, f1: A... * float32 }"),
                  dshape("{ f0 : { g0 : ... * int }, f1: int32 }"),
                  (dshape("... * int32"),),
                  ]
        ds_false_negatives = list(filter(lambda ds: not has_ellipsis(ds), ds_pos))
        if len(ds_false_negatives) != 0:
            fail = True
            msg += "The following dshapes should have an ellipsis\n  %s\n" \
                   % "\n  ".join(map(str, ds_false_negatives))
        ds_neg = [dshape("float32"),
                  dshape("10 * var * float32"),
                  dshape("M * float32"),
                  dshape("(int32, M * int16) -> var * int8"),
                  dshape("(int32, int16) -> var * int8"),
                  dshape("10 * { f0: int32, f1: 10 * float32 }"),
                  dshape("{ f0 : { g0 : 2 * int }, f1: int32 }"),
                  (dshape("M * int32"),),
                  ]
        ds_false_positives = list(filter(has_ellipsis, ds_neg))
        if len(ds_false_positives) != 0:
            fail = True
            msg += "The following dshapes should not have an ellipsis\n  %s \n" \
                   % "\n  ".join(map(str, ds_false_positives))

        self.assertFalse(fail, msg)

if __name__ == '__main__':
    unittest.main()

