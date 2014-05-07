"""
Utility functions that are unrelated to datashape

Do not import datashape modules into this module.  See util.py in that case
"""

from __future__ import print_function, division, absolute_import


class IndexCallable(object):
    """ Provide getitem syntax for functions

    >>> def inc(x):
    ...     return x + 1

    >>> I = IndexCallable(inc)
    >>> I[3]
    4
    """
    __slots__ = 'fn',
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, key):
        return self.fn(key)


def remove(predicate, seq):
    return filter(lambda x: not predicate(x), seq)


def reverse_dict(d):
    """

    >>> reverse_dict({1: 'one', 2: 'two'}) # doctest: +SKIP
    {'one': 1, 'two': 2}
    """
    new = dict()
    for k, v in d.items():
        if v in d:
            raise ValueError("Repated values")
        new[v] = k
    return new
