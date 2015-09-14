from abc import ABCMeta

from ..py2help import with_metaclass
from ..coretypes import (
    DataShape,
    DateTime,
    Option,
    Record,
    String,
    Time,
    TimeDelta,
    Tuple,
    Units,
)
from ..dispatch import dispatch


def _fmt_path(path):
    """Format the path for final display.

    Parameters
    ----------
    path : iterable of str
        The path to the values that are not equal.

    Returns
    -------
    fmtd : str
        The formatted path to put into the error message.
    """
    if not path:
        return ''
    return 'path: _' + ''.join(path)


@dispatch(DataShape, DataShape)
def assert_dshape_equal(a, b, check_dim=True, path=None, **kwargs):
    ashape = a.shape
    bshape = b.shape

    if path is None:
        path = ()

    if check_dim:
        for n, (adim, bdim) in enumerate(zip(ashape, bshape)):
            if adim != bdim:
                path += '.shape[%d]' % n,
                raise AssertionError(
                    'dimensions do not match: %s != %s%s\n%s' % (
                        adim,
                        bdim,
                        ('\n%s != %s' % (
                            ' * '.join(map(str, ashape)),
                            ' * '.join(map(str, bshape)),
                        )) if len(a.shape) > 1 else '',
                        _fmt_path(path),
                    ),
                )

    path += '.measure',
    assert_dshape_equal(
        a.measure,
        b.measure,
        check_dim=check_dim,
        path=path,
        **kwargs
    )


class Slotted(with_metaclass(ABCMeta)):
    @classmethod
    def __subclasshook__(cls, subcls):
        return hasattr(subcls, '__slots__')


@assert_dshape_equal.register(Slotted, Slotted)
def _check_slots(a, b, path=None, **kwargs):
    """Genric checker that iterates over the ``__slots__`` and asserts they
    are equal. This is a non-recursive function.

    Parameters
    ----------
    a, b : Slotted
        The shapes to check.
    path : iteratable of str, optional
        The path to the current ``a`` and ``b`` values.

    Raises
    ------
    AssertionError
        When the slots of ``a`` and ``b`` are not equal.
    """
    if type(a) != type(b):
        return _base_case(a, b, path=path, **kwargs)

    if a.__slots__ != b.__slots__:
        raise AssertionError(
            'slots mismatch: %r != %r\n%s' % (
                a.__slots__, b.__slots__, _fmt_path(path),
            ),
        )
    if path is None:
        path = ()
    for slot in a.__slots__:
        if getattr(a, slot) != getattr(b, slot):
            path += '.' + slot,
            raise AssertionError(
                "%s %ss do not match: %r != %r\n%s" % (
                    type(a).__name__.lower(),
                    slot,
                    getattr(a, slot),
                    getattr(b, slot),
                    _fmt_path(path),
                ),
            )


@assert_dshape_equal.register(object, object)
def _base_case(a, b, path=None, **kwargs):
    if a != b:
        raise AssertionError('%s != %s\n%s' % (a, b, _fmt_path(path)))


@dispatch((DateTime, Time), (DateTime, Time))
def assert_dshape_equal(a, b, path=None, check_tz=True, **kwargs):
    if type(a) != type(b):
        return _base_case(a, b)
    if check_tz:
        _check_slots(a, b, path)


@dispatch(TimeDelta, TimeDelta)
def assert_dshape_equal(a, b, path=None, check_timedelta_unit=True, **kwargs):
    if check_timedelta_unit:
        _check_slots(a, b, path)


@dispatch(Units, Units)
def assert_dshape_equal(a, b, path=None, **kwargs):
    if path is None:
        path = ()

    if a.unit != b.unit:
        path += '.unit',
        raise AssertionError(
            '%s units do not match: %r != %s\n%s' % (
                type(a).__name__.lower(), a.unit, b.unit, _fmt_path(path),
            ),
        )

    path.append('.tp')
    assert_dshape_equal(a.tp, b.tp, **kwargs)


@dispatch(String, String)
def assert_dshape_equal(a,
                        b,
                        path=None,
                        check_str_encoding=True,
                        check_str_fixlen=True,
                        **kwargs):
    if path is None:
        path = ()
    if check_str_encoding and a.encoding != b.encoding:
        path += '.encoding',
        raise AssertionError(
            'string encodings do not match: %r != %r\n%s' % (
                a.encoding, b.encoding, _fmt_path(path),
            ),
        )
    if check_str_fixlen and a.fixlen != b.fixlen:
        path += '.fixlen',
        raise AssertionError(
            'string fixlens do not match: %d != %d\n%s' % (
                a.fixlen, b.fixlen, _fmt_path(path),
            ),
        )


@dispatch(Option, Option)
def assert_dshape_equal(a, b, path=None, **kwargs):
    if path is None:
        path = ()
    path += '.ty',
    return assert_dshape_equal(a.ty, b.ty, path=path, **kwargs)


@dispatch(Record, Record)
def assert_dshape_equal(a, b, check_record_order=True, path=None, **kwargs):
    afields = a.fields
    bfields = b.fields

    if len(afields) != len(bfields):
        raise AssertionError(
            'records have mismatched field counts: %d != %d\n%r != %r\n%s' % (
                len(afields), len(bfields), a, b, _fmt_path(path),
            ),
        )

    if not check_record_order:
        afields = sorted(afields)
        bfields = sorted(bfields)

    if path is None:
        path = ()
    for n, ((aname, afield), (bname, bfield)) in enumerate(
            zip(afields, bfields)):

        if aname != bname:
            raise AssertionError(
                'record field name at position %d does not match: %r != %r\n%s'
                % (n, aname, bname, _fmt_path(path)),
            )

        assert_dshape_equal(
            afield,
            bfield,
            path=path + ('[%s]' % repr(aname),),
            **kwargs
        )


@dispatch(Tuple, Tuple)
def assert_dshape_equal(a, b, path=None, **kwargs):
    if len(a.dshapes) != len(b.dshapes):
        raise AssertionError(
            'tuples have mismatched field counts: %d !+ %d\n%r != %r\n%s' % (
                len(a.dshapes), len(b.dshapes), a, b, _fmt_path(path),
            ),
        )
    if path is None:
        path = ()
    path += '.dshapes',
    for n, (ashape, bshape) in enumerate(zip(a.dshapes, b.dshapes)):
        assert_dshape_equal(
            ashape,
            bshape,
            path=path + ('[%d]' % n,),
            **kwargs
        )
