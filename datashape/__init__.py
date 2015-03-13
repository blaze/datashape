from __future__ import absolute_import

from . import lexer, parser
from . import type_equation_solver
from .coretypes import *
from .predicates import *
from .typesets import *
from .user import *
from .type_symbol_table import *
from .overload_resolver import *
from .discovery import discover
from .util import *
from .coercion import coercion_cost
from .promote import promote, optionify
from .error import (DataShapeSyntaxError, OverloadError, UnificationError,
                    CoercionError)


__version__ = '0.4.4'


def test(verbose=False, xunitfile=None, exit=False):
    """
    Runs the full Datashape test suite, outputting
    the results of the tests to sys.stdout.

    This uses py.test tests to discover which tests to
    run. By default, it runs any tests in any 'test_*py'
    files and any function or method prefixed with 'test_'.
    It also runs any doctests in the doc strings of any *py
    files.
    Test collection rules are customizable. Full default rules
    found here:
    http://pytest.org/latest/goodpractises.html#test-discovery

    Parameters
    ----------
    verbose : Bool, optional
        True prints the name of each test when runnig it
    xunitfile : string, optional
        If provided, writes the test results to an xunit
        style xml file. This is useful for running the tests
        in a CI server such as Jenkins.
    exit : bool, optional
        If True, the function will call sys.exit with an
        error code after the tests are finished.
    """
    import pytest
    import os
    import sys
    argv = []
    if verbose:
        argv.extend(['--verbose'])
    # Output an xml xunit file if requested
    if xunitfile:
        argv.extend(['--junitxml==%s' % xunitfile])
    # Include doctests in modules
    argv.append('--doctest-modules')
    # Starting directory for where to start test collection
    # is the directory containing this file
    argv.append(os.path.abspath(os.path.dirname(__file__)))

    # print versions (handy when reporting problems)
    print('Datashape version: %s' % __version__)
    print('args {0}'.format(argv))
    sys.stdout.flush()
    # py.test execution
    ret = pytest.main(argv)
    if exit:
        import sys
        sys.exit(ret)
    else:
        return ret

