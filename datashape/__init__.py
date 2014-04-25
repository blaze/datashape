from __future__ import absolute_import

from . import lexer, parser
from . import type_equation_solver
from .coretypes import *
from .predicates import *
from .typesets import *
from .user import *
from .type_symbol_table import *
from .overload_resolver import *
from .util import *
from .coercion import coercion_cost
from .error import (DataShapeSyntaxError, OverloadError, UnificationError,
                    CoercionError)

__version__ = '0.1.1-dev'

def test(verbosity=1, xunitfile=None, exit=False):
    """
    Runs the full Datashape test suite, outputting
    the results of the tests to sys.stdout.

    This uses nose tests to discover which tests to
    run, and runs tests in any 'tests' subdirectory
    within the Datashape module.

    Parameters
    ----------
    verbosity : int, optional
        Value 0 prints very little, 1 prints a little bit,
        and 2 prints the test names while testing.
    xunitfile : string, optional
        If provided, writes the test results to an xunit
        style xml file. This is useful for running the tests
        in a CI server such as Jenkins.
    exit : bool, optional
        If True, the function will call sys.exit with an
        error code after the tests are finished.
    """
    import nose
    import os
    import sys
    argv = ['nosetests', '--verbosity=%d' % verbosity]
    # Output an xunit file if requested
    if xunitfile:
        argv.extend(['--with-xunit', '--xunit-file=%s' % xunitfile])
    # Set the logging level to warn
    argv.extend(['--logging-level=WARN'])
    # Add all 'tests' subdirectories to the options
    rootdir = os.path.dirname(__file__)
    for root, dirs, files in os.walk(rootdir):
        if 'tests' in dirs:
            testsdir = os.path.join(root, 'tests')
            argv.append(testsdir)
            print('Test dir: %s' % testsdir[len(rootdir)+1:])
    # print versions (handy when reporting problems)
    print('Datashape version: %s' % __version__)
    sys.stdout.flush()
    # Ask nose to do its thing
    return nose.main(argv=argv, exit=exit)
