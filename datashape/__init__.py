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

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
