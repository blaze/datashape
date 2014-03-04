"""Error handling"""

class DataShapeException(Exception):
    """Base class for DataShape Exceptions"""

#------------------------------------------------------------------------
# Generic Syntax Errors
#------------------------------------------------------------------------

syntax_error = """

  File {filename}, line {lineno}
    {line}
    {pointer}

{error}: {msg}
"""

class CustomSyntaxError(DataShapeException):
    """
    Makes datashape parse errors look like Python SyntaxError.
    """
    def __init__(self, lexpos, filename, text, msg=None):
        self.lexpos = lexpos
        self.filename = filename
        self.text = text
        self.msg = msg or 'invalid syntax'
        self.lineno = text.count('\n', 0, lexpos) + 1
        # Get the extent of the line with the error
        linestart = text.rfind('\n', 0, lexpos)
        if linestart < 0:
            linestart = 0
        else:
            linestart = linestart + 1
        lineend = text.find('\n', lexpos)
        if lineend < 0:
            lineend = len(text)
        self.line = text[linestart:lineend]
        self.col_offset = lexpos - linestart

        print(str(self)) # REMOVEME

    def __str__(self):
        pointer = ' '*self.col_offset + '^'

        return syntax_error.format(
            filename=self.filename,
            lineno=self.lineno,
            line=self.line,
            pointer=pointer,
            msg=self.msg,
            error=self.__class__.__name__,
        )

    def __repr__(self):
        return str(self)
#------------------------------------------------------------------------
# Typing errors
#------------------------------------------------------------------------

class DataShapeSyntaxError(CustomSyntaxError):
    pass

class DataShapeTypeError(DataShapeException):
    "Raised when there is an error with a datashape type"

class DataShapeError(DataShapeTypeError):
    """Raised for malformed datashape types"""

class UnificationError(DataShapeTypeError):
    """Raised when two DataShape types cannot be unified"""

class CoercionError(DataShapeTypeError):
    """Raised when we can't coerce a type to another type"""
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    def __str__(self):
        return 'Cannot broadcast/coerce %s to %s' % (self.src, self.dst)

    def __repr__(self):
        return str(self)


class OverloadError(DataShapeTypeError):
    """
    Raised when we can't determine which overload to select for given input
    types.
    """
