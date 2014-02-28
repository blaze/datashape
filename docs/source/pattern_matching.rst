Pattern Matching DataShapes
===========================

DataShape includes type variables, as symbols beginning with a
capital letter. For example `A * int32` represents a one-dimensional
array of `int32`, where the size or type of the dimension is
unspecified. Similarly, `3 * A` represents a size 3 one-dimensional
array where the data type is unspecified.

The main usage of pattern matching in the DataShape system is for
specifying function signatures. To provide a little bit of motivation,
let's examine what happens in NumPy ufuncs, and see how we can
represent their behaviors via DataShape types.

NumPy `ldexp` UFunc Example
---------------------------

We're going to use the `ldexp` ufunc, which is for the C/C++
function with overloads `double ldexp(double x, int exp)`
and `float ldexp(float x, int exp)`, computing `x * 2^exp`
by tweaking the exponent in the floating point format. (We're
ignoring the long double for now.)

These C++ functions can be represented with the DataShape
function signatures::

    (float32, int32) -> float32
    (float64, int32) -> float64

As a NumPy ufunc, there is an behavior for arrays, where the
source arrays are "broadcasted" together, and the function is
computed elementwise.

In the simplest case, given two arrays which match, the result
is an array of the same size. When one array has size one in a
dimension, it gets repeated to match the size of the other dimension. 
When one array has fewer dimensions, it gets repeated to fill
in the outer dimensions. The "broadcast" array shape is the result
of combining all these repetitions, and is the shape of the output.
Represented as DataShape function signatures, some examples are::

    (12 * float32, 12 * int32) -> 12 * float32
    (10 * float64, 1 * int32) -> 10 * float64
    (float32, 3 * 4 * int32) -> 3 * 4 * float32
    (3 * float64, 4 * 1 * int64) -> 4 * 3 * float64

Ellipsis for Broadcasting
-------------------------

To represent the general broadcasting behavior, DataShape provides
ellipsis type variables.::

    (A... * float32, A... * int32) -> A... * float32
    (A... * float64, A... * int64) -> A... * float64

Coercions/Broadcasting as a System of Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say as input we get two arrays with datashapes
`3 * 4 * float64` and `int32`. We can express this as
two systems of coercion equations as follows (using ==>
as a "coerces to" operator)::

    # float32 prototype
    3 * 4 * float64 ==> A... * float32
    int32 ==> A... * int32

    # float64 prototype
    3 * 4 * float64 ==> A... * float64
    int32 ==> A... * int32

To solve these equations, we evaluate the legality
of each coercion, and accumulate the set of values
the `A...` type variable must take.::

    # float32 prototype
    float64 ==> float32 # ILLEGAL
    3 * 4 * ==> A... *  # "3 * 4 *" in A...
    int32 ==> int32     # LEGAL
    * ==> A...          # "*" in A...

    # float64 prototype
    float64 ==> float64 # LEGAL
    3 * 4 * ==> A... *  # "3 * 4 *" in A...
    int32 ==> int32     # LEGAL
    * ==> A...          # "*" in A...

The float32 prototype can be discarded because it requires an
illegal coercion. In the float64 prototype, we collect the set
of all `A...` values `{"3 * 4 *", "*"}`, broadcast them together
to get `"3 * 4 *"`, and substitute this in the output. Doing
all the substitutions in the full prototype produces::

    (3 * 4 * float64, int32) -> 3 * 4 * float64

as the matched function prototype that results.

Disallowing Coercion
--------------------

In the particular function we picked, ideally we don't want to
allow implicit coercion of the type, because the nature of the
function is to "load the exponent" in particular formats of
floating point number. Saying `ldexp(True, 3)`, and having it
work is kind of weird.

One way to tackle this would be to add an `exact` type, both
as a dimension and a data type, which indicates that broadcasting
should be disallowed. For the discussion, in addition to `ldexp`,
lets introduce a vector magnitude function `mag`, where we want
to disallow scalar arrays to broadcast into it.::

    # ldexp signatures
    (A... * exact[float32], A... * int32) -> A... * float32
    (A... * exact[float64], A... * int64) -> A... * float64

    # mag signatures
    (A... * exact[2] * float32) -> A... * float32
    (A... * exact[3] * float32) -> A... * float32

    # ufunc but disallowing broadcasting
    (exact[A...] * int32, exact[A...] * int32) -> A... * int32

A possible syntactic sugar (which I'm not attached to, I think
this needs some exploration) for this is::

    # ldexp signatures
    (A... * float32=, A... * int32) -> A... * float32
    (A... * float64=, A... * int64) -> A... * float64

    # mag signatures
    (A... * 2= * float32) -> A... * float32
    (A... * 3= * float32) -> A... * float32

    # ufunc but disallowing broadcasting
    (A=.. * int32, A=.. * int32) -> A... * int32

Factoring a Set of Signatures
-----------------------------

One of the main things the multiple dispatch in DataShape has
to do is match input arrays against a set of signatures very
efficiently. We need to be able to hide the abstraction we're
creating, and provide performance competitive with, but ideally
superior to, what NumPy provides in its ufunc system.

Factoring the set of signatures into two or more stages which
are simpler to solve and can prune the possibilities more quickly
is one way to do this abstraction hiding. Let's use the `add` function
for our example, with the following subset of signatures. We've
included the `datetime` signatures to dispel any notion that the
signatures will always match precisely.::

    # add signatures
    (A... * int32, A... * int32) -> A... * int32
    (A... * int64, A... * int64) -> A... * int64
    (A... * float32, A... * float32) -> A... * float32
    (A... * float64, A... * float64) -> A... * float64
    (A... * timedelta, A... * timedelta) -> A... * timedelta
    (A... * datetime, A... * timedelta) -> A... * datetime
    (A... * timedelta, A... * datetime) -> A... * datetime

Because the broadcasting of all these cases is identical, we
can transform this set of signatures into two stages as follows::

    # broadcasting stage
    (A... * X, A... * Y) -> A... * Z

    # data type stage matched against (X, Y)
    (int32, int32) -> int32
    (int64, int64) -> int64
    (float32, float32) -> float32
    (float64, float64) -> float64
    (timedelta, timedelta) -> timedelta
    (datetime, timedelta) -> datetime
    (timedelta, datetime) -> datetime

Let's work through this example to illustrate how it works.::

    # Stage 1: Input arrays "3 * 1 * int32", "4 * float32"
    #    (A... * X, A... * Y) -> A... * Z
    int32 ==> X       # "int32" in X
    3 * 1 * ==> A...  # "3 * 1 *" in A...
    float32 ==> Y     # "float32" in Y
    4 * ==> A...      # "4 *" in A...

    # Solution: A... is "3 * 4 *", X is "int32", and Y is "float32"
    # Stage 2: Input arrays "int32" and "float32"
    #    (int32, int32) -> int32
    int32 ==> int32     # LEGAL
    float32 ==> int32   # ILLEGAL
    #    (float32, float32) -> float32
    int32 ==> float32   # LEGAL
    float32 ==> float32 # LEGAL
    # etc.

    # Assume we picked (float32, float32) -> float32
    # so the variables are:
    # X is "float32"
    # Y is "float32"
    # Z is "float32"
    # giving the solution substituted into stage 1:
    (3 * 1 * float32, 4 * float32) -> 3 * 4 * float32

