from datashape import promote, Option, float64, int64, float32, optionify


def test_simple():
    x = int64
    y = float32
    z = promote(x, y)
    assert z == float64


def test_option():
    x = int64
    y = Option(float32)
    z = promote(x, y)
    assert z == Option(float64)


def test_option_in_parent():
    x = int64
    y = Option(float32)
    z = optionify(x, y, y)
    assert z == y
