from datashape import dshape, to_json, from_json


def test_to_json():
    result = to_json(dshape('var * 2 * {a: int32, b: 10 * {c: ?string}}'))
    expected = {
        'shape': ['Var', 2],
        'measure': {
            'a': 'int32',
            'b': {
                'shape': [10],
                'measure': {
                    'c': '?string'
                }
            }
        }
    }
    assert result == expected


def test_from_json():
    received = {
        'shape': ['Var', 2],
        'measure': {
            'a': 'int32',
            'b': {
                'shape': [10],
                'measure': {
                    'c': '?string'
                }
            }
        }
    }
    expected = dshape('var * 2 * {a: int32, b: 10 * {c: ?string}}')
    result = from_json(received)
    assert result == expected
