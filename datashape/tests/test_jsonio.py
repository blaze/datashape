from datashape import dshape, to_json, from_json


def test_to_json_record():
    result = to_json(dshape('var * 2 * {a: int32, b: 10 * {c: ?string}}'))
    expected = {
        'shape': ['var', 2],
        'measure': {
            'names': ['a', 'b'],
            'types': {
                'a': 'int32',
                'b': {
                    'shape': [10],
                    'measure': {
                        'names': ['c'],
                        'types': {
                            'c': '?string'
                        }
                    }
                }
            }
        }
    }
    assert result == expected


def test_from_json_record():
    received = {
        'shape': ['var', 2],
        'measure': {
            'names': ['a', 'b'],
            'types': {
                'a': 'int32',
                'b': {
                    'shape': [10],
                    'measure': {
                        'names': ['c'],
                        'types': {
                            'c': '?string'
                        }
                    }
                }
            }
        }
    }
    expected = dshape('var * 2 * {a: int32, b: 10 * {c: ?string}}')
    result = from_json(received)
    assert result == expected


def test_to_json_array():
    result = to_json(dshape("10 * var * 2 * 30 * ?int64"))
    expected = {
        'shape': [10, 'var', 2, 30, ]
    }
