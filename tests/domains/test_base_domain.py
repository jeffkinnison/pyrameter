import pytest

from pyrameter.domains.base import Domain


def test_init():
    d = Domain()
    assert d.name is None
    assert d.current is None
    assert d._complexity is None

    d = Domain('foo')
    assert d.name == 'foo'
    assert d.current is None
    assert d._complexity is None


def test_complexity():
    d = Domain()
    assert d.complexity == 1
    assert d._complexity == 1
    assert d.complexity == 1

    d = Domain()
    assert d.complexity == 1
    assert d._complexity == 1
    assert d.complexity == 1


def test_generate():
    d = Domain()
    with pytest.raises(NotImplementedError):
        d.generate()


def test_map_to_domain():
    inputs = [None, 'foo', 1, 1.0, True, False]
    d = Domain()

    for i in inputs:
        with pytest.raises(NotImplementedError):
            d.map_to_domain(i)


def test_to_index():
    inputs = [None, 'foo', 1, 1.0, True, False]
    d = Domain()

    for i in inputs:
        assert d.to_index(i) == i


def test_to_json():
    d = Domain()
    assert d.to_json() == {'name': None,
                           'type': 'pyrameter.domains.base.Domain'}

    d = Domain('foo')
    assert d.to_json() == {'name': 'foo',
                           'type': 'pyrameter.domains.base.Domain'}
