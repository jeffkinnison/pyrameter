import numpy as np
import pytest

from pyrameter.domains.discrete import DiscreteDomain


def test_init():
    d = DiscreteDomain('foo', [1, 2, 3, 4])
    assert d.id == 0
    assert d.name == 'foo'
    assert all(map(lambda x: x[0] == x[1], zip(d.domain, [1, 2, 3, 4])))
    assert d.random_state is None
    assert d._complexity is None
    assert d._current is None

    d = DiscreteDomain('bar', range(1, 5), seed=42)
    assert d.id == 1
    assert d.name == 'bar'
    assert all(map(lambda x: x[0] == x[1], zip(d.domain, [1, 2, 3, 4])))
    assert d._complexity is None
    assert d._current is None
    assert isinstance(d.random_state, np.random.RandomState)
    d_rs = d.random_state.get_state()
    c_rs = np.random.RandomState(42).get_state()
    for i in range(5):
        if not isinstance(d_rs[i], np.ndarray):
            assert d_rs[i] == c_rs[i]
        else:
            assert np.all(d_rs[i] == c_rs[i])

    names = ['baz', 'qux', 'meep', 'moop', 'eek', 'barba', 'durkle']
    domains = [1, 1.0, 'hi', (1, 2), True, False, None]
    for i, name, domain in zip(range(len(names)), names, domains):
        d = DiscreteDomain(name, domain)
        assert d.id == i + 2
        assert d.name == name
        assert isinstance(d.domain, list)
        assert d.domain[0] == domain
        assert d.random_state is None
        assert d._complexity is None
        assert d._current is None


def test_complexity():
    domain = []
    for i in range(1000):
        d = DiscreteDomain('foo', domain)
        correct = 2 - (1 / len(domain)) if len(domain) > 0 else 1
        assert d.complexity == correct
        domain.append(i)


def test_generate():
    domain = []
    for i in range(100):
        d = DiscreteDomain('foo', domain)
        if i > 0:
            for _ in range(100):
                assert 0 <= d.generate() < i
        else:
            assert d.generate() is None
        domain.append(i)


def test_map_to_domain():
    domain = range(35, 135)
    d = DiscreteDomain('foo', domain)
    for idx in range(len(domain)):
        assert d.map_to_domain(idx) == domain[idx]

    assert d.map_to_domain(2000) == domain[-1]
    assert d.map_to_domain(2000, bound=False) is None

    assert d.map_to_domain(-1) == domain[0]
    assert d.map_to_domain(-1, bound=False) is None


def test_to_index():
    domain = range(35, 135)
    d = DiscreteDomain('foo', domain)
    for idx in range(len(domain)):
        assert d.to_index(domain[idx]) == idx

    assert d.to_index(-1) is None
    assert d.to_index(34) is None
    assert d.to_index(135) is None
    assert d.to_index(20975) is None


def test_to_json():
    d = DiscreteDomain('foo', [1, 2, 3, 4])
    correct = {
        'name': 'foo',
        'domain': [1, 2, 3, 4],
        'random_state': None
    }
    assert d.to_json() == correct

    d = DiscreteDomain('bar', range(1, 5), seed=42)
    rs = np.random.RandomState(42).get_state()
    correct = {
        'name': 'bar',
        'domain': [1, 2, 3, 4],
        'random_state': [rs[0], list(rs[1]), rs[2], rs[3], rs[4]]
    }
    assert d.to_json() == correct

    names = ['baz', 'qux', 'meep', 'moop', 'eek', 'barba', 'durkle']
    domains = [1, 1.0, 'hi', (1, 2), True, False, None]
    for name, domain in zip(names, domains):
        d = DiscreteDomain(name, domain)
        correct = {
            'name': name,
            'domain': [domain],
            'random_state': None
        }
        assert d.to_json() == correct
