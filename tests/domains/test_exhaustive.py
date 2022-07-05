import numpy as np
import pytest

from pyrameter.domains.exhaustive import ExhaustiveDomain


def test_init():
    d = ExhaustiveDomain('foo', [1, 2, 3, 4])
    assert d.name == 'foo'
    assert all(map(lambda x: x[0] == x[1], zip(d.domain, [1, 2, 3, 4])))
    assert len(d.domain) == 4
    assert d._complexity is None
    assert d.current is None

    d = ExhaustiveDomain('bar', range(1, 5))
    assert d.name == 'bar'
    assert all(map(lambda x: x[0] == x[1], zip(d.domain, [1, 2, 3, 4])))
    assert len(d.domain) == 4
    assert d._index == 0
    assert d._complexity is None
    assert d.current is None

    names = ['baz', 'qux', 'meep', 'moop', 'eek', 'barba', 'durkle']
    domains = [1, 1.0, 'hi', (1, 2), True, False, None]
    for i, name, domain in zip(range(len(names)), names, domains):
        d = ExhaustiveDomain(name, domain)
        assert d.name == name
        assert isinstance(d.domain, list)
        assert len(d.domain) == 1
        assert d.domain[0] == domain
        assert d._index == 0
        assert d._complexity is None
        assert d.current is None

    d = ExhaustiveDomain('test', domains)
    assert d.name == 'test'
    assert isinstance(d.domain, list)
    assert all(map(lambda x: x[0] == x[1], zip(d.domain, domains)))
    assert len(d.domain) == len(domains)
    assert d._index == 0
    assert d._complexity is None
    assert d.current is None


def test_complexity():
    domain = []
    for i in range(1000):
        d = ExhaustiveDomain('foo', domain)
        correct = 2 - (1 / len(domain)) if len(domain) > 0 else 1
        assert d.complexity == correct
        domain.append(i)


def test_generate():
    d = ExhaustiveDomain('foo', 1)
    assert d.generate() == 1
    assert d._index == 0

    d = ExhaustiveDomain('foo', [])
    assert d.generate() is None

    domains = [1, 1.0, 'hi', (1, 2), True, False, None]
    for i in range(1, len(domains)):
        d = ExhaustiveDomain('foo', domains[:i])
        
        for j in range(i):
            assert d.generate() == domains[j]
            assert d._index == (j + 1 if j < i - 1 else 0)


def test_to_index():
    domain = range(35, 135)
    d = ExhaustiveDomain('foo', domain)
    for idx in range(len(domain)):
        assert d.to_index(domain[idx]) == idx

    assert d.to_index(-1) is None
    assert d.to_index(34) is None
    assert d.to_index(135) is None
    assert d.to_index(20975) is None


def test_to_json():
    d = ExhaustiveDomain('foo', [1, 2, 3, 4])
    correct = {
        'name': 'foo',
        'type': 'pyrameter.domains.exhaustive.ExhaustiveDomain',
        'domain': [1, 2, 3, 4],
        'index': 0,
    }
    assert d.to_json() == correct

    d = ExhaustiveDomain('bar', range(1, 5))
    correct = {
        'name': 'bar',
        'type': 'pyrameter.domains.exhaustive.ExhaustiveDomain',
        'domain': [1, 2, 3, 4],
        'index': 0,
    }
    assert d.to_json() == correct

    names = ['baz', 'qux', 'meep', 'moop', 'eek', 'barba', 'durkle']
    domains = [1, 1.0, 'hi', (1, 2), True, False, None]
    for name, domain in zip(names, domains):
        d = ExhaustiveDomain(name, domain)
        correct = {
            'name': name,
            'type': 'pyrameter.domains.exhaustive.ExhaustiveDomain',
            'domain': [domain],
            'index': 0,
        }
        assert d.to_json() == correct
