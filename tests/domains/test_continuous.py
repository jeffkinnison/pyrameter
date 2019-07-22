import numpy as np
import pytest
import scipy.stats

from pyrameter.domains.continuous import ContinuousDomain


def test_init():
    names = ['foo', 'bar', 'baz', 'qux', 'meep', 'moop']
    domains = ['norm', 'uniform', 'pareto', 'boltzmann', 'gamma', 'beta']

    for i, name, domain in zip(range(len(domains)), names, domains):
        d = ContinuousDomain(name, domain)
        assert d.id  == i
        assert d.name == name
        assert d.domain is getattr(scipy.stats, domain)
        assert d._current is None
        assert d._complexity is None


def test_complexity():
    d = ContinuousDomain('foo', 'uniform', loc=0, scale=1)
    a, b = scipy.stats.uniform.interval(0.999, loc=0, scale=1)
    correct = 2 + np.abs(b - a)
    assert d.complexity == correct
    assert d._complexity == correct
    assert d.complexity == correct

    d = ContinuousDomain('foo', 'norm', loc=-873, scale=98)
    a, b = scipy.stats.norm.interval(0.999, loc=-873, scale=98)
    correct = 2 + np.abs(b - a)
    assert d.complexity == correct
    assert d._complexity == correct
    assert d.complexity == correct


def test_generate():
    d = ContinuousDomain('foo', 'uniform', loc=0, scale=1, seed=42)
    rs = np.random.RandomState(42)

    for _ in range(1000):
        correct = scipy.stats.uniform.rvs(loc=0, scale=1, random_state=rs)
        assert d.generate() == correct

    d = ContinuousDomain('foo', 'norm', loc=-873, scale=98, seed=42)
    rs = np.random.RandomState(42)

    for _ in range(1000):
        correct = scipy.stats.norm.rvs(loc=-873, scale=98, random_state=rs)
        assert d.generate() == correct


def test_map_to_domain():
    d = ContinuousDomain('foo', 'uniform', loc=0, scale=1, seed=42)
    assert d.map_to_domain(-1) == -1
    assert d.map_to_domain(0) == 0
    assert d.map_to_domain(0.5) == 0.5
    assert d.map_to_domain(1) == 1
    assert d.map_to_domain(2) == 2

    assert d.map_to_domain(-1, bound=True) == None
    assert d.map_to_domain(0, bound=True) == 0
    assert d.map_to_domain(0.5, bound=True) == 0.5
    assert d.map_to_domain(1, bound=True) == 1
    assert d.map_to_domain(2, bound=True) == None


def test_to_index():
    d = ContinuousDomain('foo', 'uniform', loc=0, scale=1, seed=42)
    assert d.to_index(-1) == -1
    assert d.to_index(0) == 0
    assert d.to_index(0.5) == 0.5
    assert d.to_index(1) == 1
    assert d.to_index(2) == 2

    assert d.to_index(-1, bound=True) == None
    assert d.to_index(0, bound=True) == 0
    assert d.to_index(0.5, bound=True) == 0.5
    assert d.to_index(1, bound=True) == 1
    assert d.to_index(2, bound=True) == None


def test_to_json():
    d = ContinuousDomain('foo', 'uniform', loc=0, scale=1, seed=42)
    rs = np.random.RandomState(42).get_state()
    correct = {
        'name': 'foo',
        'domain': 'uniform',
        'domain_args': tuple(),
        'domain_kwargs': {
            'loc': 0,
            'scale': 1,
            'random_state': [rs[0], list(rs[1]), rs[2], rs[3], rs[4]]
        }
    }

    d = ContinuousDomain('foo', 'norm', loc=0, scale=1, seed=1337)
    rs = np.random.RandomState(1337).get_state()
    correct = {
        'name': 'foo',
        'domain': 'uniform',
        'domain_args': tuple(),
        'domain_kwargs': {
            'loc': 0,
            'scale': 1,
            'random_state': [rs[0], list(rs[1]), rs[2], rs[3], rs[4]]
        }
    }
