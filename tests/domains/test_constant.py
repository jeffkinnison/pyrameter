import pytest

from pyrameter.domains.constant import ConstantDomain


def test_init():
    names = ['foo', 'bar', 'baz', 'qux', 'meep', 'moop']
    domains = [1.0, 1, True, False, None, 'foo']

    for i, name, domain in zip(range(len(domains)), names, domains):
        d = ConstantDomain(name, domain)
        assert d.id  == i
        assert d.name == name
        assert d.domain == domain
        assert isinstance(d.domain, type(domain))
        assert d._current is None
        assert d._complexity is None


def test_complexity():
    d = ConstantDomain('foo', 1.0)
    assert d.complexity == 1
    assert d._complexity == 1
    assert d.complexity == 1

    d = ConstantDomain('bar', 1.0)
    assert d.complexity == 1
    assert d._complexity == 1
    assert d.complexity == 1


def test_generate():
    names = ['foo', 'bar', 'baz', 'qux', 'meep', 'moop']
    domains = [1.0, 1, True, False, None, 'foo']

    for name, domain in zip(names, domains):
        d = ConstantDomain(name, domain)
        assert d.generate() == domain


def test_map_to_domain():
    inputs = [None, 'foo', 1, 1.0, True, False]
    d = ConstantDomain('foo', 1.0)

    for i in inputs:
        assert d.map_to_domain(i) == i


def test_to_index():
    inputs = [None, 'foo', 1, 1.0, True, False]
    d = ConstantDomain('foo', 1.0)

    for i in inputs:
        assert d.to_index(i) == i


def test_to_json():
    names = ['foo', 'bar', 'baz', 'qux', 'meep', 'moop']
    domains = [1.0, 1, True, False, None, 'foo']

    for name, domain in zip(names, domains):
        d = ConstantDomain(name, domain)
        assert d.to_json() == {'name': name, 'domain': domain}
