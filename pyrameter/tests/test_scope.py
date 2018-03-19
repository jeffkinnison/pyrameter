import pytest

from pyrameter import Scope
from pyrameter.domain import Domain, ContinuousDomain, DiscreteDomain, \
                             ExhaustiveDomain
from pyrameter.models import RandomSearchModel, TPEModel, GPBayesModel

from scipy.stats import uniform


class TestScope(object):
    def test_init(self):
        # Test the default initialization.
        s = Scope()
        assert s.children == {}
        assert s.exclusive is False
        assert s.optional is False
        assert s.model is RandomSearchModel

        # Test passing domains in as tuples
        s = Scope(('a', Domain()), (1, Domain()), exclusive=True, model='random')
        assert 'a' in s.children
        assert '1' in s.children
        assert s.exclusive is True
        assert s.optional is False
        assert s.model is RandomSearchModel

        # Test passing domains in as keyword args
        s = Scope(a=Domain(), b=Domain(), optional=True, model='tpe')
        assert 'a' in s.children
        assert 'b' in s.children
        assert s.exclusive is False
        assert s.optional is True
        assert s.model is TPEModel

        # Test passing domains in as both tuples and keyword args
        s = Scope(('a', Domain()), ('b', Domain()), c=Domain(), d=Domain(),
                  exclusive=True, optional=True, model='gp')
        assert 'a' in s.children
        assert 'b' in s.children
        assert 'c' in s.children
        assert 'd' in s.children
        assert s.exclusive is True
        assert s.optional is True
        assert s.model is GPBayesModel

        # Test passing nested scopes as tuples and kws
        s = Scope(('a', Scope(('b', Domain()))), c=Scope(d=Domain()))
        assert 'a' in s.children
        assert isinstance(s.children['a'], Scope)
        assert 'b' in s.children['a'].children
        assert isinstance(s.children['a'].children['b'], Domain)
        assert 'c' in s.children
        assert isinstance(s.children['c'], Scope)
        assert 'd' in s.children['c'].children
        assert isinstance(s.children['c'].children['d'], Domain)

    def test_split(self):
        # Test splitting an empty Scope
        s = Scope()
        assert s.split()[0] == RandomSearchModel()

        # Test splitting with a single domain
        s = Scope(a=ContinuousDomain(uniform))
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/a')])]
        assert s.split() == res

        # Test splitting with a single exclusive domain
        s = Scope(a=ContinuousDomain(uniform), exclusive=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/a')])]
        assert s.split() == res

        # Test splitting with a single optional domain
        s = Scope(a=ContinuousDomain(uniform), optional=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/a')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting with a single exclusive and optional domain
        s = Scope(a=ContinuousDomain(uniform), exclusive=True, optional=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/a')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting with two domains
        s = Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3]))
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/a'),
                                          DiscreteDomain([1, 2, 3], path='/b')])]
        assert s.split() == res

        # Test splitting with two exclusive domains
        s = Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3]),
                  exclusive=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/a')]),
               RandomSearchModel(domains=[DiscreteDomain([1, 2, 3], path='/b')])]
        assert s.split() == res

        # Test splitting with two optional domains
        s = Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3]),
                  optional=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/a'),
                                          DiscreteDomain([1, 2, 3], path='/b')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting with two exclusive and optional domains
        s = Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3]),
                  exclusive=True, optional=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/a')]),
               RandomSearchModel(domains=[DiscreteDomain([1, 2, 3], path='/b')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting with multiple domains
        s = Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3]),
                  c=ExhaustiveDomain([4, 5, 6]))
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/a'),
                                          DiscreteDomain([1, 2, 3], path='/b'),
                                          ExhaustiveDomain([4, 5, 6], path='/c')])]
        assert s.split() == res

        # Test splitting with multiple exclusive domains
        s = Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3]),
                  c=ExhaustiveDomain([4, 5, 6]), exclusive=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/a')]),
               RandomSearchModel(domains=[DiscreteDomain([1, 2, 3], path='/b')]),
               RandomSearchModel(domains=[ExhaustiveDomain([4, 5, 6], path='/c')])]
        assert s.split() == res

        # Test splitting with multiple optional domains
        s = Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3]),
                  c=ExhaustiveDomain([4, 5, 6]), optional=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/a'),
                                          DiscreteDomain([1, 2, 3], path='/b'),
                                          ExhaustiveDomain([4, 5, 6], path='/c')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting with multiple exclusive and optional domains
        s = Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3]),
                  c=ExhaustiveDomain([4, 5, 6]), exclusive=True, optional=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/a')]),
               RandomSearchModel(domains=[DiscreteDomain([1, 2, 3], path='/b')]),
               RandomSearchModel(domains=[ExhaustiveDomain([4, 5, 6], path='/c')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting one nested Scope
        s = Scope(A=Scope(a=ContinuousDomain(uniform)))
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a')])]
        assert s.split() == res

        # Test splitting one nested Scope with a single exclusive domain
        s = Scope(A=Scope(a=ContinuousDomain(uniform), exclusive=True))
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a')])]
        assert s.split() == res

        # Test splitting one nested Scope with a single optional domain
        s = Scope(A=Scope(a=ContinuousDomain(uniform), optional=True))
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting one nested Scope with a single exclusive and optional domain
        s = Scope(A=Scope(a=ContinuousDomain(uniform), exclusive=True, optional=True))
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting one nested Scope with two domains
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3])))
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a'),
                                          DiscreteDomain([1, 2, 3], path='/A/b')])]
        assert s.split() == res

        # Test splitting one nested Scope with two exclusive domains
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3])),
                  exclusive=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a'),
                                          DiscreteDomain([1, 2, 3], path='/A/b')])]
        assert s.split() == res

        # Test splitting one nested Scope with two optional domains
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3])),
                  optional=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a'),
                                          DiscreteDomain([1, 2, 3], path='/A/b')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting one nested Scope with two exclusive and optional domains
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3])),
                  exclusive=True, optional=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a'),
                                          DiscreteDomain([1, 2, 3], path='/A/b')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting one nested exclusive Scope
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3]),
                  exclusive=True))
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a')]),
               RandomSearchModel(domains=[DiscreteDomain([1, 2, 3], path='/A/b')])]
        assert s.split() == res

        # Test splitting one nested optional Scope
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3]),
                  optional=True))
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a'),
                                          DiscreteDomain([1, 2, 3], path='/A/b')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting one nested exclusive, optional Scope
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3]),
                  exclusive=True, optional=True))
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a')]),
               RandomSearchModel(domains=[DiscreteDomain([1, 2, 3], path='/A/b')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting one nested scope and a top-level domain
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4])),
                  c=ExhaustiveDomain([5, 6, 7, 8]))
        res = [RandomSearchModel(domains=[
                    ContinuousDomain(uniform, path='/A/a'),
                    DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                    ExhaustiveDomain([5, 6, 7, 8], path='/c')])]
        assert s.split() == res

        # Test splitting one nested scope and one exclusive top-level domain
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4])),
                  c=ExhaustiveDomain([5, 6, 7, 8]), exclusive=True)
        res = [RandomSearchModel(domains=[
                    ContinuousDomain(uniform, path='/A/a'),
                    DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
               RandomSearchModel(domains=[ExhaustiveDomain([5, 6, 7, 8], path='/c')])]
        assert s.split() == res

        # Test splitting one exclusive nested scope and one top-level domain
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                  exclusive=True), c=ExhaustiveDomain([5, 6, 7, 8]))
        res = [RandomSearchModel(domains=[
                    ContinuousDomain(uniform, path='/A/a'),
                    ExhaustiveDomain([5, 6, 7, 8], path='/c')]),
               RandomSearchModel(domains=[
                    DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                    ExhaustiveDomain([5, 6, 7, 8], path='/c')])]
        assert s.split() == res

        # Test splitting one exclusive nested scope and one exclusive top-level domain
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                  exclusive=True), c=ExhaustiveDomain([5, 6, 7, 8]), exclusive=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a')]),
               RandomSearchModel(domains=[DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
               RandomSearchModel(domains=[ExhaustiveDomain([5, 6, 7, 8], path='/c')])]
        assert s.split() == res

        # Test splitting one nested scope and one optional top-level domain
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4])),
                  c=ExhaustiveDomain([5, 6, 7, 8]), optional=True)
        res = [RandomSearchModel(domains=[
                    ContinuousDomain(uniform, path='/A/a'),
                    DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                    ExhaustiveDomain([5, 6, 7, 8], path='/c')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting one optional nested scope and one top-level domain
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                  optional=True), c=ExhaustiveDomain([5, 6, 7, 8]))
        res = [RandomSearchModel(domains=[
                    ContinuousDomain(uniform, path='/A/a'),
                    DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                    ExhaustiveDomain([5, 6, 7, 8], path='/c')]),
               RandomSearchModel(domains=[
                    ExhaustiveDomain([5, 6, 7, 8], path='/c')])]
        assert s.split() == res

        # Test splitting one optional nested scope and one optional top-level domain
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                  optional=True), c=ExhaustiveDomain([5, 6, 7, 8]), optional=True)
        res = [RandomSearchModel(domains=[
                    ContinuousDomain(uniform, path='/A/a'),
                    DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                    ExhaustiveDomain([5, 6, 7, 8], path='/c')]),
               RandomSearchModel(domains=[ExhaustiveDomain([5, 6, 7, 8], path='/c')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting one nested scope and one exclusive, optional top-level domain
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4])),
                  c=ExhaustiveDomain([5, 6, 7, 8]), exclusive=True, optional=True)
        res = [RandomSearchModel(domains=[
                    ContinuousDomain(uniform, path='/A/a'),
                    DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
               RandomSearchModel(domains=[ExhaustiveDomain([5, 6, 7, 8], path='/c')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting one exclusive, optional nested scope and one top-level domain
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                  exclusive=True, optional=True), c=ExhaustiveDomain([5, 6, 7, 8]))
        res = [RandomSearchModel(domains=[
                    ContinuousDomain(uniform, path='/A/a'),
                    ExhaustiveDomain([5, 6, 7, 8], path='/c')]),
               RandomSearchModel(domains=[
                    DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                    ExhaustiveDomain([5, 6, 7, 8], path='/c')]),
               RandomSearchModel(domains=[
                    ExhaustiveDomain([5, 6, 7, 8], path='/c')])]
        assert s.split() == res

        # Test splitting one exclusive, optional nested scope and one exclusive, optional top-level domain
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                  exclusive=True, optional=True), c=ExhaustiveDomain([5, 6, 7, 8]),
                  exclusive=True, optional=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a')]),
               RandomSearchModel(domains=[DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
               RandomSearchModel(),
               RandomSearchModel(domains=[ExhaustiveDomain([5, 6, 7, 8], path='/c')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting one exclusive nested scope and one exclusive, optional top-level domains
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                  exclusive=True), c=ExhaustiveDomain([5, 6, 7, 8]),
                  exclusive=True, optional=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a')]),
               RandomSearchModel(domains=[DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
               RandomSearchModel(domains=[ExhaustiveDomain([5, 6, 7, 8], path='/c')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting one optional nested scope and one exclusive, optional top-level domains
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                  optional=True), c=ExhaustiveDomain([5, 6, 7, 8]),
                  exclusive=True, optional=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a'),
                                          DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
               RandomSearchModel(),
               RandomSearchModel(domains=[ExhaustiveDomain([5, 6, 7, 8], path='/c')]),
               RandomSearchModel()]
        assert s.split() == res

        # Test splitting one exclusive, optional nested scope and one exclusive top-level domains
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                  exclusive=True, optional=True), c=ExhaustiveDomain([5, 6, 7, 8]),
                  exclusive=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a')]),
               RandomSearchModel(domains=[DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
               RandomSearchModel(),
               RandomSearchModel(domains=[ExhaustiveDomain([5, 6, 7, 8], path='/c')])]
        assert s.split() == res

        # Test splitting one exclusive, optional nested scope and one optional top-level domains
        s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                  exclusive=True, optional=True), c=ExhaustiveDomain([5, 6, 7, 8]),
                  optional=True)
        res = [RandomSearchModel(domains=[ContinuousDomain(uniform, path='/A/a'),
                                          ExhaustiveDomain([5, 6, 7, 8], path='/c')]),
               RandomSearchModel(domains=[DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                                          ExhaustiveDomain([5, 6, 7, 8], path='/c')]),
               RandomSearchModel(domains=[ExhaustiveDomain([5, 6, 7, 8], path='/c')]),
               RandomSearchModel()]
        assert s.split() == res

    # Test splitting two nested Scopes
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4])),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])))
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a'),
                DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')])]
    assert s.split() == res

    # Test splitting two exclusive nested Scopes
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4])),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])),
              exclusive=True)
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a'),
                DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
           RandomSearchModel(domains=[
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')])]
    assert s.split() == res

    # Test splitting two optional nested Scopes
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4])),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])),
              optional=True)
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a'),
                DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel()]
    assert s.split() == res

    # Test splitting two exclusive, optional nested Scopes
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4])),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])),
              exclusive=True, optional=True)
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a'),
                DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
           RandomSearchModel(domains=[
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel()]
    assert s.split() == res

    # Test splitting two nested Scopes, the first exclusive
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                      exclusive=True),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])))
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a'),
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel(domains=[
                DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')])]
    assert s.split() == res

    # Test splitting two exclusive nested Scopes,the first exclusive
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                      exclusive=True),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])),
              exclusive=True)
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a')]),
           RandomSearchModel(domains=[
                DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
           RandomSearchModel(domains=[
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')])]
    assert s.split() == res

    # Test splitting two optional nested Scopes, the first exclusive
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                      exclusive=True),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])),
              optional=True)
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a'),
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel(domains=[
                DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel()]
    assert s.split() == res

    # Test splitting two exclusive, optional nested Scopes, the first exclusive
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                      exclusive=True),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])),
              exclusive=True, optional=True)
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a')]),
           RandomSearchModel(domains=[
                DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
           RandomSearchModel(domains=[
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel()]
    assert s.split() == res

    # Test splitting two nested Scopes, the first optional
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                      optional=True),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])))
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a'),
                DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel(domains=[
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')])]
    assert s.split() == res

    # Test splitting two exclusive nested Scopes,the first optional
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                      optional=True),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])),
              exclusive=True)
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a'),
                DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
           RandomSearchModel(),
           RandomSearchModel(domains=[
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')])]
    assert s.split() == res

    # Test splitting two optional nested Scopes, the first optional
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                      optional=True),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])),
              optional=True)
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a'),
                DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel(domains=[
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel()]
    assert s.split() == res

    # Test splitting two exclusive, optional nested Scopes, the first optional
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                      optional=True),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])),
              exclusive=True, optional=True)
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a'),
                DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
           RandomSearchModel(),
           RandomSearchModel(domains=[
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel()]
    assert s.split() == res

    # Test splitting two nested Scopes, the first exclusive, optional
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                      exclusive=True, optional=True),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])))
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a'),
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel(domains=[
                DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel(domains=[
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')])]
    assert s.split() == res

    # Test splitting two exclusive nested Scopes,the first exclusive, optional
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                      exclusive=True, optional=True),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])),
              exclusive=True)
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a')]),
           RandomSearchModel(domains=[
                DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
           RandomSearchModel(),
           RandomSearchModel(domains=[
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')])]
    assert s.split() == res

    # Test splitting two optional nested Scopes, the first optional
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                      exclusive=True, optional=True),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])),
              optional=True)
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a'),
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel(domains=[
                DiscreteDomain([1, 2, 3, 4], path='/A/b'),
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel(domains=[
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel()]
    assert s.split() == res

    # Test splitting two exclusive, optional nested Scopes, the first optional
    s = Scope(A=Scope(a=ContinuousDomain(uniform), b=DiscreteDomain([1, 2, 3, 4]),
                      exclusive=True, optional=True),
              B=Scope(c=ExhaustiveDomain([5, 6, 7, 8]), d=DiscreteDomain(['a', 'b'])),
              exclusive=True, optional=True)
    res = [RandomSearchModel(domains=[
                ContinuousDomain(uniform, path='/A/a')]),
           RandomSearchModel(domains=[
                DiscreteDomain([1, 2, 3, 4], path='/A/b')]),
           RandomSearchModel(),
           RandomSearchModel(domains=[
                ExhaustiveDomain([5, 6, 7, 8], path='/B/c'),
                DiscreteDomain(['a', 'b'], path='/B/d')]),
           RandomSearchModel()]
    assert s.split() == res
