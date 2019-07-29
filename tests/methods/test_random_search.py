import pytest
from test_model import TestModel

from pyrameter.models import RandomSearchModel
from pyrameter import ContinuousDomain, DiscreteDomain

from scipy.stats import uniform


class TestRandomSearchModel(TestModel):
    __model_class__ = RandomSearchModel

    def test_generate(self):
        d1 = ContinuousDomain(uniform, path='a', loc=0.0, scale=1.0)
        d2 = DiscreteDomain(list(range(1000)), path='b')

        # Test with one continuous domain
        m = self.__model_class__(domains=[d1])
        for _ in range(1000):
            p = m()[-1]
            assert 'a' in p
            assert p['a'] >= 0 and p['a'] < 1

        # Test with one discrete domain
        m = self.__model_class__(domains=[d2])
        for _ in range(1000):
            p = m()[-1]
            assert 'b' in p
            assert p['b'] >= 0 and p['b'] < 1000

        # Test with one continuous and one discrete domain
        m = self.__model_class__(domains=[d1, d2])
        for _ in range(1000):
            p = m()[-1]
            assert 'a' in p
            assert p['a'] >= 0 and p['a'] < 1
            assert 'b' in p
            assert p['b'] >= 0 and p['b'] < 1000

        # Test with one continuous and one discrete domain within one sublevel
        d1.path = 'A/a'
        d2.path = 'A/b'
        m = self.__model_class__(domains=[d1, d2])
        for _ in range(1000):
            p = m()[-1]
            assert 'A' in p
            assert 'a' in p['A']
            assert p['A']['a'] >= 0 and p['A']['a'] < 1
            assert 'b' in p['A']
            assert p['A']['b'] >= 0 and p['A']['b'] < 1000

        # Test with one continuous and one discrete domain within two sublevels
        d1.path = 'B/a'
        d2.path = 'A/b'
        m = self.__model_class__(domains=[d1, d2])
        for _ in range(1000):
            p = m()[-1]
            assert 'A' in p
            assert 'B' in p
            assert 'a' in p['B']
            assert p['B']['a'] >= 0 and p['B']['a'] < 1
            assert 'b' in p['A']
            assert p['A']['b'] >= 0 and p['A']['b'] < 1000
