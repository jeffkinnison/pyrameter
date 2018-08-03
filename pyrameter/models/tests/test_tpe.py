import pytest
from test_model import TestModel

from pyrameter.models import TPEModel
from pyrameter.models.model import Result, Value
from pyrameter import ContinuousDomain, DiscreteDomain

from scipy.stats import uniform


class TestTPEModel(TestModel):
    __model_class__ = TPEModel

    def test_init(self):
        # Test defualt initialization
        m = self.__model_class__()
        assert m.best_split == 0.2
        assert m.n_samples == 10
        assert m.warm_up == 10
        assert m.gmm_kws == {}

        # Test with supplied arguments
        m = self.__model_class__(best_split=0.4, n_samples=20, warm_up=30,
                                 kw='yes')
        assert m.best_split == 0.4
        assert m.n_samples == 20
        assert m.warm_up == 30
        assert m.gmm_kws == {'kw': 'yes'}

        super(TestTPEModel, self).test_init()

    def test_generate(self):
        d1 = ContinuousDomain(uniform, path='a', loc=-100.0, scale=100.0)
        d2 = DiscreteDomain([i for i in range(-100, 0)] +
                            [i for i in range(1, 101)], path='b')

        # Test with one continuous domain
        m = self.__model_class__(domains=[d1])
        for _ in range(1000):
            p = m()[-1]
            m.add_result(Result(m, loss=p['a']**2, values=Value(p['a'], d1)))
            assert 'a' in p
            assert p['a'] >= -100.0 and p['a'] < 100

        # Test with one continuous domain
        m = self.__model_class__(domains=[d2])
        for _ in range(1000):
            p = m()[-1]
            m.add_result(Result(m, loss=p['b']**2, values=Value(p['b'], d2)))
            assert 'b' in p
            assert p['b'] >= -100 and p['b'] <= 100

        # Test with one continuous and one discrete domain
        m = self.__model_class__(domains=[d1, d2])
        for _ in range(1000):
            p = m()[-1]
            m.add_result(Result(m, loss=p['a'] * p['b'],
                                values=[Value(p['a'], d1), Value(p['b'], d2)]))
            assert 'a' in p
            # assert p['a'] >= -100 and p['a'] < 100
            assert 'b' in p
            assert p['b'] >= -100 and p['b'] <= 100
