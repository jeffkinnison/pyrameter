import pytest
from test_model import TestModel

from pyrameter.models.gp import GPBayesModel
from pyrameter.models.model import Result, Value
from pyrameter import ContinuousDomain, DiscreteDomain

from scipy.stats import uniform
from sklearn.gaussian_process.kernels import RBF, Matern


class TestGPBayesModel(TestModel):
    __model_class__ = GPBayesModel

    def test_init(self):
        # Test default initialization
        m = GPBayesModel()
        assert isinstance(m.id, str)
        assert m.domains == []
        assert m.results == []
        assert m.update_complexity is True
        assert m.priority_update_freq == 10
        assert m.n_samples == 10
        assert m.warm_up == 10
        assert 'kernel' in m.gp_kws
        assert isinstance(m.gp_kws['kernel'], RBF)

        super(TestGPBayesModel, self).test_init()

    def test_generate(self):
        d1 = ContinuousDomain(uniform, path='a', loc=-100.0, scale=100.0)
        d2 = DiscreteDomain([i for i in range(-100, 0)] +
                            [i for i in range(1, 101)], path='b')

        # Test with one continuous domain
        m = self.__model_class__(domains=[d1], kernel=Matern())
        print(m.domains)
        for _ in range(100):
            p = m()[-1]
            m.add_result(Result(m, loss=p['a']**2, values=Value(p['a'], d1)))
            assert 'a' in p
            assert p['a'] >= -100.0 and p['a'] < 100

        # Test with one continuous domain
        m = self.__model_class__(domains=[d2], kernel=Matern())
        for _ in range(100):
            p = m()[-1]
            m.add_result(Result(m, loss=p['b']**2, values=Value(p['b'], d2)))
            assert 'b' in p
            assert p['b'] >= -100 and p['b'] <= 100

        # Test with one continuous and one discrete domain
        m = self.__model_class__(domains=[d1, d2], kernel=Matern())
        for _ in range(100):
            p = m()[-1]
            m.add_result(Result(m, loss=p['a'] * p['b'],
                                values=[Value(p['a'], d1), Value(p['b'], d2)]))
            assert 'a' in p
            # assert p['a'] >= -100 and p['a'] < 100
            assert 'b' in p
            assert p['b'] >= -100 and p['b'] <= 100
