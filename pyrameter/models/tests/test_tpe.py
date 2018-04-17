import pytest
from test_model import TestModel

from pyrameter.models import TPEModel


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
        pass
