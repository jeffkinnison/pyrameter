import pytest
from test_model import TestModel

from pyrameter.models import GPBayesModel


class TestGPBayesModel(TestModel):
    __model_class__ = GPBayesModel

    def test_init(self):
        super(TestGPBayesModel, self).test_init()

    def test_generate(self):
        pass
