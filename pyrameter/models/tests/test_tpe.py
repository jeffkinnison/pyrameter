import pytest
from test_model import TestModel

from pyrameter.models import TPEModel


class TestTPEModel(TestModel):
    __model_class__ = TPEModel

    def test_init(self):
        super(TestTPEModel, self).test_init()

    def test_generate(self):
        pass
