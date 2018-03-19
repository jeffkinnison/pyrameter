import pytest
from test_model import TestModel

from pyrameter.models import RandomSearchModel


class TestRandomSearchModel(TestModel):
    __model_class__ = RandomSearchModel

    def test_generate(self):
        pass
