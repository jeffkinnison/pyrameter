import pytest

from pyrameter.models import get_model_class
from pyrameter.models import RandomSearchModel, TPEModel, GPBayesModel


def test_get_model_class():
    model_classes = {
        'random': RandomSearchModel,
        'tpe': TPEModel,
        'gp': GPBayesModel,
    }

    for key, val in model_classes.items():
        assert get_model_class(key) is val
        assert get_model_class(val()) is val
