import pytest

from pyrameter.models import get_model_class, RandomSearchModel, TPEModel, \
                             GPBayesModel
from pyrameter.models.model_factory import InvalidModelError


def test_get_model_class():
    model_classes = {
        'random': RandomSearchModel,
        'tpe': TPEModel,
        'gp': GPBayesModel,
    }

    for key, val in model_classes.items():
        assert get_model_class(key) is val
        assert get_model_class(val()) is val

    invalids = [None, True, False, 1, 1.0, '1', [], {}, tuple(), set()]
    for i in invalids:
        with pytest.raises(InvalidModelError):
            get_model_class(i)
