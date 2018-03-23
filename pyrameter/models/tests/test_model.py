import pytest

from pyrameter.models.model import Model, Result, Value


class TestModel(object):
    __model_class__ = Model

    def test_init(self):
        # Test default initialization
        m = self.__model_class__()
        assert isinstance(m.id, str)
        assert m.domains == []
        assert m.results == []
        assert m.complexity == 1.0
        assert m.priority == 1.0
        assert m.rank is None
        assert m.update_complexity is True
        assert m.domain_added is False
        assert m.priority_update_freq == 10
        assert m.recompute_priority is False

    def test_add_domain(self):
        pass

    def test_add_result(self):
        pass

    def test_copy(self):
        pass

    def test_merge(self):
        pass

    def test_generate(self):
        pass

    def test_complexity(self):
        pass

    def test_priority(self):
        pass

    def test_to_json(self):
        pass


class TestResult(object):
    def test_init(self):
        pass

    def test_to_json(self):
        pass


class TestValue(object):
    def test_init(self):
        pass

    def test_to_json(self):
        pass
