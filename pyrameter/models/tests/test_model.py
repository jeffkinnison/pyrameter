import pytest

from pyrameter.models.model import Model, Result, Value, InvalidDomainError, \
                                   InvalidResultError

from pyrameter.domain import Domain


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

        # Test initializing with domains and results


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
        m1 = Model()
        m2 = Model()

        # Test default initialization
        r = Result(m1)
        assert r.model() == m1
        assert r.loss is None
        assert r.results is None
        assert r.values == []



    def test_to_json(self):
        pass


class TestValue(object):
    def test_init(self):
        m1 = Model()
        m2 = Model()

        r1 = Result(m1)
        r2 = Result(m2)

        d1 = Domain()
        d2 = Domain()

        # Test initializing with various values, results, and domains
        values = [None, True, False, 1, 1.0, '1', (1,), [1], {'1': 1}, {1}]

        for val in values:
            v = Value(val, d1, r1)
            assert v.value == val
            assert v.result() is r1
            assert v.domain() is d1

            v = Value(val, d2, r1)
            assert v.value == val
            assert v.result() is r1
            assert v.domain() is d2

            v = Value(val, d1, r2)
            assert v.value == val
            assert v.result() is r2
            assert v.domain() is d1

            v = Value(val, d2, r2)
            assert v.value == val
            assert v.result() is r2
            assert v.domain() is d2

        # Test initializing with invalid domains
        for val in values:
            with pytest.raises(InvalidDomainError):
                Value(val, val, r1)
            with pytest.raises(InvalidDomainError):
                Value(val, val, r2)

        # Test initializing with invalid results
        for val in values:
            with pytest.raises(InvalidResultError):
                Value(val, d1, val)
            with pytest.raises(InvalidResultError):
                Value(val, d2, val)



    def test_to_json(self):
        pass
