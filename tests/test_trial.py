import pytest

from pyrameter.domains import *
from pyrameter.searchspace import SearchSpace
from pyrameter.trial import Trial, TrialStatus


def test_init():
    s = SearchSpace([ConstantDomain('A', 8)])
    t = Trial(s)
    assert t.searchspace() is s
    assert t.hyperparameters is None
    assert t.results is None
    assert t.objective is None
    assert t.errmsg is None
    assert t.dirty
    assert t.status == TrialStatus.INIT

    s = SearchSpace([ConstantDomain('A', 8)])
    t = Trial(s, hyperparameters=[8])
    assert t.searchspace() is s
    assert t.hyperparameters == [8]
    assert t.results is None
    assert t.objective is None
    assert t.errmsg is None
    assert t.dirty
    assert t.status == TrialStatus.READY

    s = SearchSpace([ConstantDomain('A', 8)])
    t = Trial(s, hyperparameters=[8], results={'loss': 10}, objective=10)
    assert t.searchspace() is s
    assert t.hyperparameters == [8]
    assert t.results == {'loss': 10}
    assert t.objective == 10
    assert t.errmsg is None
    assert t.dirty
    assert t.status == TrialStatus.DONE

    s = SearchSpace([ConstantDomain('A', 8)])
    t = Trial(s, hyperparameters=[8], results={'loss': 10}, objective=10,
              errmsg='HI!')
    assert t.searchspace() is s
    assert t.hyperparameters == [8]
    assert t.results == {'loss': 10}
    assert t.objective == 10
    assert t.errmsg == 'HI!'
    assert t.dirty
    assert t.status == TrialStatus.ERROR


def test_parameter_dict():
    s = SearchSpace([ConstantDomain('A', 8)])
    t = Trial(s, hyperparameters=[8])
    assert t.parameter_dict == {'A': 8}


    s = SearchSpace([ConstantDomain('/A', 8), ConstantDomain('/B/a/b', 2)])
    t = Trial(s, hyperparameters=[8, 2])
    assert t.parameter_dict == {'A': 8, 'B': {'a': {'b': 2}}}

    s = SearchSpace([ConstantDomain('/A', 8), ConstantDomain('/B/a/b', 2),
                     ConstantDomain('/B/a/c', 4)])
    t = Trial(s, hyperparameters=[8, 2, 4])
    assert t.parameter_dict == {'A': 8, 'B': {'a': {'b': 2, 'c': 4}}}


def test_to_json():
    s = SearchSpace([ConstantDomain('A', 8)])
    t = Trial(s)
    assert t.to_json() == {'searchspace': s.id,
                           'status': TrialStatus.INIT.value,
                           'hyperparameters': None,
                           'results': None,
                           'objective': None,
                           'errmsg': None}

    s = SearchSpace([ConstantDomain('A', 8)])
    t = Trial(s, hyperparameters=[8])
    assert t.to_json() == {'searchspace': s.id,
                           'status': TrialStatus.READY.value,
                           'hyperparameters': [8],
                           'results': None,
                           'objective': None,
                           'errmsg': None}

    s = SearchSpace([ConstantDomain('A', 8)])
    t = Trial(s, hyperparameters=[8], objective=0.374)
    assert t.to_json() == {'searchspace': s.id,
                           'status': TrialStatus.READY.value,
                           'hyperparameters': [8],
                           'results': None,
                           'objective': 0.374,
                           'errmsg': None}

    s = SearchSpace([ConstantDomain('A', 8)])
    t = Trial(s, hyperparameters=[8], objective=0.374, errmsg='hi')
    assert t.to_json() == {'searchspace': s.id,
                           'status': TrialStatus.ERROR.value,
                           'hyperparameters': [8],
                           'results': None,
                           'objective': 0.374,
                           'errmsg': 'hi'}
