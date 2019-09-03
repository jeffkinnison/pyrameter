import json
import os

import pytest

from pyrameter.backend.local import JSONBackend
from pyrameter.domains import *
from pyrameter.searchspace import SearchSpace
from pyrameter.trial import Trial


def test_init(tmpdir):
    b = JSONBackend('results.json')
    assert b.path == os.path.abspath('results.json')
    assert b.backups == 1

    b = JSONBackend(str(tmpdir))
    assert b.path == os.path.abspath(os.path.join(str(tmpdir), 'results.json'))
    assert b.backups == 1


def test_load():
    pass


def test_save():
    j = JSONBackend('results.json')
    searchspace = SearchSpace([ConstantDomain('A', 0)])

    j.save([searchspace])
    with open('results.json', 'r') as f:
        objs = json.load(f)

    assert os.path.exists(j.path) and os.path.isfile(j.path)
    assert SearchSpace.from_json(objs[0]) == searchspace

    searchspace2 = SearchSpace([ConstantDomain('B', 12)])
    searchspace2()
    print(len(searchspace2.trials))

    j.save([searchspace2])
    assert os.path.exists(j.path) and os.path.isfile(j.path)
    backup = j.path + '.bak.1'
    assert os.path.exists(backup) and os.path.isfile(backup)

    with open(backup, 'r') as f:
        objs = json.load(f)
    assert SearchSpace.from_json(objs[0]) == searchspace

    with open(j.path, 'r') as f:
        objs = json.load(f)
    assert SearchSpace.from_json(objs[0]) == searchspace2
