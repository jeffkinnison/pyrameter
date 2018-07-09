import pytest

from pyrameter.db.local import JsonStorage
from pyrameter.models.model import Model

from pyrameter.models.model import Model, Result, Value
from pyrameter.domain import ContinuousDomain, DiscreteDomain

import os
import weakref

from scipy.stats import uniform

import json
import ast


@pytest.fixture(scope='module')
def dummy_models():
    d1 = ContinuousDomain(uniform, loc=0, scale=1)
    d2 = DiscreteDomain([1, 2, 3, 4])

    models = []

    models.append(Model())
    models.append(Model(domains=[d1]))
    models.append(Model(domains=[d2]))
    models.append(Model(domains=[d1, d2]))

    m = Model()
    r1 = Result(m, loss=0.37276)
    m.add_result(r1)
    models.append(m)

    m = Model(domains=[d1])
    r1 = Result(m, loss=0.0241)
    m.add_result(r1)
    models.append(m)

    m = Model(domains=[d1])
    r1 = Result(m, loss=0.37276)
    r2 = Result(m, loss=0.0241)
    m.add_result(r1)
    m.add_result(r2)
    models.append(m)

    m = Model(domains=[d2])
    r1 = Result(m, loss=0.0241)
    m.add_result(r1)
    models.append(m)

    m = Model(domains=[d2])
    r1 = Result(m, loss=0.37276)
    r2 = Result(m, loss=0.0241)
    m.add_result(r1)
    m.add_result(r2)
    models.append(m)

    m = Model(domains=[d1, d2])
    r1 = Result(m, loss=0.0241)
    m.add_result(r1)
    models.append(m)

    m = Model(domains=[d1, d2])
    r1 = Result(m, loss=0.37276)
    r2 = Result(m, loss=0.0241)
    m.add_result(r1)
    m.add_result(r2)
    models.append(m)

    return models


@pytest.fixture
def save_dummy_models(dummy_models, tmpdir):
    load_path = os.path.join(tmpdir.strpath, 'load')
    if not os.path.isdir(load_path):
        os.mkdir(load_path)
    paths = []
    for m in dummy_models:
        mpath = os.path.join(load_path, m.id)
        with open(mpath, 'w') as f:
            json.dump([m.to_json()], f)
        paths.append(mpath)
    return paths


class TestJsonStorage(object):
    def test_init(self, tmpdir):
        # Test default instantiation
        s = JsonStorage(tmpdir.strpath)
        assert s.path == os.path.join(tmpdir.strpath, 'results.json')
        assert s.backups == 1

        # Test with supplied filename
        s = JsonStorage(os.path.join(tmpdir.strpath, 'foo.json'), keep_previous=5)
        assert s.path == os.path.join(tmpdir.strpath, 'foo.json')
        assert s.backups == 5

        # Test with nonexistent path
        with pytest.raises(OSError):
            JsonStorage('/foo/bar')

        with pytest.raises(OSError):
            JsonStorage('/foo/bar/baz.json')

    def test_load(self, tmpdir, dummy_models, save_dummy_models):
        for path in save_dummy_models:
            s = JsonStorage(path)
            loaded = s.load()
            for l in loaded:
                assert l in dummy_models

    def test_save(self, tmpdir, dummy_models):
        s = JsonStorage(tmpdir.strpath)

        # Test with no models
        s.save([])
        assert os.path.isfile(os.path.join(tmpdir.strpath, 'results.json'))

        # Test with single model
        models = []
        models.append(Model())
        s.save(models)

        json_list = []
        for model in models:
            json_list.append(model.to_json())
        json_list = json.dumps(json_list)

        with open(os.path.join(tmpdir.strpath, 'results.json')) as json_file:
            data = json.load(json_file)
            data = json.dumps(data)

        assert data == json_list

        # Test with multiple models
        models = dummy_models
        s.save(models)

        json_list = []
        for model in models:
            json_list.append(model.to_json())
        json_list = json.dumps(json_list)

        with open(os.path.join(tmpdir.strpath, 'results.json')) as json_file:
            data = json.load(json_file)
            data = json.dumps(data)

        assert data == json_list
