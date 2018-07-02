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
def setup_dummy_models():
    d1 = ContinuousDomain(uniform, loc=0, scale=1)
    d2 = DiscreteDomain([1, 2, 3, 4])

    r1 = Result(None, loss=0.37276)
    r2 = Result(None, loss=1.346)

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
    r1 = Result(m, loss=0.37276)
    m.add_result(r1)
    models.append(m)

    m = Model(domains=[d1])
    r1 = Result(m, loss=0.37276)
    m.add_result(r1)
    models.append(m)

    return models


@pytest.fixture(scope='module')
def save_dummy_models():
    pass


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

    def test_load(self, tmpdir, setup_dummy_models):
        s = JsonStorage(tmpdir.strpath)

        models = setup_dummy_models

        # convert models to json
        json_models = []
        for model in models:
            if isinstance(model, Model):
                m = model.to_json()
            json_models.append(m)

        # save models to file
        with open(os.path.join(tmpdir.strpath, 'results.json'), 'w') as json_file:
            json.dump(json_models, json_file)

        loaded = s.load()

        assert loaded == models


    def test_save(self, tmpdir, setup_dummy_models):
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
        models = setup_dummy_models
        s.save(models)

        json_list = []
        for model in models:
            json_list.append(model.to_json())
        json_list = json.dumps(json_list)
        
        with open(os.path.join(tmpdir.strpath, 'results.json')) as json_file:
            data = json.load(json_file)
            data = json.dumps(data)

        assert data == json_list

