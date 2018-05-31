import pytest

from pyrameter.db.local import JsonStorage
from pyrameter.models.model import Model

from pyrameter.models.model import Model, Result, Value
from pyrameter.domains import ContinuousDomain, DiscreteDomain

import os
import weakref

from scipy.stats import uniform


@pytest.scope('module')
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

    def test_load(self, tmpdir):
        pass

    def test_save(self, tmpdir):
        s = JsonStorage(tmpdir.strpath)
        assert s.path == os.path.join(tmpdir.strpath, 'results.json')

        # Test with no models
