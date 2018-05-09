import pytest

from pyrameter.db.local import JsonStorage

import os


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
        pass
