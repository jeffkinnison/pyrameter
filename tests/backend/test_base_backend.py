import pytest

from pyrameter.backend.base import BaseBackend


def test_load():
    b = BaseBackend()
    with pytest.raises(NotImplementedError):
        b.load()


def test_save():
    b = BaseBackend()
    with pytest.raises(NotImplementedError):
        b.save()
