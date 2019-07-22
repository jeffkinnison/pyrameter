import pytest

from pyrameter.domains import *
from pyrameter.specification import Specification


def test_init():
    # Test default initialization
    s = Specification()
    assert s.name == ''
    assert not s.exclusive
    assert not s.optional
    assert s.children == {}

    s = Specification(name='foo', exclusive=True, optional=True, x=5,
                      y=(1, 2, 3), z=[7, 8, 9], a={'bar': 4},
                      b=JointDomain(name='b', meep='moop'))
    assert s.name == 'foo'
    assert s.exclusive
    assert s.optional
    assert isinstance(s.x, ConstantDomain)
    assert s.x.domain == 5
    assert isinstance(s.y, SequenceDomain)
    assert s.y.domain == (1, 2, 3)
    assert isinstance(s.z, DiscreteDomain)
    assert s.z.domain == [7, 8, 9]
    assert isinstance(s.a, Specification)
    assert s.a.name == 'a'
    assert isinstance(s.a.bar, ConstantDomain)
    assert s.a.bar.domain == 4
    assert isinstance(s.b, Specification)
    assert s.b.name == 'b'
    assert isinstance(s.b.meep, ConstantDomain)
    assert s.b.meep.domain == 'moop'


def test_split():
    s = Specification()
    split = s.split()
    assert split == [[]]

    s = Specification(exclusive=True)
    split = s.split()
    assert split == []

    s = Specification(optional=True)
    split = s.split()
    assert split == [[], []]

    s = Specification(exclusive=True, optional=True)
    split = s.split()
    assert split == [[]]

    s = Specification(a=5)
    split = s.split()
    assert split == [[s.a]]

    s = Specification(a=5, exclusive=True)
    split = s.split()
    assert split == [[s.a]]

    s = Specification(a=5, optional=True)
    split = s.split()
    assert split == [[s.a], []]

    s = Specification(a=5, exclusive=True, optional=True)
    split = s.split()
    assert split == [[s.a], []]

    s = Specification(a=5, b='bar')
    split = s.split()
    assert split == [[s.a, s.b]]

    s = Specification(a=5, b='bar', exclusive=True)
    split = s.split()
    assert split == [[s.a], [s.b]]

    s = Specification(a=5, b='bar', optional=True)
    split = s.split()
    assert split == [[s.a, s.b], []]

    s = Specification(a=5, b='bar', exclusive=True, optional=True)
    split = s.split()
    assert split == [[s.a], [s.b], []]

    s = Specification(meep={'a': 5, 'b': 'bar'})
    split = s.split()
    assert split == [[s.meep.a, s.meep.b]]

    s = Specification(meep={'a': 5, 'b': 'bar'}, exclusive=True)
    split = s.split()
    assert split == [[s.meep.a, s.meep.b]]

    s = Specification(meep={'a': 5, 'b': 'bar'}, optional=True)
    split = s.split()
    assert split == [[s.meep.a, s.meep.b], []]

    s = Specification(meep={'a': 5, 'b': 'bar'}, exclusive=True, optional=True)
    split = s.split()
    assert split == [[s.meep.a, s.meep.b], []]

    s = Specification(meep={'a': 5, 'b': 'bar', 'exclusive': True})
    split = s.split()
    assert split == [[s.meep.a], [s.meep.b]]

    s = Specification(meep={'a': 5, 'b': 'bar', 'optional': True})
    split = s.split()
    assert split == [[s.meep.a, s.meep.b], []]

    s = Specification(meep={'a': 5, 'b': 'bar', 'exclusive': True, 'optional': True})
    split = s.split()
    assert split == [[s.meep.a], [s.meep.b], []]

    s = Specification(A={'a': 5, 'b': 'bar'},
                      B={'c': 1.0, 'd': 'hoops', 'exclusive': True},
                      C={'e': True, 'f': None, 'optional': True},
                      D={'g': '1209', 'h': (1, 2, 3), 'exclusive': True,
                         'optional': True})
    split = s.split()
    assert split == [[s.A.a, s.A.b, s.B.c, s.C.e, s.C.f, s.D.g],
                     [s.A.a, s.A.b, s.B.c, s.C.e, s.C.f, s.D.h],
                     [s.A.a, s.A.b, s.B.c, s.C.e, s.C.f],
                     [s.A.a, s.A.b, s.B.c, s.D.g],
                     [s.A.a, s.A.b, s.B.c, s.D.h],
                     [s.A.a, s.A.b, s.B.c],
                     [s.A.a, s.A.b, s.B.d, s.C.e, s.C.f, s.D.g],
                     [s.A.a, s.A.b, s.B.d, s.C.e, s.C.f, s.D.h],
                     [s.A.a, s.A.b, s.B.d, s.C.e, s.C.f],
                     [s.A.a, s.A.b, s.B.d, s.D.g],
                     [s.A.a, s.A.b, s.B.d, s.D.h],
                     [s.A.a, s.A.b, s.B.d],
                    ]

    s = Specification(A=ExhaustiveDomain('A', [1, 2, 3, 4]))
    split = s.split()
    assert split == [[ConstantDomain('A', 1)],
                     [ConstantDomain('A', 2)],
                     [ConstantDomain('A', 3)],
                     [ConstantDomain('A', 4)],]

    s = Specification(A=ExhaustiveDomain('A', [1, 2, 3, 4]), B=7)
    split = s.split()
    assert split == [[ConstantDomain('A', 1), s.B],
                     [ConstantDomain('A', 2), s.B],
                     [ConstantDomain('A', 3), s.B],
                     [ConstantDomain('A', 4), s.B],]
