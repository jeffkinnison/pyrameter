"""Hyperparameter search space specification.

Classes
-------
Specification
    Easy bindings for specifying hyperparameter domains.
"""

import copy
import itertools
import os

from pyrameter.domains import Domain, ConstantDomain, ContinuousDomain, \
                              DiscreteDomain, ExhaustiveDomain, JointDomain, \
                              RepeatedDomain, SequenceDomain 


class Specification(object):
    """Easy bindings for specifying hyperparameter domains.

    Parameters
    ----------
    name : str
        The name of this search space.
    exclusive : bool
        If True, only generate values from one member of this search space at a
        time.
    optional : bool
        If True, either generate from all members of this search space or none
        of them.

    Attributes
    ----------
    children : dict
        The collection of child domains and search spaces contained in this
        search space.
    """

    def __init__(self, name='', exclusive=False, optional=False, **kwargs):
        self.name = name
        self.exclusive = exclusive
        self.optional = optional
        kwargs.pop('name', None)
        kwargs.pop('exclusive', None)
        kwargs.pop('optional', None)

        self.children = {}

        for key, val in kwargs.items():
            setattr(self, key, val)

    def __hash__(self):
        return hash(self.name)

    def __contains__(self, key):
        return (key in self.children)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __getattr__(self, key):
        if 'children' in self.__dict__ and key in self.__dict__['children']:
            return self.children[key]
        return self.__dict__[key]

    def __setattr__(self, key, val):
        if key not in ['name', 'exclusive', 'optional', 'children']:
            if isinstance(val, dict):
                self.children[key] = Specification(name=key, **val)
            elif isinstance(val, JointDomain):
                self.children[key] = Specification(name=key, **val.domain)
            elif isinstance(val, list):
                self.children[key] = DiscreteDomain(key, val)
            elif isinstance(val, tuple):
                self.children[key] = SequenceDomain(key, val)
            elif isinstance(val, RepeatedDomain) and isinstance(val.domain[0], JointDomain):
                self.children[key] = RepeatedDomain(key, Specification(**val.domain[0].domain), val.repetitions)
            elif isinstance(val, (Domain, Specification)):
                copyval = copy.deepcopy(val)
                copyval.name = key
                # val.name = key
                self.children[key] = copyval
            else:
                self.children[key] = ConstantDomain(key, val)
        else:
            self.__dict__[key] = val

    def __getitem__(self, key):
        return self.children[key]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def split(self, root=None):
        """Convert this search space into a directed graph.

        Parameters
        ----------
        root : string
            The root name of this search space for generating paths.
        """
        if root is None:
            root = ''

        # Gather all sets of domains corresponding to differnt machine learning
        # algorithms to be hyperparameterized.
        domainsets = [[]] if not self.exclusive else []

        # Iterate over all domains in this sepcification
        for key, val in self.children.items():
            val.name = '.'.join([root, val.name])

            # Recurse into nested specifications and merge the results
            if isinstance(val, Specification):
                subdomainsets = val.split(root=val.name)
                if self.exclusive:
                    domainsets.extend(subdomainsets)
                else:
                    new_domainsets = []
                    for ds1 in domainsets:
                        for ds2 in subdomainsets:
                            new_domainsets.append(ds1 + ds2)
                    domainsets = new_domainsets
            elif isinstance(val, ExhaustiveDomain):
                new_domainsets = []
                for i, entry in enumerate(val.domain):
                    newval = ConstantDomain(val.name, entry)
                    if self.exclusive:
                        domainsets.append([newval])
                    else:
                        for ds1 in domainsets:
                            new_domainsets.append(ds1 + [newval])
                if len(new_domainsets) > 0:
                    domainsets = new_domainsets
            elif isinstance(val, RepeatedDomain) and val.should_split:
                new_domainsets = []
                split = val.split()
                if self.exclusive:
                    for s in split:
                        if isinstance(s, Specification):
                            s = s.split(root=val.name)
                    domainsets.extend(split)
                else:
                    for ds1 in domainsets:
                        for ds2 in split:
                            subdomain = []
                            if isinstance(ds2.domain[0], Specification):
                                for d in ds2.domain:
                                    subdomain.extend(itertools.chain.from_iterable(d.split(root=val.name)))
                            else:
                                subdomain.extend(ds2.domain)
                            new_domainsets.extend([ds1 + subdomain])
                domainsets = new_domainsets
            else:
                if self.exclusive:
                    domainsets.append([val])
                else:
                    for ds1 in domainsets:
                        ds1.append(val)

        if self.optional:
            domainsets.append([])

        return domainsets
