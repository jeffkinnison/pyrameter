"""Hyperparameter search space specification.

Classes
-------
Specification
    Easy bindings for specifying hyperparameter domains.
"""

import copy
import os

from pyrameter.domains import *


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

    def __getattr__(self, key):
        if key in self.children:
            return self.children[key]
        return self.__dict__[key]

    def __setattr__(self, key, val):
        if key not in ['exclusive', 'optional']:
            name = os.path.join(self.name, key)
            if isinstance(val, dict):
                self.children[key] = SearchSpace(name, **val)
            elif isinstance(val, list):
                self.children[key] = DiscreteDomain(name, val)
            elif isinstance(val, tuple):
                self.children[key] = SequenceDomain(name, val)
            elif isinstance(val, Domain):
                self.children[key] = val
            else:
                self.children[key] = ConstantDomain(name, val)
        else:
            self.__dict__[key] = val

    def __getitem__(self, key):
        return self.children[key]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def build_graph(self, root=None):
        """Convert this search space into a directed graph.

        Parameters
        ----------
        root : string
            The root name of this search space for generating paths.
        """
        G = nx.OrderedMultiDiGraph()

        if root is None:
            root = ''

        root = os.path.join(root, self.name)
        G.add_node(root)

        nodesets = [[]] if not self.exclusive else []

        for key, val in self.children.items():
            val.name = os.path.join(root, val.name)
            if isinstance(val, SearchSpace):
                sub, subnodesets = val.build_graph()
                G.update(edges=sub.edges(), nodes=sub.nodes)
                G.add_edge(root, val.name)
                if self.exclusive:
                    nodesets.extend(subnodesets)
                else:
                    new_nodesets = []
                    for ns1 in nodesets:
                        for ns2 in subnodesets:
                            new_nodesets.append(ns1 + ns2)
                    nodesets = new_nodesets
            elif isinstance(val, ExhaustiveDomain):
                new_nodesets = []
                for i, entry in enumerate(val.domain):
                    newval = ConstantDomain(val.name, entry)

                    if self.exclusive:
                        nodesets.append([newval])
                    else:
                        for ns1 in nodesets:
                            new_nodesets.append(ns1 + [newval])

                    G.add_node(root, newval)
            else:
                root.add_node(val)
                root.add_edge(root, val)
                if self.exclusive:
                    nodesets.append([val])
                else:
                    for ns1 in nodesets:
                        ns1.append(val)

        if self.optional:
            nodesets.append([])

        return G, nodesets
