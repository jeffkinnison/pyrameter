"""MongoDB backend storage engine.

Classes
-------
MongoBackend
    Save search results in a MongoDB database.
"""

import pymongo
from pymongo.operations import UpdateOne
from bson.objectid import ObjectId

from pyrameter.backend.base import BaseBackend


class MongoBackend(BaseBackend):
    """Save search results in a MongoDB database.

    Parameters
    ----------
    url : str
        MongoDB url to connect to the database instance.
        (e.g., 'mongodb://localhost:27017')
    database : str
        The name of the database to create/access. This name should be unique
        to the search being conducted, e.g. the experiment key.

    Attributes
    ----------
    connection : ``pymongo.database.Database``
        Connection to the database for this experiment.
    """

    def __init__(self, url, database):
        self.connection = pymongo.MongoClient(url)[database]

    def load(self):
        """Load a hyperparameter search state.

        Returns
        -------
        searchspaces : list of `pyrameter.domains.searchspace.SearchSpace`
            The search spaces in the eperiment as of their most recent save.
        """
        searchspaces = self.connection['searchspaces'].find()
        for searchspace in searchspaces:
            domains = self.connection['domains'].find(
                {'_id': {'$in': searchspace['domains']}})
            results = self.connection['trials'.find(
                {'_id': {'$in': searchspace['trials']}}
            )]

    def save(self, searchspaces):
        """Save a hyperparameter search state.

        Parameters
        ----------
        searchspaces : list of pyrameter.domains.SearchSpace
            Experiment state to save.
        """
        ssupdates = [UpdateOne({'_id': ss.id},
                               ss.to_json(simplify=True),
                               upsert=True)
                     for ss in searchspaces]
        result = self.connection['searchspaces'].bulk_write(ssupdates)
        for key, val in result.upserted_ids.items():
            searchspaces[key].id = val

        domainset = set()
        trials = []
        for ss in searchspaces:
            domainset = domainset.union(set([d for d in ss.domains]))
            trials.extend([r for r in ss.results if r.dirty])

        domainset = list(domainset)
        domupdates = [UpdateOne({'_id': domain.id},
                                domain.to_json(),
                                upsert=True)
                      for domain in domainset]
        result = self.connection['domains'].bulk_write(domupdates)
        for key, val in result.upserted_ids.items():
            domainset[key].id = val

        trialupdates = [UpdateOne({'_id': trial.id},
                                  trial.to_json(),
                                  upsert=True)
                        for trial in trials]
        result = self.connection['trials'].bulk_write(trialupdates)
        for key, val in result.upserted_ids.items():
            trials[key].id = val
