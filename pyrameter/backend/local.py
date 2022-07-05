"""JSON file backend storage system.

Classes
-------
JSONBackend
    Save search results as JSON objects.
"""

import json
import os
import shutil

from pyrameter.backend.base import BaseBackend
from pyrameter.searchspace import SearchSpace
from pyrameter.utils import PyrameterDecoder, PyrameterEncoder


class JSONBackend(BaseBackend):
    """Save search results as JSON objects.

    Parameters
    ----------
    path : str
        JSON file to save data to.
    backups : int
        The number of backup files to maintain
    """

    def __init__(self, path, backups=1):
        if os.path.isdir(path):
            path = os.path.join(path, 'results.json')

        self.path = os.path.abspath(path)
        self.backups = backups

    def load(self):
        """Load a hyperparameter search state.

        Returns
        -------
        searchspaces : list of pyrameter.domains.SearchSpace
            The experiment state in the JSON file.
        """
        with open(self.path, 'r') as f:
            objs = json.load(f, cls=PyrameterDecoder)
        return [SearchSpace.from_json(obj) for obj in objs]

    def save(self, searchspaces):
        """Save a hyperparameter search state.

        Parameters
        ----------
        searchspaces : list of pyrameter.domains.SearchSpace
            Experiment state to save.
        """
        out = [s.to_json() for s in searchspaces]

        if not os.path.isdir(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if os.path.exists(self.path):
            for i in range(self.backups, 0, -1):
                if i > 1:
                    srcfile = self.path + '.bak.' + str(i - 1)
                else:
                    srcfile = self.path
                destfile = self.path + '.bak.' + str(i)
                shutil.copyfile(srcfile, destfile)

        with open(self.path, 'w') as f:
            json.dump(out, f, cls=PyrameterEncoder)
