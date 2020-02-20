"""Tools to optimize hyperparameters drawn from a search space.

Classes
-------
FMin
    Minimize an objective function to optimize a set of hyperparameters.
"""
import numpy as np
import scipy.stats

from pyrameter.backend import *
from pyrameter.domains.base import Domain
from pyrameter.domains.joint import JointDomain
import pyrameter.methods
from pyrameter.searchspace import SearchSpace
from pyrameter.specification import Specification


class FMin(object):
    """Minimize an objective function to optimize a set of hyperparameters.

    Parameters
    ----------
    exp_key : str
        Unique key for this experiment (e.g., the name of the experiment,
        name+number, etc.).
    spec
        Specification of the hyperparameter domains to optimize.
    method : {'random','tpe','spearmint','smac'}
        Hyperparameter selection method.
    backend
        Data storage backend.

    Attributes
    ----------
    spec : pyrameter.specification.Specification
    searchspaces : list of pyrameter.searchspace.SearchSpace
    method : callable
    """

    def __init__(self, exp_key, spec, method, backend, max_evals=None):
        self.exp_key = exp_key

        if not isinstance(spec, Specification):
            if isinstance(spec, JointDomain):
                spec = Specification('', **spec.domain)
            elif isinstance(spec, dict):
                spec = Specification('', **spec)
            elif isinstance(spec, Domain):
                spec = Specification('', **{spec.name: spec})
            else:
                spec = Specification('', domain=spec)

        self.spec = spec
        self.searchspaces = [SearchSpace(ss, exp_key=exp_key)
                             for ss in self.spec.split()]
        self.trials = {}
        self.active = [ss for ss in self.searchspaces]

        self.max_evals = max_evals if max_evals is not None else np.inf

        if isinstance(backend, str):
            if '.json' in backend:
                self.backend = JSONBackend(backend)
        else:
            self.backend = backend

        if not isinstance(self.backend, BaseBackend) and self.backend is not None:
            raise ValueError(
                'Provided backend {} is not a valid backend.'.format(self.backend))

        self._did_sort = False

        try:
            self.method = getattr(pyrameter.methods, method)
        except AttributeError:
            self.method = pyrameter.methods.random

    def copy(self):
        return FMin(self.exp_key, self.spec, self.method, self.backend)

    def generate(self, ssid=None):
        """Generate a set of hyperparameters from a search space.

        Parameters
        ----------
        ssid
            The id of the search space to generate from.

        Returns
        -------
        trial : pyrameter.trial.Trial
            The generated hyperparameters and metadata about the search space.

        Raises
        ------
        KeyError, IndexError
            Raised when ``ssid`` is provided and does not map to a search space
            being optimized.
        """
        if ssid is None:
            if self._did_sort:
                probs = scipy.stats.planck.pmf(
                    range(len(self.active)), 0.5)
            else:
                probs = np.ones(len(self.active))
            probs /= probs.sum()
            idx = np.random.choice(np.arange(len(self.active)), p=probs)

            while idx < len(self.active) and self.active[idx].done:
                idx += 1

            try:
                ss = self.active[idx]
                if not ss.done:
                    trial = ss(method=self.method)
                else:
                    trial = None
            except IndexError:
                ss = None
                trial = None
        else:
            ss = [ss for ss in self.active if ss.id == ssid][0]

            if not ss.done:
                trial = ss(method=self.method)
            else:
                trial = None

        if trial is not None:
            self.trials[trial.id] = trial
        if ss and len(ss.trials) >= self.max_evals:
            ss.done = True
        return trial

    def load(self):
        """Load experiment state from the backend."""
        if self.backend is not None:
            self.searchspaces = self.backend.load()
            for s in self.searchspaces:
                for t in s.trials:
                    self.trials[t.id] = t

    def optimum(self):
        """Retrieve the optimal observed set of hyperparameter values."""
        best = None
        for searchspace in self.searchspaces:
            candidate = searchspace.optimum()
            if best is None or candidate.objective < best.objective:
                best = candidate
        return best

    def register_result(self, ssid, trial_id, objective=None, results=None,
                        errmsg=None):
        """Register a result in the proper trial.

        Parameters
        ----------
        ssid : int or `bson.objectid.ObjectId`
            The id of the search space that generated the trial.
        trial_id : int or `bson.objectid.ObjectId`
            The id of the trial that generated the result.
        objective : float, optional
            The value of the objective function generated by the trial.
        results : dict, optional
            Additional results to record with the trial, e.g. timing,
            performance metrics, etc.
        errmsg : str
            Error message output by the trial if it failed.
        """
        ss = [ss for ss in self.searchspaces if str(ss.id) == str(ssid)][0]

        if not isinstance(trial_id, list):
            trial = [t for t in ss.trials if str(t.id) == str(trial_id)][0]
            trial.objective = objective
            trial.results = results
            trial.errmsg = errmsg
            hyperparameters = trial.hyperparameters
            if trial.id not in self.trials:
                self.trials[trial.id] = trial
            trial.submissions += 1

            if ss.complexity == 1:
                n_done = sum([1 for t in ss.trials
                              if t.status.value == 3])
                if n_done > self.max_evals:
                    ss.done = True
                    self.active.remove(ss)
        else:
            hyperparameters = []
            for i, tid in enumerate(trial_id):
                trial = [t for t in ss.trials if str(t.id) == str(tid)][0]
                trial.objective = objective[i]
                trial.results = results[i]
                trial.errmsg = errmsg
                hyperparameters.append(trial.hyperparameters)
                if trial.id not in self.trials:
                    self.trials[trial.id] = trial
                trial.submissions += 1

                if ss.complexity == 1:
                    n_done = sum([1 for t in ss.trials
                                  if t.status.value == 3])
                    if n_done > self.max_evals:
                        ss.done = True
                        self.active.remove(ss)

        return trial.submissions, hyperparameters

    def save(self):
        """Save the state of the experiment."""
        if self.backend is not None:
            self.backend.save(self.searchspaces)

    def sort_spaces(self, use_complexity=True, use_uncertainty=True):
        """Sort the search spaces being optimized with heuristic properties.

        Parameters
        ----------
        use_complexity : bool
            If True, jointly sort by complexity and other flagged heuristics.
        use_uncertainty : bool
            If True, jointly sort by uncertainty and other flagged heuristics.
        """
        for searchspace in self.searchspaces:
            searchspace.rank = 1

        if use_complexity:
            idx = np.argsort([searchspace.complexity
                              for searchspace in self.searchspaces])
            for i in range(idx.shape[0]):
                self.searchspaces[i].rank *= idx[i]

        if use_uncertainty:
            idx = np.argsort([searchspace.uncertainty
                              for searchspace in self.searchspaces])
            for i in range(idx.shape[0]):
                self.searchspaces[i].rank *= idx[i]

        if use_complexity or use_uncertainty:
            self.searchspaces.sort(key=lambda x: x.rank)
            self._did_sort = True

    @property
    def trial_count(self):
        return len(self.trials)
