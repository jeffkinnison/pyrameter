import copy
import uuid
import weakref

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


class Model(object):
    """Hierarchical hyperparameter search tree matching a learning model.

    This class manages hyperparameter domains and search information for a
    single machine learning model (e.g., SVM with RBF kernel, 5-layer vs
    10-layer neural network, etc.).

    Parameters
    ----------
    id : str, optional
        The id of this model. If not supplied, an id will be generated.
    domains : list of ``arbor.Domain``, optional
        The domains contained within this model.
    results : list of ``arbor.models.Result``, optional
        The results observed from evaluting different hyperparameterizations of
        this model.
    update_complexity : bool, optional
        Whether to update model complexity when adding a new domain. Enabled by
        default.
    priority_update_freq : int, optional
        How often the priority heuristic is updated, based on the number of
        results. Values less than or equal to 0 disable priority updates.
        Default: update every 10 results.

    Notes
    -----
    The complexity and priority hearustics are computed as described by
    Kinnison *et al.* [1]_ .

    References
    ----------
    .. [1] Kinnison, J., Kremer-Herman, N., Thain, D., & Scheirer, W. (2017).
       SHADHO: Massively Scalable Hardware-Aware Distributed Hyperparameter
       Optimization. arXiv preprint arXiv:1707.01428.
    """
    def __init__(self, id=None, domains=None, results=None,
                 update_complexity=True, priority_update_freq=10):
        self.id = str(uuid.uuid4())
        self.domains = [] if domains is None else domains
        self.results = [] if results is None else results

        self._priority = 1.0
        self._complexity = 1.0
        self.rank = None

        self.update_complexity = update_complexity
        self.domain_added = True
        self.priority_update_freq = priority_update_freq
        self.recompute_priority = True

    def add_domain(self, domain):
        """Add a domain to this model.

        Parameters
        ----------
        domain : arbor.Domain
            The new domain to include in this model.

        Notes
        -----
        If complexity updates are enabled, adding a domain with this method
        will trigger a recalculation of the complexity.
        """
        self.domains.append(domain)
        self.domain_added = True

    def add_result(self, result):
        """Add a result to this model.

        Parameters
        ----------
        domain : arbor.Domain
            The new domain to include in this model.

        Notes
        -----
        If complexity updates are enabled, adding a domain with this method
        will trigger a recalculation of the complexity.
        """
        self.results.append(result)
        should_update = (len(self.results) % priority_update_freq == 0)
        if not self.recompute_priority and should_update:
            self.recompute_priority = True

    def copy(self):
        """Make a copy of this model.

        Returns
        -------
        A new ``arbor.models.Model`` instance with copies of all model
        attributes.
        """
        m = Model(domains=[d for d in self.domains],
                  results=[r for r in self.results],
                  update_complexity=self.update_complexity,
                  priority_update_freq=self.priority_update_freq)
        return m

    def merge(self, other):
        """Merge the domains of two models."""
        # TODO: Implement results merging in a sane way (placeholder vals?)
        self.domains.extend(other.domains)
        # self.results.extend(other.results)

    def __results_to_feature_vector(self):
        """Convert hyperparameter values to a feature vector.

        For use with methods that model the function mapping hyperparameter
        values to their performance.

        Returns
        -------
        An array with shape (r, v + 1), where r is the number of results in
        this model and v is the number of hyperparameter values. The last
        entry in each row is the performance (e.g. loss).
        """
        vec = np.zeros((len(self.results), len(self.results[0]) + 1),
                       dtype=np.float32)
        for i in range(len(self.results)):
            vec[i, -1] += self.results[i].loss
            for j in range(len(self.results[i].values)):
                vec[i, j] += self.results[i].values[j]
        return vec

    def generate(self):
        """Generate hyperparameter values.

        Randomly generates hyperparameter values from each domain in this
        model.

        Returns
        -------
        A dictionary containing hyperparameter values structured the same as
        the specification tree that created this model.

        Notes
        -----
        This method implements random search as described by Bergstra and
        Bengio [1]_ . Subclasses should be override this to implement different
        hyperparameter generation methodologies.

        References
        ----------
        .. [1] Bergstra, J., & Bengio, Y. (2012). Random search for
           hyper-parameter optimization. Journal of Machine Learning Research,
           13(Feb), 281-305.
        """
        params = {}
        for domain in self.domains:
            path = domain.path.split('/')
            curr = params
            for p in path[:-1]:
                if p not in curr:
                    curr[p] = {}
                curr = curr[p]
            curr[path[-1]] = domain.generate()
        return params

    @property
    def complexity(self):
        # Only compute complexity if requested and an update is necessary
        if self.update_complexity and self.domain_added:
            self._complexity = 0.0
            for domain in self.domain:
                self._complexity += domain.complexity
        return self._complexity

    @property
    def priority(self):
        # Only compute priority if requested and an update is necessary
        if self.priority_update_freq >= 0 and self.recompute_priority:
            vec = self.__results_to_feature_vector()

            split = int(np.ceil(vec.shape[0] *
                        (0.8 if vec.shape[0] < 10 else 1.0)))
            scales = np.zeros((50,), dtype=np.float32)

            for _ in range(scales.shape[0]):
                np.shuffle(vec)
                features = np.copy(vec[:split, :-1])
                losses = np.reshape(np.copy(vec[:split, -1]), (-1, 1))
                est = np.random.uniform(0.1, 2.0)
                gp = GaussianProcessRegressor(kernel=RBF(length_scale=est))
                gp.fit(features, losses)
                scales[i] += (1.0 / gp.kernel.theta[0])

            self._priority = np.linalg.norm(scales.max() - scales.min())

        return self.priority

    def to_json(self):
        """Convert the model into a JSON-serializable format.

        Returns
        -------
        A dictionary containing all relevant model attributes in a valid
        JSON-serializable format.
        """
        return {
            'domains': [d.to_json() for d in self.domains],
            'results': [r.to_json() for r in self.results],
            'priority': self.priority,
            'complexity': self.complexity,
            'rank': self.rank,
            'update_complexity': self.update_complexity,
            'priority_update_freq': self.priority_update_freq
        }


class Result(object):
    """A hyperparameter evaluation result for a given model.

    Parameters
    ----------
    model : ``arbor.models.Model``
        The model associated with this result.
    loss : float, optional
        The performance of the hyperparameterized model.
    results : dict, optional
        Additional performance inforamtion and results (e.g., accuracy, recall,
        running time, etc.).
    values : list of ``arbor.models.Value``, optional
        The values that generated this result.

    """
    def __init__(self, model, loss=None, results=None, values=None):
        self.model = weakref.ref(model)
        self.values = [] if values is None else values
        self.loss = loss
        self.results = results
        self.values = [] if values is None else values

    def to_json(self):
        """Convert this result into a JSON-serializable format.

        Returns
        -------
        A dictionary containing all relevant result attributes in a valid
        JSON-serializable format.
        """
        return {
            'loss': self.loss,
            'results': self.results,
            'values': [v.to_json() for v in self.values],
            'model': self.model().id
        }


class Value(object):
    """Container for generated hyperparameter values.

    This class exists as a convenient means of mapping hyperparameter values
    to both the generating domain and the returned performance.

    Parameters
    ----------
    value
        The hyperparameter value.
    domain : ``arbor.domains.Domain``
        The domain that generated this value.
    result : ``arbor.models.Result``
        The result generated by this value.
    """
    def __init__(self, value, domain, result):
        self.value = value
        self.domain = weakref.ref(domain)
        self.result = weakref.ref(result)

    def to_json(self):
        """Convert this result into a JSON-serializable format.

        Returns
        -------
        A dictionary containing all relevant result attributes in a valid
        JSON-serializable format.
        """
        return {
            'value': self.value,
            'domain': self.domain().id,
            'result': self.result().id
        }
