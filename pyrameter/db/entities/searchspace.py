"""A disjoint hyperparameter search space to be optimized.

Classes
-------
SearchSpace
    A hyperparameter serch space composed of multiple domains.

"""
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernel import RBF

from pyrameter.db.entities.entity import Entity
from pyrameter.db.entities.domain import Domain
from pyrameter.db.entities.result import Result


class SearchSpace(Entity):
    """A hyperparameter search space composed of multiple domains.

    Attributes
    ----------
    id : int
        Unique id of this search space.
    method : str
        The hyperparameter generation/optimzation to use.
    domains : list of pyrameter.db.entities.Domain
        The hyperparameter domains searched by this space.
    results : list of pyrameter.db.entities.Result
        Hyperparameter evaluation results recorded by this search space.
    complexity : float
        The size of this search space.
    uncertainty : float
        An approximation of the certianty of performance over searched
        hyperparameters.
    uncertainty_update_frequency : int
        The rate at which to update the undertainty approximation with respect
        to the number of evaluated hyperparameterizations.
    """

    __tablename__ = 'search_spaces'

    def __init__(self, method=None, domains=None, results=None,
                 complexity=None, uncertainty=None,
                 uncertainty_update_frequency=None):
        super(SearchSpace, self).__init__()
        self.method = method
        self.domains = domains if domains is not None else []
        self.results = results if results is not None else []
        self.__complexity = complexity
        self.__uncertainty = uncertainty
        self.uncertainty_update_frequency = uncertainty_update_frequency

    @property
    def completed_results(self):
        return [r for r in self.results if r.loss is not None]

    @property
    def complexity(self):
        if self.__complexity is None:
            self.__complexity = 1.0
            for d in self.domains:
                self.__complexity *= d.complexity
        return self.__complexity

    def to_array(self):
        """Convert the results list into an array for analysis.

        Returns
        -------
        vec : numpy.ndarray
            An array with shape (r, v + 1), where r is the number of results in
            this search space and v is the number of hyperparameter values. The
            last entry in each row is the performance (e.g. loss).
        """
        results = self.completed_results
        if results:
            vec = np.zeros((len(results), len(self.domains) + 1),
                           dtype=np.float32)
            for i, res in enumerate(self.results):
                vec[i][-1] += res.loss
                for j, val in enumerate(res.values):
                    vec[i, j] += self.domains[j].map_to_index(val)
        else:
            vec = None
        return vec

    def to_json(self):
        """Serialize this entity as JSON.

        Returns
        -------
        serialized : dict
            A JSON-serializable dictionary representation of this entity.
        """
        return {
            'id': self.id,
            'domains': [d.to_json() for d in self.domains],
            'results': [r.to_json() for r in self.results],
            'method': self.method,
            'complexity': self.complexity,
            'uncertainty': self.uncertainty,
            'uncertainty_update_frequency': self.uncertainty_update_frequency
        }

    @property
    def uncertainty(self):
        vec = self.to_array()
        uf = self.uncertainty_update_frequency
        if uf > 0 and vec.shape[0] % uf:
            split = int(np.ceil(vec.shape[0] *
                        (0.8 if vec.shape[0] < 10 else 1.0)))
            scales = np.zeros((50,), dtype=np.float32)

            for i in range(scales.shape[0]):
                np.random.shuffle(vec)
                features = np.copy(vec[:split, :-1])
                losses = np.reshape(np.copy(vec[:split, -1]), (-1, 1))
                est = np.random.uniform(0.1, 2.0)
                gp = GaussianProcessRegressor(kernel=RBF(length_scale=est),
                                              alpha=1e-5)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp.fit(features, losses)
                scales[i] += (1.0 / gp.kernel.theta[0])

            self.__uncertainty = np.linalg.norm(scales.max() - scales.min())
        return self.__uncertainty
