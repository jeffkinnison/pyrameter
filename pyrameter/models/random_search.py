from pyrameter.models.model import Model


class RandomSearchModel(Model):
    """Generate hyperparameters for a model via random search.

    Parameters
    ----------
    id : str, optional
        The id of this model. If not supplied, an id will be generated.
    domains : list of ``pyrameter.Domain``, optional
        The domains contained within this model.
    results : list of ``pyrameter.models.Result``, optional
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
    Random search as a viable method for hyperparameter optimization was
    introduced by Bergstra and Bengio [1]_ as an alternative to grid search.

    The complexity and priority heuristics are computed as described by
    Kinnison *et al.* [2]_ .

    References
    ----------
    .. [1] Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter
       optimization. Journal of Machine Learning Research, 13(Feb), 281-305.

    .. [2] Kinnison, J., Kremer-Herman, N., Thain, D., & Scheirer, W. (2017).
       SHADHO: Massively Scalable Hardware-Aware Distributed Hyperparameter
       Optimization. arXiv preprint arXiv:1707.01428.
    """

    TYPE = 'random'

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
        params = []
        for domain in self.domains:
            params.append(domain.generate())
        return params
