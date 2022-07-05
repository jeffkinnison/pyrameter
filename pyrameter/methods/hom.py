import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from pygam import GAM

from pyrameter.methods.method import BilevelMethod


class HOM(BilevelMethod):
    """Homotopy Optimization Method for Hyperparameter Optimization

    Parameters
    ----------
    inner_method : `pyrameter.methods.method.Method`
        Optimization method to call within HOM. This should be instantiated
        with its own arguments specified.
    k : float
        The fraction of completed trials used to train one of the GAMs.
        Must be in ``[0, 1]``. Default ``0.5``.
    iterations : int
        The number of minimizations to run to compute the homotopy. Must
        be greater than ``0``. Default ``5``.
    jitter_strength : float
        Strength of the random kick to use for local perturbation search
        around the observed minimum. Default ``0.005``.
    warm_up : int
        The number of random or ``inner_method`` trials to run.
        Default ``20``.

    Attributes
    ----------
    inner_method : `pyrameter.methods.method.Method`
        Optimization method to call within HOM. This should be instantiated
        with its own arguments specified.
    warm_up : int
        The number of random or ``inner_method`` trials to run.
    k : float
        The fraction of completed trials used to train one of the GAMs.
    iterations : int
        The number of minimizations to run to compute the homotopy.
    jitter_strength : float
        Strength of the random kick to use for local perturbation search
        around the observed minimum.
    eps : array_like
        Size of the interval of each hyperparameter domain being searched.
    """

    def __init__(self, inner_method, k=0.5, iterations=5, jitter_strength=0.005,
                 warm_up=20):
        super().__init__(inner_method)
        self.warm_up = warm_up
        self.k = k
        self.iterations = iterations
        self.jitter_strength = jitter_strength
        self.eps = None

    def generate(self, trial_data, domains):
        n_trials = trial_data.shape[0]

        if self.eps is None:
            # Get the effective size of each hyperparameter domain.
            self.eps = np.abs([hi - lo for lo, hi in
                               map(lambda d: d.bounds, domains)])

        # Start with warm up inner method sampling (including random) or
        # inject samples at a standard interval.
        if n_trials < self.warm_up or n_trials % 5 in [0, 2]:
            params = self.inner_method.generate(trial_data, domains)

        # Test points randomly sampled from around the best observed point
        # at a standard interval.
        elif n_trials % 5 in [3, 4]:
            # Extract hyperparameters and losses, and get the index of the
            # best observed hyperparameters/loss pair.
            features, losses = trial_data[:, :-1], trial_data[:, -1].ravel()
            idx = np.argsort(losses)
            best_features = features[idx[0]]

            best_10 = features[idx[:int(len(idx) * 0.1)]]
            params = best_features + \
                np.random.uniform(low=-np.var(best_10) * self.jitter_strength,
                                  high=np.var(best_10) * self.jitter_strength)

        # Compute the new optimal point along the surrogate at a standard
        # interval.
        else:
            # Extract hyperparameters and losses, and get the index of the
            # best observed hyperparameters/loss pair.
            features, losses = trial_data[:, :-1], trial_data[:, -1].ravel()
            idx = np.argsort(losses)[0]

            # Shift the data to have 0 mean and unit variance.
            scaler = StandardScaler(copy=False)
            features = scaler.fit_transform(features, y=losses)

            # Fit one GAM to a subsampling of the most recent trials
            k_recent = int(np.round(n_trials * self.k))
            gam1 = GAM(lam=1e-4, n_splines=25)
            gam1.fit(features[-k_recent:], losses[-k_recent:])
            
            # Fit one GAM to all completed trials.
            gam2 = GAM(lam=1e-4, n_splines=25)
            gam2.fit(features, losses)

            # Create a flattened array of inputs to ``minimize``.
            opt_vars = features[idx]

            # Get the bounds, shifted to the scaled data.
            bounds = np.array([list(d.bounds) for d in domains]).T
            bounds = scaler.transform(bounds)
            bounds = [(b[0], b[1]) for b in bounds.T]

            t = 1.0
            delta = 1 / self.iterations
            x_new = np.zeros((self.iterations, len(domains)))

            for i in range(self.iterations):
                # Minimize with the current X0, GAMs, and threshold.
                res = minimize(
                    fun_gam1,  # this needs to figure
                    opt_vars,  # initial guess
                    args=(gam1, gam2, t),  # additional static parameters
                    method='Nelder-Mead',  # minimization method
                    bounds=bounds,
                    tol=1e-8
                )

                # Update X0 for the next round and record the result.
                opt_vars = res.x
                x_new[i] += opt_vars

                # Reduce the threshold.
                t = t - delta

            # Predict scores for the recorded X values and determine
            # the best.
            f_value = gam2.predict(x_new)
            idx_fv = np.argsort(f_value)[0]

            # Rescale to the original domains.
            params = scaler.inverse_transform(
                np.expand_dims(x_new[idx_fv], axis=0)).ravel()

        return params

def fun_gam1(params, gam1, gam2, t):
    params = np.expand_dims(params, axis=0)
    y1 = gam1.predict(params)
    y2 = gam2.predict(params)
    return (t * y1) + ((1 - t) * y2)
