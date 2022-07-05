import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from pyrameter.methods.method import BilevelMethod


class NCQS(BilevelMethod):
    """Template for a class-based NCQS implementation.

    Parameters
    ----------
    inner_method : `pyrameter.methods.method.Method`
        Optimization method to call within NCQS. This should be instantiated
        with its own arguments specified.
    jitter_strength : float

    warm_up : int
        The number of random or ``inner_method`` trials to run.

    Attributes
    ----------
    inner_method : `pyrameter.methods.method.Method`
        Optimization method to call within NCQS. This should be instantiated
        with its own arguments specified.
    warm_up : int
        The number of random or ``inner_method`` trials to run.
    eps : array_like
        Size of the interval of each hyperparameter domain being searched.
    mean_eps : float
        Mean of ``eps``.
    Mu_indices : tuple of array_like
        Precomputed upper-triangular indices of the M matrix for lookup.
    """
    def __init__(self, inner_method, jitter_strength=0.05, warm_up=20):
        super().__init__(inner_method)
        self.warm_up = warm_up
        self.jitter_strength = jitter_strength
        self.eps = None
        self.Mu_indices = None

    def generate(self, trial_data, domains):
        n_trials = trial_data.shape[0]

        if self.eps is None:
            # Get the effective size of each hyperparameter domain.
            self.eps = np.abs([hi - lo for lo, hi in
                                map(lambda d: d.bounds, domains)])

        if self.Mu_indices is None:
            # Since the number of hyperparameters does not change, precompute
            # the indices of the upper triangular for later.
            # Expanded for clarity.
            self.Mu_indices = np.nonzero(
                np.triu(
                    np.ones((len(domains), len(domains)))
                )
            )

        # Start with warm up inner method sampling (including random) or
        # inject samples at a standard interval.
        if n_trials < self.warm_up or n_trials % 5 in [0, 2]:
            params = self.inner_method.generate(trial_data, domains)
        
        # Test points randomly sampled from around the best observed point
        # at a standard interval.
        elif trial_data.shape[0] % 5 in [3, 4]:
            # Extract hyperparameters and losses, and get the index of the
            # best observed hyperparameters/loss pair.
            features, losses = trial_data[:, :-1], trial_data[:, -1].ravel()
            idx = np.argsort(losses)
            best_features = features[idx[0]]

            best_10 = features[idx[:int(len(idx) * 0.1)]]
            params = best_features + \
                np.random.uniform(low=-np.var(best_10) * self.jitter_strength, high=np.var(best_10) * self.jitter_strength)

        # Compute the new optimal point along the surrogate at a standard
        # interval.
        else:
            # Extract hyperparameters and losses, and get the index of the
            # best observed hyperparameters/loss pair.
            features, losses = trial_data[:, :-1], trial_data[:, -1].ravel()
            idx = np.argsort(losses)[0]

            scaler = StandardScaler(copy=False)
            features = scaler.fit_transform(features, y=losses)

            distance_decay = self.eps / (n_trials * 0.01)
            if np.linalg.norm(distance_decay) < np.linalg.norm(self.eps) * 0.2:
                distance_decay = self.eps * 0.2

            # Static features are unaltered
            X = features 
            o = losses

            # Optimized values are initialized.
            M = np.identity(len(domains))[self.Mu_indices].ravel()
            Y = features[idx]
            fY = losses[idx]

            # Create a flattened array of inputs to ``minimize``.
            # Expanded for clarity.
            opt_vars = np.concatenate([M, trial_data[idx]], axis=0).ravel()

            # Only tell the optimizer to bound the hyperparameter values it
            # is optimizing.
            bounds = [(-np.inf, np.inf) for _ in range(self.Mu_indices[0].shape[0])]
            bounds.extend([d.bounds for d in domains])
            bounds.append((-np.inf, np.inf))

            static_args = (X, o, distance_decay, self.Mu_indices)

            res = minimize(
                surrogate,      # function to optimize
                opt_vars,       # initial guess
                static_args,    # additional static parameters
                bounds=bounds,  # bounds for each entry in the guess
                tol=1e-6)
            
            params = res.x[self.Mu_indices[0].shape[0]:-1]
            params = scaler.inverse_transform(np.expand_dims(params, axis=0)).ravel()

        return params


def surrogate(opt_params, *args):
    """
    Parameters
    ----------
    opt_params : array_like
        1D array of shape ``(N * (N + 1) + 1,)``, where ``N`` is the number of
        hyperparameters, containing the Hessian matrix ``M``, ``Y`` and ``fY``
        to be optimized.
    X : array_like
        2D array of shape ``(P, N)`` containing evaluated hyperparameter sets.
    loss : array_like
        1D array of shape ``(P,)`` containing the loss corresponding to each
        evaluated hyperparameter set in ``X``.
    

    Returns
    -------
    float
    """
    X, o, distance_decay, Mu_indices = args

    # Indexing information
    n = X.shape[1]
    Mu_offset = Mu_indices[0].shape[0]

    # Extract the three optimization inputs.
    M = np.zeros((n, n)) 
    M[Mu_indices] = opt_params[:Mu_offset]
    Y = opt_params[Mu_offset:-1]
    b = opt_params[-1]

    # Each entry in ``d`` is a flag set to True if the corresponding
    # hyperparameter set is in range of Y and False otherwise. On cast,
    # True evaluates to 1.0 and False evaluates to 0.0.
    d = np.less(np.linalg.norm(X - Y), np.linalg.norm(distance_decay)).astype(np.float32).ravel()
    
    # Compute the dot product ``X_i dot Mu`` for every ``X_i`` simultaneously.
    # This results in an output of shape ``(P, N)``. Then, the remainder of
    # f(X) is computed
    Mx_yprod = np.dot(np.expand_dims(X - Y, axis=1), M).squeeze()
    Mx_yprodnorm = 0.5 * np.sum(np.power(Mx_yprod, 2), axis=1)
    fX = Mx_yprodnorm + b
    return np.sum(d * (fX - o))
