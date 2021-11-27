from pyrameter.domains.continuous import ContinuousDomain
import numpy as np
from scipy.stats import uniform

from pyrameter.methods.method import PopulationMethod


class PSO(PopulationMethod):
    """Particle swarm optimization with persistent state.

    Arguments
    ---------
    population_size : int
        The number of concurrent parameter sets to optimize. Default: ``50``.
    omega : float
        Velocity scaling at each update. Default: ``0.5``.
    phi_p : float
        Scaling for the update based on the best observed parameter set in the
        current population. Default: ``0.5``.
    phi_g : float
        Scaling for the update based on the best observed parameter set over
        all generations of the search. Default: ``0.5``.
    delta : float
        Default: ``0.0001``.
    epsilon : float
        Default: ``0.0001``.

    Attributes
    ----------
    population_size : int
        The number of concurrent parameter sets to optimize.
    velocities : np.ndarray
        Buffer of velocity values for each member of the current population.
    best : np.ndarray
        The values of the best performing parameter set ordered by domain.
    fmin : float
        Global best objective value over all generations.
    omega : float
        Velocity scaling at each update (sort of like learning rate).
    phi_p : float
        Scaling for the update based on the best observed parameter set in the
        current population.
    phi_g : float
        Scaling for the update based on the best observed parameter set over
        all generations of the search.
    delta : float
        
    epsilon : float
    """
    def __init__(self, population_size=50, omega=0.5, phi_p=0.5, phi_g=0.5, delta=0.0001, epsilon=0.0001):
        self.population_size = population_size
        self.velocities = None
        self.pbest = None
        self.pfmin = None
        self.gbest = None
        self.gfmin = None
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.delta = delta
        self.epsilon = epsilon

    def init_velocities(self, domains):
        """Initialize velocities based on the 

        Parameters
        ----------
        domains : list of pyrameter.domains.base.Domain
            The domains from which the initial population was drawn from.
            Velocities are initialized using the viable range of the domains.
        """
        velocities = np.zeros((len(domains), self.population_size))
        
        # Draw ``self.population_size`` random values from a uniform
        # distribution bounded by the "viable range" of a domain. For
        # categorical data, the viable range is [0, len(domain)]. For
        # continuous data, the viable range is the interval over which
        # 99.999% of the CDF is defined.
        for i, d in enumerate(domains):
            lo, hi = d.bounds
            velocities[i] += uniform.rvs(loc=lo, scale=(hi - lo),
                                         size=(self.population_size,))
        self.velocities = velocities.T

    def generate(self, population_data, domains):
        # Initialize velocities if they are not.
        if self.velocities is None:
            self.init_velocities(domains)
        
        # Prep the previous population data.
        prev_pop = population_data
        prev_fmins = prev_pop[:, -1].ravel()
        prev_pop = prev_pop[:, :-1]

        # Get the overall best and current-generation best hyperparameter
        # values. On first iteration, set the values up. On subsequent
        # iterations, update the overall best as necessary.
        if self.pfmin is None:
            self.pbest = prev_pop
            self.pfmin = prev_fmins

            generation_best = np.argmin(prev_pop[:, -1].ravel())
            generation_fmin = prev_fmins[generation_best]
            generation_best = prev_pop[generation_best]

            self.gbest = generation_best
            self.gfmin = generation_fmin
        else:
            for i, p in enumerate(prev_fmins):
                if p < self.pfmin[i]:
                    self.pbest[i] = prev_pop[i]
                    self.pfmin[i] = p
                    
                    if p < self.gfmin:
                        self.gbest = prev_pop[i]
                        self.gfmin = p

        # Compute the exploration (pop_term) and exploitation (gen_term)
        # components of the update. This computes two updates based on the
        # difference between the two best observed particles and the
        # current population.
        r_p, r_g = uniform.rvs(loc=0, scale=1, size=(2,))
        pop_term = self.phi_p * r_p * (self.pbest - prev_pop)
        gen_term = self.phi_g * r_g * (self.gbest - prev_pop)

        # Decay the velocities and update with the two terms.
        self.velocities *= self.omega
        self.velocities += (pop_term + gen_term)
        pop = prev_pop + self.velocities

        return pop
