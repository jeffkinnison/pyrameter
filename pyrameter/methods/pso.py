from pyrameter.domains.continuous import ContinuousDomain
import numpy as np
from scipy.stats import uniform

from pyrameter.methods.random import random_search


class PSO(object):
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
        Velocity scaling at each update.
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

    def __call__(self, space):
        if not all([isinstance(d, ContinuousDomain) for d in space.domains]):
            raise TypeError('All search domains must be continuous to use PSO')

        if space.population is None or len(space.population) == 0:
            pop = [random_search(space) for _ in range(self.population_size)]
            velocities = np.zeros((len(space.domains), self.population_size))
            for i, d in enumerate(space.domains):
                lo, hi = d.bounds
                velocities[i] += uniform.rvs(loc=lo, scale=(hi - lo), size=(self.population_size,))
            self.velocities = velocities.T
            
        else:
            prev_pop = space.population_to_array()
            prev_fmins = prev_pop[:, -1].ravel()
            prev_pop = prev_pop[:, :-1]


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

            r_p, r_g = uniform.rvs(loc=0, scale=1, size=(2,))
            pop_term = self.phi_p * r_p * (self.pbest - prev_pop)
            gen_term = self.phi_g * r_g * (self.gbest - prev_pop)

            self.velocities *= self.omega
            self.velocities += (pop_term + gen_term)
            pop = prev_pop + self.velocities

            for i, p in enumerate(pop):
                for j, d in enumerate(space.domains):
                    lo, hi = d.bounds
                    p[j] = min(hi, max(lo, d.map_to_domain(p[j])))

        return pop


# def pso(space, population_size=50):
#     """Particle-swarm optimization for hyperparameter selection.

    
#     """
#     best = None
#     f_best = None
#     velocities = None
#     omega = 0.5
#     phi_p = 0.5
#     phi_g = 0.5
#     delta = 0.0001
#     epsilon = 0.0001
#     print('in pso')

#     while True:
#         if len(space.population) == 0:
#             pop = [random_search(space) for _ in range(space.population_size)]
#             velocities = np.zeros((len(pop), len(pop[0])), dtype=np.float64)
#         else:
#             prev_pop = [[] for _ in range(population_size)]
#             for p in space.population:
#                 vec = []
#                 for i, d in space.domains:
#                     vec.append([d.to_index(p.hyperparameters[i])])
#                 prev_pop.append(vec)
#             del vec
#             prev_pop = np.stack([p.hyperparameters for p in space.population], axis=0)
#             prev_f = np.array([p.objective for p in space.population])


#             s_star = prev_pop[np.argmin(prev_f)]
#             f_s_star = prev_f[np.argmin(prev_f)]

#             r_p, r_g = uniform.rvs(loc=0, scale=1, size=2)
#             pop_term = phi_p * r_p * (prev_pop - s_star)
#             gen_term = phi_g * r_g * (prev_pop - best)
#             velocities = omega * velocities + pop_term + gen_term
#             pop = prev_pop + velocities

#             for i, p in enumerate(pop):
#                 for j, d in enumerate(space.domains):
#                     p[j] = d.map_to_domain(p[j])

#         yield pop

