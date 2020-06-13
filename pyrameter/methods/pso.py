import numpy as np
from scipy.stats import uniform

from pyramter.methods.random import random_search


def pso(space):
    """Particle-swarm optimization for hyperparameter selection.

    
    """
    best = None
    f_best = None
    velocities = None
    omega = 0.5
    phi_p = 0.5
    phi_g = 0.5
    delta = 0.0001
    epsilon = 0.0001

    while True:
        if len(space.population) == 0:
            pop = [random_search(space) for _ in range(space.population_size)]
            velocities = np.zeros((len(pop), len(pop[0])), dtype=np.float64)
        else:
            prev_pop = [[] for _ in range(space.population_size)]
            for p in space.population:
                vec = []
                for i, d in space.domains:
                    vec.append([d.to_index(p.hyperparameters[i])])
                prev_pop.append(vec)
            del vec
            prev_pop = np.stack([p.hyperparameters for p in space.population], axis=0)
            prev_f = np.array([p.objective for p in space.population])


            s_star = prev_pop[np.argmin(prev_f)]
            f_s_star = prev_f[np.argmin(prev_f)]

            r_p, r_g = uniform.rvs(loc=0, scale=1, size=2)
            pop_term = phi_p * r_p * (prev_pop - s_star)
            gen_term = phi_g * r_g * (prev_pop - best)
            velocities = omega * velocities + pop_term + gen_term
            pop = prev_pop + velocities

            for i, p in enumerate(pop):
                for j, d in enumerate(space.domains):
                    p[j] = d.map_to_domain(p[j])

        yield pop

