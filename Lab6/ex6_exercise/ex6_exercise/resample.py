import numpy as np
import random


def resample(particles, weights):
    """
    resample the particles based on their weights,
    and return these new particles along with their corresponding weights.
    """
    particles_updated = np.zeros_like(particles)
    weights_updated = np.zeros_like(weights)
    weights = weights / sum(weights)
    idx_resample = np.random.choice(range(particles.shape[0]), particles.shape[0], p=weights)
    for i in range(particles.shape[0]):
        particles_updated[i] = particles[idx_resample[i], :]
        weights_updated[i] = weights[idx_resample[i]]
    return particles_updated, weights_updated
