import numpy as np

# just calculate the weighted average of particles to the estimated next state
def estimate(particles, particles_w):
    """
        :particles: size[num_p, state_dim]
        :particles_w: size[num_p, 1]
        Return
        :mean state: size[1, state_dim]
    """
    particles_w = np.array(particles_w)
    # print(particles_w)
    weighted_p = np.dot(particles_w.T, particles)
    # print(weighted_p / np.sum(particles_w))
    return weighted_p / np.sum(particles_w)