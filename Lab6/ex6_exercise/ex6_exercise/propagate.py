import numpy as np
import random

# particles: [num_p, state_dim]
# A deterministic matrix: [state_dim, state_dim]

# Because the each particle is a row vector, so we change the prediction function into:
# S_t = S_{t-1} * A + W_{t-1}
# in this case if the model is static, A is a 2 dimensional identity matrix;
# else if the model is with constant velocity, A is a 4 dimensional matrix below.

# And for each particle, we should add noise according to the params.
# Finally make sure they lie in the frame
def propagate(particles, frame_height, frame_width, params):
    next_state = np.zeros_like(particles)
    if params["model"] == 0:
        A = np.eye(2)
        next_state = particles.dot(A)

        for i, element in enumerate(next_state):
            next_state[i, 0] += random.gauss(0, params["sigma_position"])
            next_state[i, 1] += random.gauss(0, params["sigma_position"])

            # make sure the center lies in the frame
            next_state[i, 0] = np.maximum(0, next_state[i, 0])
            next_state[i, 0] = np.minimum(frame_width, next_state[i, 0])
            next_state[i, 1] = np.maximum(0, next_state[i, 1])
            next_state[i, 1] = np.minimum(frame_height, next_state[i, 1])
    else:
        A = [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [1, 0, 1, 0],
             [0, 1, 0, 1]]
        next_state = particles.dot(A)

        for i, element in enumerate(next_state):

            next_state[i, 0] += random.gauss(0, params["sigma_position"])
            next_state[i, 1] += random.gauss(0, params["sigma_position"])
            next_state[i, 2] += random.gauss(0, params["sigma_velocity"])
            next_state[i, 3] += random.gauss(0, params["sigma_velocity"])

            # make sure the center lies in the frame
            next_state[i, 0] = np.maximum(0, next_state[i, 0])
            next_state[i, 0] = np.minimum(frame_width, next_state[i, 0])
            next_state[i, 1] = np.maximum(0, next_state[i, 1])
            next_state[i, 1] = np.minimum(frame_height, next_state[i, 1])
    return next_state
