import numpy as np
from chi2_cost import chi2_cost
from color_histogram import color_histogram


def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    prob = []
    for i, element in enumerate(particles):
        # print(bbox_width)
        # print(bbox_height)
        # print(element)
        xmin = element[0] - bbox_width / 2
        xmax = element[0] + bbox_width / 2
        ymin = element[1] - bbox_height / 2
        ymax = element[1] + bbox_height / 2
        hist_obs = color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin)
        dist = chi2_cost(hist_obs, hist)
        prob.append(1/np.sqrt(2*np.pi)/sigma_observe*np.exp(-(dist**2)/2/(sigma_observe**2)))
        weight = prob/sum(prob)
    return weight
