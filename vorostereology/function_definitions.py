import numpy as np
from itertools import combinations
from math import sqrt

# Author: Thomas van der Jagt


def new_alpha(weights, previous_weights, volumes, previous_volumes, alpha):
    denominator = np.dot(volumes - previous_volumes, weights - previous_weights)
    # Check for zero division, this is extremely rare but it may happen.
    # In that case the alpha from the previous iteration is returned.
    if np.abs(denominator) > 1e-15:
        return np.sum(np.square(weights - previous_weights)) / denominator
    return alpha


def new_parameters_MM(weights, previous_weights, volumes, previous_volumes, alpha, theta, num_iter):
    if num_iter == 0:
        new_alpha_ = 0.5 * np.linalg.norm(weights - previous_weights) / np.linalg.norm(volumes - previous_volumes)
        new_theta = new_alpha_ / alpha
    else:
        new_alpha_ = min(sqrt(1 + theta) * alpha, 0.5 * np.linalg.norm(weights - previous_weights) /
                         np.linalg.norm(volumes - previous_volumes))
        new_theta = new_alpha_ / alpha
    return new_alpha_, new_theta


def permute_targets(volumes, target_volumes, n):
    # Implements the permutation step for the volumes
    volumes_sort_indices = np.argsort(volumes)
    target_sort_indices = np.argsort(target_volumes)
    mapping = {volumes_sort_indices[i]: target_sort_indices[i] for i in range(n)}
    return target_volumes[list(map(mapping.get, np.arange(n)))]
