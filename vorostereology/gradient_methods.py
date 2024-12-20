import numpy as np
from . import voroplusplus
from math import sqrt
from scipy.sparse.linalg import minres


def permute_targets(volumes, target_volumes, n):
    # Implements the permutation step for the volumes
    volumes_sort_indices = np.argsort(volumes)
    target_sort_indices = np.argsort(target_volumes)
    mapping = {volumes_sort_indices[i]: target_sort_indices[i] for i in range(n)}
    return target_volumes[list(map(mapping.get, np.arange(n)))]


def dn_solver(weights_init, points, target_volumes, domain, periodic, permute=False,
              max_iter=100, tol=0.01, disp=False):
    """
    For now this function is not fully documented, most arguments are inherited from "compute_centroidal_laguerre".
        This function finds the right weights to obtain a Laguerre diagram with a desired volume distribution by
        minimizing a convex function using a damped Newton method.
    :param weights_init:
    :param points:
    :param target_volumes:
    :param domain:
    :param periodic:
    :param permute:
    :param max_iter:
    :param tol:
    :param disp:
    :param num_cpus:
    :return:
    """
    weights = weights_init
    n = weights.shape[0]

    # we cannot take the square root of negative weights, but adding a constant to all weights does not change
    # the Laguerre diagram.
    min_weight = np.min(weights)
    if min_weight < 0:
        weights -= min_weight
    volumes = voroplusplus.get_3d_volumes(weights, points, domain, periodic)
    epsilon0 = 0.5*min(np.min(volumes), np.min(target_volumes))
    if epsilon0 == 0.0:
        weights = np.zeros(n)
        volumes = voroplusplus.get_3d_volumes(weights, points, domain, periodic)
        epsilon0 = 0.5 * min(np.min(volumes), np.min(target_volumes))

    # Optional, permutation step might speed up the algorithm in some cases
    if permute:
        target_volumes_ = permute_targets(volumes, target_volumes, volumes.shape[0])
    else:
        target_volumes_ = target_volumes

    residual = np.max(np.abs(volumes - target_volumes_))
    stopping_criterion = np.min(target_volumes) * tol
    minimization_num_iterations = 0
    
    while residual > stopping_criterion and minimization_num_iterations < max_iter:
        hess = voroplusplus.get_3d_hessian(weights, points, domain, periodic)
        direction, _ = minres(hess, volumes - target_volumes_, tol=1e-08)

        line_search_iteration = 1
        old_dist = np.linalg.norm(volumes - target_volumes_)
        while True:
            new_weights = weights + (0.5 ** line_search_iteration) * direction

            min_new_weight = np.min(new_weights)
            if min_new_weight < 0:
                new_weights -= min_new_weight

            new_volumes = voroplusplus.get_3d_volumes(new_weights, points, domain, periodic)
            new_dist = np.linalg.norm(new_volumes - target_volumes_)
            if np.min(new_volumes) >= epsilon0 and new_dist <= (1 - 0.5 ** (line_search_iteration + 1)) * old_dist:
                weights = new_weights
                volumes = new_volumes
                break
            else:
                line_search_iteration += 1

        # Optional, permutation step might speed up the algorithm in some cases
        if permute:
            target_volumes_ = permute_targets(volumes, target_volumes_, n)

        # Update residual for the next iteration
        residual = np.max(np.abs(volumes - target_volumes_))

        if disp:
            print("Minimization iteration: ", minimization_num_iterations)
        minimization_num_iterations += 1

    if residual > stopping_criterion:
        return weights, False
    elif disp:
        print("Minimization number of iterations: " + str(minimization_num_iterations))

    return weights, True


def bb_solver(weights_init, points, target_volumes, domain, periodic, permute=False,
              max_iter=10000, tol=0.01, disp=False):
    """
    For now this function is not fully documented, most arguments are inherited from "compute_centroidal_laguerre".
        This function finds the right weights to obtain a Laguerre diagram with a desired volume distribution by
        minimizing a convex function using the Barzilai-Borwein method.
    :param weights_init:
    :param points:
    :param target_volumes:
    :param domain:
    :param periodic:
    :param permute:
    :param max_iter:
    :param tol:
    :param disp:
    :param num_cpus:
    :return:
    """
    l1 = domain[0][1] - domain[0][0]
    l2 = domain[1][1] - domain[1][0]
    l3 = domain[2][1] - domain[2][0]
    weights = weights_init
    n = weights.shape[0]
    diameter = np.sqrt(l1*l1 + l2*l2 + l3*l3)

    # we cannot take the square root of negative weights, but adding a constant to all weights does not change
    # the Laguerre diagram.
    min_weight = np.min(weights)
    if min_weight <= 0:
        weights = weights - min_weight + diameter
    volumes = voroplusplus.get_3d_volumes(weights, points, domain, periodic)
    previous_volumes = volumes

    # Optional, permutation step might speed up the algorithm in some cases
    if permute:
        target_volumes_ = permute_targets(volumes, target_volumes, volumes.shape[0])
    else:
        target_volumes_ = target_volumes

    residual = np.max(np.abs(volumes - target_volumes_))
    stopping_criterion = np.min(target_volumes) * tol
    previous_weights = weights
    minimization_num_iterations = 0
    alpha = 0.1 / ((l1*l2*l3) ** (1./3))

    while residual > stopping_criterion and minimization_num_iterations < max_iter:
        # Update weights and calculate volumes of the tessellation with these new weights
        weights = previous_weights - alpha * (volumes - target_volumes_)
        # we cannot take the square root of negative weights, but adding a constant to all weights does not change
        # the Laguerre diagram.
        min_weight = np.min(weights)
        if min_weight <= 0:
            weights = weights - min_weight + diameter
            previous_weights = previous_weights - min_weight + diameter
        volumes = voroplusplus.get_3d_volumes(weights, points, domain, periodic)

        # Optional, permutation step might speed up the algorithm in some cases
        if permute:
            target_volumes_ = permute_targets(volumes, target_volumes_, n)

        # Update residual for the next iteration
        residual = np.max(np.abs(volumes - target_volumes_))

        # Update variables for the next iteration
        denominator = np.dot(volumes - previous_volumes, weights - previous_weights)
        # Check for zero division, this is extremely rare but it may happen.
        if np.abs(denominator) > 1e-15:
            alpha = np.sum(np.square(weights - previous_weights)) / denominator

        previous_volumes = volumes
        previous_weights = weights

        if disp and minimization_num_iterations and minimization_num_iterations % 100 == 0:
            print("Minimization iteration: ", minimization_num_iterations)
        minimization_num_iterations += 1

    if residual > stopping_criterion:
        return weights, False
    elif disp:
        print("Minimization number of iterations: " + str(minimization_num_iterations))

    return weights, True


def mm_solver(weights_init, points, target_volumes, domain, periodic, permute=False,
              max_iter=10000, tol=0.01, disp=False):
    """
    For now this function is not fully documented, most arguments are inherited from "compute_centroidal_laguerre".
        This function finds the right weights to obtain a Laguerre diagram with a desired volume distribution by
        minimizing a convex function using the Malitsky-Mishchenko method.
    :param weights_init:
    :param points:
    :param target_volumes:
    :param domain:
    :param periodic:
    :param permute:
    :param max_iter:
    :param tol:
    :param disp:
    :param num_cpus:
    :return:
    """
    l1 = domain[0][1] - domain[0][0]
    l2 = domain[1][1] - domain[1][0]
    l3 = domain[2][1] - domain[2][0]
    weights = weights_init
    n = weights.shape[0]
    diameter = np.sqrt(l1 * l1 + l2 * l2 + l3 * l3)

    # we cannot take the square root of negative weights, but adding a constant to all weights does not change
    # the Laguerre diagram.
    min_weight = np.min(weights)
    if min_weight <= 0:
        weights = weights - min_weight + diameter
    volumes = voroplusplus.get_3d_volumes(weights, points, domain, periodic)
    previous_volumes = volumes

    # Optional, permutation step might speed up the algorithm in some cases
    if permute:
        target_volumes_ = permute_targets(volumes, target_volumes, volumes.shape[0])
    else:
        target_volumes_ = target_volumes

    residual = np.max(np.abs(volumes - target_volumes_))
    stopping_criterion = np.min(target_volumes) * tol
    previous_weights = weights
    minimization_num_iterations = 0
    theta = 0
    alpha = 0.1 / ((l1*l2*l3) ** (1./3))

    while residual > stopping_criterion and minimization_num_iterations < max_iter:
        # Update weights and calculate volumes of the tessellation with these new weights
        weights = previous_weights - alpha * (volumes - target_volumes_)
        # we cannot take the square root of negative weights, but adding a constant to all weights does not change
        # the Laguerre diagram.
        min_weight = np.min(weights)
        if min_weight <= 0:
            weights = weights - min_weight + diameter
            previous_weights = previous_weights - min_weight + diameter
        volumes = voroplusplus.get_3d_volumes(weights, points, domain, periodic)

        # Optional, permutation step might speed up the algorithm in some cases
        if permute:
            target_volumes_ = permute_targets(volumes, target_volumes_, n)

        # Update residual for the next iteration
        residual = np.max(np.abs(volumes - target_volumes_))

        # Update variables for the next iteration
        if minimization_num_iterations == 0:
            new_alpha = 0.5 * np.linalg.norm(weights - previous_weights) / np.linalg.norm(volumes - previous_volumes)
            theta = new_alpha / alpha
            alpha = new_alpha
        else:
            new_alpha = min(sqrt(1 + theta) * alpha, 0.5 * np.linalg.norm(weights - previous_weights) /
                            np.linalg.norm(volumes - previous_volumes))
            theta = new_alpha / alpha
            alpha = new_alpha

        previous_volumes = volumes
        previous_weights = weights

        if disp and minimization_num_iterations and minimization_num_iterations % 100 == 0:
            print("Minimization iteration: ", minimization_num_iterations)
        minimization_num_iterations += 1

    if residual > stopping_criterion:
        return weights, False
    elif disp:
        print("Minimization number of iterations: " + str(minimization_num_iterations))

    return weights, True
