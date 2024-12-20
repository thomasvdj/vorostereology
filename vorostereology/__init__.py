from . import voroplusplus
from . import gradient_methods as gm
from . import cross_sections as cs
from . import tessellation as ts
import numpy as np
import os
import warnings


# Author: Thomas van der Jagt
# TODO separate 2D and 3D Laguerre classes. 3D has cross section method. 2D has get_areas instead of get_volumes
# TODO class CrossSection
# TODO check duplicates in cross section


class Laguerre3D(object):
    def __init__(self, points, weights, domain, periodic=False):
        self.points = points
        self.weights = weights
        self.domain = domain
        self.periodic = periodic

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, new_points):
        # Verify that 'new_points' is indeed a numpy array.
        if type(new_points) is not np.ndarray:
            raise TypeError("Expected type 'numpy.ndarray' for attribute 'points'. Received type: " +
                            str(type(new_points)))
        # Verify that the dimensions of points and weights match. During initialization weights does not exist yet,
        # hence the need for the exception.
        try:
            if self._weights.shape[0] != new_points.shape[0]:
                raise ValueError("Dimensions of the arrays 'weights' and 'points' do not match.")
        except AttributeError:
            pass
        if new_points.shape[1] != 3:
            raise ValueError("'points' have to be coordinates in 3D space (x, y, z).")
        # Verify that the dimensions of 'points' and 'domain' match. Also verify that all points are located in
        # the domain. During initialization 'domain' does not exist yet, hence the need for the exception.
        try:
            if new_points.shape[1] != len(self._domain):
                raise ValueError("Dimensions of the 'domain' and the 'points' should match")
            if np.min(new_points[:, 0]) < self._domain[0][0] or np.max(new_points[:, 0]) > self._domain[0][1]:
                raise ValueError("Not all points in 'points' are located inside the domain.")
            if np.min(new_points[:, 1]) < self._domain[1][0] or np.max(new_points[:, 1]) > self._domain[1][1]:
                raise ValueError("Not all points in 'points' are located inside the domain.")
            if np.min(new_points[:, 2]) < self._domain[2][0] or np.max(new_points[:, 2]) > self._domain[2][1]:
                raise ValueError("Not all points in 'points' are located inside the domain.")
        except AttributeError:
            pass
        # If all conditions are met we set 'points' to 'new_points'.
        self._points = new_points

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        # Verify that 'new_weights' is indeed a numpy array.
        if type(new_weights) is not np.ndarray:
            raise TypeError("Expected type 'numpy.ndarray' for attribute 'weights'. Received type: " +
                            str(type(new_weights)))
        # Verify that the number of 'points' is equal to the number of 'weights'.
        if new_weights.shape[0] != self._points.shape[0]:
            raise ValueError("Dimensions of the arrays 'weights' and 'points' do not match.")
        # We may be dealing with an array of shape (n, ) or (n, 1). Verify that 'new_weights' has one of these shapes.
        if len(new_weights.shape) == 2:
            if new_weights.shape[1] != 1:
                raise ValueError("Expected a one-dimensional array for attribute 'weights'.")
        elif len(new_weights.shape) != 1:
            raise ValueError("Expected a one-dimensional array for attribute 'weights'.")
        # Cast array to type double, and to shape (n, )
        w = new_weights.astype(np.float64)
        if len(w.shape) == 2:
            w = np.reshape(w, (w.shape[0],))
        # If negative weights exist add a constant to make them non-negative. This does not change the resulting
        # Laguerre tessellation.
        min_w = np.min(w)
        if min_w < 0:
            w -= min_w
        # If all conditions are met we set 'weights' to 'new_weights'
        self._weights = w

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, new_domain):
        if type(new_domain) is not list and type(new_domain) is not np.ndarray:
            raise TypeError("Expected type 'list' or 'numpy.ndarray' for attribute 'domain'. Received type: " +
                            str(type(new_domain)))
        d = np.array(new_domain, dtype=np.float64)
        if d[0][1] < d[0][0] or d[1][1] < d[1][0] or d[2][1] < d[2][0]:
            raise ValueError("Not a proper domain")
        if np.min(self._points[:, 0]) < d[0][0] or np.max(self._points[:, 0]) > d[0][1]:
            raise ValueError("Not all points in 'points' are located inside the domain.")
        if np.min(self._points[:, 1]) < d[1][0] or np.max(self._points[:, 1]) > d[1][1]:
            raise ValueError("Not all points in 'points' are located inside the domain.")
        if np.min(self._points[:, 2]) < d[2][0] or np.max(self._points[:, 2]) > d[2][1]:
            raise ValueError("Not all points in 'points' are located inside the domain.")
        self._domain = d

    @property
    def periodic(self):
        return self._periodic

    @periodic.setter
    def periodic(self, value):
        if type(value) is not bool:
            raise TypeError("Expected type 'bool' for attribute 'periodic'. Received type: " + str(type(value)))
        self._periodic = value

    def get_volumes(self):
        return voroplusplus.get_3d_volumes(self._weights, self._points, self._domain, self._periodic)

    def get_surface_areas(self):
        return voroplusplus.get_3d_surface_areas(self._weights, self._points, self._domain, self._periodic)

    def get_centroids(self):
        return voroplusplus.get_3d_centroids(self._weights, self._points, self._domain, self._periodic)

    def get_hessian(self):
        return voroplusplus.get_3d_hessian(self._weights, self._points, self._domain, self._periodic)

    def compute_section(self, coeffs, offset, respect_bounds=True):
        return cs.cross_section(self._weights, self._points, self._domain, self._periodic, coeffs, offset,
                                respect_bounds)

    def get_cells(self, respect_bounds=True):
        return ts.compute_voronoi(self._weights, self._points, self._domain, self._periodic, respect_bounds)


def compute_centroidal_laguerre3d(target_volumes, points_init=None, weights_init=None, domain=None, periodic=False,
                                  permute=False, max_iterations=100, gradient_max_iterations=None, tol=0.05,
                                  gradient_tol=0.01, disp=False, fix_num_iterations=False, solver="DN",
                                  respect_bounds=True, rng=None):
    if type(target_volumes) is not np.ndarray:
        raise TypeError("Expected type 'numpy.ndarray' for argument 'target_volumes'. Received type: " +
                        str(type(target_volumes)))
    if np.min(target_volumes) <= 0:
        raise ValueError("The argument 'target_volumes' may only contain strictly positive numbers.")
    n = target_volumes.shape[0]
    if weights_init is None:
        weights_init = np.zeros(n)
    elif weights_init.shape[0] != n:
        raise ValueError("The lengths of 'target_volumes' and 'weights_init' are not equal.")
    if rng is None:
        rng = np.random.default_rng()
    if domain is None:
        if points_init is None:
            side_length = np.sum(target_volumes) ** (1./3)
            domain = [[0, side_length], [0, side_length], [0, side_length]]
            points_init = rng.uniform(low=0.0, high=side_length, size=(n, 3))
        else:
            raise TypeError("When providing the argument 'points_init' you also need to provide the argument 'domain'.")
    else:
        if domain[0][1] < domain[0][0] or domain[1][1] < domain[1][0] or domain[2][1] < domain[2][0]:
            raise ValueError("Not a proper domain")
        if points_init is None:
            pts_x = rng.uniform(low=domain[0][0], high=domain[0][1], size=n)
            pts_y = rng.uniform(low=domain[1][0], high=domain[1][1], size=n)
            pts_z = rng.uniform(low=domain[2][0], high=domain[2][1], size=n)
            points_init = np.column_stack((pts_x, pts_y, pts_z))
        else:
            if points_init.shape[1] != 3:
                raise ValueError("The array 'points_init' should be of shape (n, 3). It has shape: " +
                                 str(points_init.shape))
            if np.min(points_init[:, 0]) < domain[0][0] or np.max(points_init[:, 0]) > domain[0][1]:
                raise ValueError("Not all points in 'points' are located inside the domain.")
            if np.min(points_init[:, 1]) < domain[1][0] or np.max(points_init[:, 1]) > domain[1][1]:
                raise ValueError("Not all points in 'points' are located inside the domain.")
            if np.min(points_init[:, 2]) < domain[2][0] or np.max(points_init[:, 2]) > domain[2][1]:
                raise ValueError("Not all points in 'points' are located inside the domain.")
    l1, l2, l3 = domain[0][1] - domain[0][0], domain[1][1] - domain[1][0], domain[2][1] - domain[2][0]
    lengths = np.array([l1, l2, l3])
    volume = l1 * l2 * l3
    if not np.allclose(np.sum(target_volumes), volume):
        raise ValueError("The sum of 'target_volumes' is not equal to the volume of the domain.")
    elif points_init.shape != (n, 3):
        raise ValueError("The length of 'target_volumes' and the number of rows in 'points_init' are not equal.")
    if gradient_max_iterations is None:
        if solver == "DN":
            gradient_max_iterations = 100
        else:
            gradient_max_iterations = weights_init.shape[0] * 10

    weights = weights_init
    points = points_init
    previous_points = np.copy(points_init)
    # max_side = np.max(lengths)

    num_iterations = 0
    # stopping_criterion = tol * tol * volume * max_side * max_side
    # residual = 10 * stopping_criterion
    residual = 10 * tol
    gradient_convergence = True

    while residual > tol and num_iterations < max_iterations:
        if solver == "DN":
            weights, gradient_convergence = gm.dn_solver(weights, points, target_volumes, domain, periodic, permute,
                                                         gradient_max_iterations, gradient_tol, disp)
        elif solver == "BB":
            weights, gradient_convergence = gm.bb_solver(weights, points, target_volumes, domain, periodic, permute,
                                                         gradient_max_iterations, gradient_tol, disp)
        elif solver == "MM":
            weights, gradient_convergence = gm.mm_solver(weights, points, target_volumes, domain, periodic, permute,
                                                         gradient_max_iterations, gradient_tol, disp)
        else:
            print("Invalid solver provided, defaulting to DN.")
            compute_centroidal_laguerre3d(target_volumes,  points_init, weights_init, domain, periodic, permute,
                                          max_iterations, gradient_max_iterations, tol, gradient_tol, disp,
                                          fix_num_iterations, "DN", respect_bounds)

        centroids = voroplusplus.get_3d_centroids(weights, points, domain, periodic)

        if not fix_num_iterations:
            if periodic:
                # Calculate periodic distance
                min_dists = np.min(np.dstack(((points - centroids) % lengths, (centroids - points) % lengths)), axis=2)
                residual = np.mean(np.linalg.norm(min_dists, axis=1) / np.power(target_volumes, 1./3))
                # residual = np.sum(target_volumes * np.sum(np.square(min_dists), axis=1))
                # (np.linalg.norm(min_dists, axis=1) euclidean distances centroids and points
            else:
                # Calculate euclidean distance
                residual = np.mean(np.linalg.norm(points - centroids, axis=1) / np.power(target_volumes, 1. / 3))
                # residual = np.sum(target_volumes * np.sum(np.square(points - centroids), axis=1))

        previous_points = points
        points = centroids
        num_iterations += 1

        if not gradient_convergence:
            break

    if periodic:
        previous_points[:, 0] = np.remainder(previous_points[:, 0] - domain[0][0], l1) + domain[0][0]
        previous_points[:, 1] = np.remainder(previous_points[:, 1] - domain[1][0], l2) + domain[1][0]
        previous_points[:, 2] = np.remainder(previous_points[:, 2] - domain[2][0], l3) + domain[2][0]

    if (residual > tol and not fix_num_iterations) or not gradient_convergence:
        warnings.warn("Algorithm did not converge in max_iterations iterations.")
        return Laguerre3D(previous_points, weights, domain, periodic), False
    elif disp:
        print("Algorithm number of iterations: ", num_iterations)

    return Laguerre3D(previous_points, weights, domain, periodic), True
