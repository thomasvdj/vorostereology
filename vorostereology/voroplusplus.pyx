# cython: language_level=3
# distutils: language = c++
#
# Edited by Thomas van der Jagt
#
# voroplusplus.pyx : pyvoro cython interface to voro++
#
# this file provides a python interface for performing 3D voronoi tesselations
# using voro++.
#
# this extension to voro++ is released under the original modified BSD license
# and constitutes an Extension to the original project.
#
# Copyright (c) Joe Jordan 2012
# contact: <joe.jordan@imperial.ac.uk> or <tehwalrus@h2j9k.org>
#

from __future__ import division

from libcpp.vector cimport vector
from libcpp cimport bool
from cython.operator cimport dereference as deref
from libcpp.pair cimport pair
from math import sqrt
import numpy as np
import warnings
from . import function_definitions
cimport cython


cdef extern from "vpp.h":
    void* container_poly_create(double ax_, double bx_, double ay_, double by_,
        double az_, double bz_, int nx_, int ny_, int nz_, int px_, int py_, int pz_)
    void put_particle(void* container_poly_, int i_, double x_, double y_, double z_, double r_)
    void put_particles(void* container_poly_, int n_, double* x_, double* y_, double* z_, double* r_)
    pair[vector[double], vector[double]] compute_volumes_centroids(const vector[double] &weights, 
      double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
      const vector[double] &x_, const vector[double] &y_, const vector[double] &z_, int num_cpus)
    vector[double] compute_volumes(const vector[double] &weights, 
      double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
      const vector[double] &x_, const vector[double] &y_, const vector[double] &z_, int num_cpus)
    void** compute_voronoi_tesselation(void* container_poly_, int n_)
    pair[void**, vector[int]] compute_voronoi_tessellation_par(void* container_poly_, int n_)
    pair[void**, vector[bool]] compute_voronoi_tessellation_bounded(void* container_poly_, int n_, vector[bool] cells_to_compute, 
        vector[bool] x_outside, vector[bool] y_outside, vector[bool] z_outside)
    pair[void**, vector[int]] compute_section(void* container_poly_, int n_, double a, double b, 
        double c, double x_0, double y_0, double z_0)
    pair[void**, vector[int]] compute_section_bounded(void* container_poly_, int n_, double a, double b, 
        double c, double x_0, double y_0, double z_0, vector[bool] cells_to_compute, vector[bool] x_outside, vector[bool] y_outside, vector[bool] z_outside)
    vector[vector[bool]] compute_num_fragments(void* container_poly_, int n_)
    double cell_get_volume(void* cell_)
    double cell_get_transportcost(void* cell_)
    vector[double] cell_get_centroid(void* cell_)
    vector[double] cell_get_areas(void* cell_)
    vector[double] cell_get_vertex_positions(void* cell_, double x_, double y_, double z_)
    void** cell_get_vertex_adjacency(void* cell_)
    void** cell_get_faces(void* cell_)
    void dispose_container(void* container_poly_)
    void dispose_all(void* container_poly_, void** vorocells, int n_)
    void dispose_cells(void** vorocells, int n_)


cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr)
    void* malloc(size_t size)

import sys
import math

class VoronoiPlusPlusError(Exception):
    pass


def get_constructor(obj):
    """
    Input arg format:
    obj = the object we want to get the constructor for
    """
    typ = type(obj)

    # Test if we have a numpy array
    if hasattr(typ, '__module__'):
        if typ.__module__ == 'numpy':
            numpy = sys.modules['numpy']
            typ = numpy.array
        
    return typ

def same_rows(a, b, tol=8):
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return np.all(np.any(rows_close, axis=-1), axis=-1) and np.all(np.any(rows_close, axis=0), axis=0)


def transform_2d(cross_cells, coeffs, offset):
    norm = np.linalg.norm(coeffs)
    norm_xy = np.linalg.norm(coeffs[0:2])

    cos_theta = coeffs[2] / norm
    sin_theta = norm_xy / norm
    u1 = coeffs[1] / norm_xy
    u2 = -1 * coeffs[0] / norm_xy

    rot_matrix = np.zeros((3, 3))
    rot_matrix[0, 0] = cos_theta + u1 * u1 * (1 - cos_theta)
    rot_matrix[0, 1] = u1 * u2 * (1 - cos_theta)
    rot_matrix[0, 2] = u2 * sin_theta
    rot_matrix[1, 0] = u1 * u2 * (1 - cos_theta)
    rot_matrix[1, 1] = cos_theta + u2 * u2 * (1 - cos_theta)
    rot_matrix[1, 2] = -1 * u1 * sin_theta
    rot_matrix[2, 0] = -1 * u2 * sin_theta
    rot_matrix[2, 1] = u1 * sin_theta
    rot_matrix[2, 2] = cos_theta

    return [np.dot(rot_matrix, (np.array(cell) - offset).T).T[:, 0:2] for cell in cross_cells]


def BB_solver(weights_init, points, target_volumes, domain, block_size, periodicity, permute=False,
              max_iter=10000, tol=0.01, disp=False, num_cpus=None):
    """
    For now this function is not fully documented, most arguments are inherited from "compute_centroidal_laguerre".
        This function finds the right weights to obtain a Laguerre diagram with a desired volume distribution by
        minimizing a convex function using the Barzilai-Borwein method.
    :param weights_init:
    :param points:
    :param target_volumes:
    :param domain:
    :param block_size:
    :param periodicity:
    :param permute:
    :param max_iter:
    :param tol:
    :param disp:
    :return:
    """
    L1 = domain[0][1] - domain[0][0]
    L2 = domain[1][1] - domain[1][0]
    L3 = domain[2][1] - domain[2][0]
    weights = weights_init
    n = weights.shape[0]
    diameter = np.sqrt(L1*L1 + L2*L2 + L3*L3)
    # we cannot take the square root of negative weights, but adding a constant to all weights does not change
    # the Laguerre diagram.
    min_weight = np.min(weights)
    if min_weight <= 0:
        weights = weights - min_weight + diameter
    volumes = np.array(compute_volumes(weights, 
            <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
            <int>periodicity[0], <int>periodicity[1], <int>periodicity[2], n, points[:, 0], points[:, 1], points[:, 2], <int>num_cpus))
    previous_volumes = volumes

    # Optional, permutation step might speed up the algorithm in some cases
    if permute:
        target_volumes_ = function_definitions.permute_targets(volumes, target_volumes, volumes.shape[0])
    else:
        target_volumes_ = target_volumes

    residual = np.max(np.abs(volumes - target_volumes_)) # np.linalg.norm(volumes - target_volumes_)
    stopping_criterion = np.min(target_volumes) * tol
    previous_weights = weights
    minimization_num_iterations = 0
    alpha = 0.1 / np.cbrt(L1*L2*L3)

    while residual > stopping_criterion and minimization_num_iterations < max_iter:
        # Update weights and calculate volumes of the tessellation with these new weights
        weights = previous_weights - alpha * (volumes - target_volumes_)
        # we cannot take the square root of negative weights, but adding a constant to all weights does not change
        # the Laguerre diagram.
        min_weight = np.min(weights)
        if min_weight <= 0:
            weights = weights - min_weight + diameter
            previous_weights = previous_weights - min_weight + diameter
        volumes = np.array(compute_volumes(weights, 
            <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
            <int>periodicity[0], <int>periodicity[1], <int>periodicity[2], n, points[:, 0], points[:, 1], points[:, 2], <int>num_cpus))

        # Optional, permutation step might speed up the algorithm in some cases
        if permute:
            target_volumes_ = function_definitions.permute_targets(volumes, target_volumes_, n)

        # Update residual for the next iteration
        residual = np.max(np.abs(volumes - target_volumes_)) #np.linalg.norm(volumes - target_volumes_)

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


def MM_solver(weights_init, points, target_volumes, domain, block_size, periodicity, permute=False,
              max_iter=10000, tol=0.01, disp=False, num_cpus=None):
    """
    For now this function is not fully documented, most arguments are inherited from "compute_centroidal_laguerre".
        This function finds the right weights to obtain a Laguerre diagram with a desired volume distribution by
        minimizing a convex function using the Malitsky-Mishchenko method.
    :param weights_init:
    :param points:
    :param target_volumes:
    :param domain:
    :param block_size:
    :param periodicity:
    :param permute:
    :param max_iter:
    :param tol:
    :param disp:
    :return:
    """
    L1 = domain[0][1] - domain[0][0]
    L2 = domain[1][1] - domain[1][0]
    L3 = domain[2][1] - domain[2][0]
    weights = weights_init
    n = weights.shape[0]
    diameter = np.sqrt(L1*L1 + L2*L2 + L3*L3)
    # we cannot take the square root of negative weights, but adding a constant to all weights does not change
    # the Laguerre diagram.
    min_weight = np.min(weights)
    if min_weight <= 0:
        weights = weights - min_weight  + diameter
    volumes = np.array(compute_volumes(weights, 
            <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
            <int>periodicity[0], <int>periodicity[1], <int>periodicity[2], n, points[:, 0], points[:, 1], points[:, 2], <int>num_cpus))
    previous_volumes = volumes

    # Optional, permutation step might speed up the algorithm in some cases
    if permute:
        target_volumes_ = function_definitions.permute_targets(volumes, target_volumes, volumes.shape[0])
    else:
        target_volumes_ = target_volumes

    residual = np.max(np.abs(volumes - target_volumes_)) # np.linalg.norm(volumes - target_volumes_)
    stopping_criterion = np.min(target_volumes) * tol
    previous_weights = weights
    minimization_num_iterations = 0
    theta = 0
    alpha = 0.1 / np.cbrt(L1*L2*L3)

    while residual > stopping_criterion and minimization_num_iterations < max_iter:
        # Update weights and calculate volumes of the tessellation with these new weights
        weights = previous_weights - alpha * (volumes - target_volumes_)
        # we cannot take the square root of negative weights, but adding a constant to all weights does not change
        # the Laguerre diagram.
        min_weight = np.min(weights)
        if min_weight <= 0:
            weights = weights - min_weight  + diameter
            previous_weights = previous_weights - min_weight + diameter
        volumes = np.array(compute_volumes(weights, 
            <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
            <int>periodicity[0], <int>periodicity[1], <int>periodicity[2], n, points[:, 0], points[:, 1], points[:, 2], <int>num_cpus))

        # Optional, permutation step might speed up the algorithm in some cases
        if permute:
            target_volumes_ = function_definitions.permute_targets(volumes, target_volumes_, n)

        # Update residual for the next iteration
        residual = np.max(np.abs(volumes - target_volumes_)) # np.linalg.norm(volumes - target_volumes_)

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



def compute_centroidal_laguerre(weights_init, points_init, target_volumes, domain, periodicity, permute, 
    max_iterations, gradient_max_iterations, tol, gradient_tol, disp, fix_num_iterations, solver, respect_bounds, num_cpus):
    
    n = weights_init.shape[0]
    cdef pair[vector[double], vector[double]] volumes_centroids
    weights = weights_init
    points = points_init
    previous_points = points_init
    L1 = domain[0][1] - domain[0][0]
    L2 = domain[1][1] - domain[1][0]
    L3 = domain[2][1] - domain[2][0]
    Lengths = np.array([L1, L2, L3])
    volume = L1*L2*L3
    block_size = np.max(Lengths)*0.2
    max_side = np.max(Lengths)

    volumes = np.zeros(n)

    num_iterations = 0
    stopping_criterion = tol * tol * volume * max_side * max_side
    residual = 10 * stopping_criterion
    gradient_convergence = True

    while residual > stopping_criterion and num_iterations < max_iterations:
        if solver == "BB":
            weights, gradient_convergence = BB_solver(weights, points, target_volumes, domain, block_size, periodicity, 
                permute=permute, max_iter=gradient_max_iterations, tol=gradient_tol, disp=disp, num_cpus=num_cpus)
        elif solver == "MM":
            weights, gradient_convergence = MM_solver(weights, points, target_volumes, domain, block_size, periodicity, 
                permute=permute, max_iter=gradient_max_iterations, tol=gradient_tol, disp=disp, num_cpus=num_cpus)
        else:
            print("Invalid solver provided, defaulting to MM.")
            compute_centroidal_laguerre(weights_init, points_init, target_volumes, domain, periodicity, permute, 
                max_iterations, gradient_max_iterations, tol, gradient_tol, disp, fix_num_iterations, "MM", respect_bounds, num_cpus)
        
        volumes_centroids = compute_volumes_centroids(weights, 
            <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
            <int>periodicity[0], <int>periodicity[1], <int>periodicity[2], n, points[:, 0], points[:, 1], points[:, 2], <int>num_cpus)
        volumes = np.array(volumes_centroids.first)
        centroids = np.reshape(volumes_centroids.second, (n, 3), order='F')

        if not fix_num_iterations:
            residual = np.sum(target_volumes * np.sum(np.square(points - centroids), axis=1))

        previous_points = points
        points = centroids
        num_iterations += 1
        
        if not gradient_convergence:
            break

    if permute:
        target_volumes_ = np.array(function_definitions.permute_targets(volumes, target_volumes, n))
    else:
        target_volumes_ = target_volumes

    if periodicity[0]:
        previous_points[:, 0] = np.remainder(previous_points[:, 0], L1)
    if periodicity[1]:
        previous_points[:, 1] = np.remainder(previous_points[:, 1], L2)
    if periodicity[2]:
        previous_points[:, 2] = np.remainder(previous_points[:, 2], L3)

    if (residual > stopping_criterion and not fix_num_iterations) or not gradient_convergence:
        warnings.warn("Algorithm did not converge in max_iterations iterations.")
        return {"volumes": volumes, "weights": weights, "points": previous_points, 
            "target_volumes": target_volumes_, "convergence": False, "centroids": points}
    elif disp:
        print("Algorithm number of iterations: ", num_iterations)

    return {"volumes": volumes, "weights": weights, "points": previous_points, 
        "target_volumes": target_volumes_, "convergence": True, "centroids": points}


def compute_centroidal_laguerre2d(weights_init, points_init, target_areas, domain, periodicity, permute, 
    max_iterations, gradient_max_iterations, tol, gradient_tol, disp, fix_num_iterations, solver, respect_bounds,
    num_cpus):
    
    n = weights_init.shape[0]
    cdef pair[vector[double], vector[double]] volumes_centroids
    weights = weights_init
    points = np.hstack([points_init, np.ones((weights.shape[0], 1))*0.5])
    previous_points = np.hstack([points_init, np.ones((weights.shape[0], 1))*0.5])
    L1 = domain[0][1] - domain[0][0]
    L2 = domain[1][1] - domain[1][0]
    L3 = 1
    domain_ = domain + [[0, 1]]
    Lengths = np.array([L1, L2, L3])
    volume = L1*L2*L3
    block_size = np.max(Lengths)*0.2
    max_side = np.max(Lengths)
    periodicity_ = periodicity + [False]

    areas = np.zeros(n)

    num_iterations = 0
    stopping_criterion = tol * tol * volume * max_side * max_side
    residual = 10 * stopping_criterion
    gradient_convergence = True

    while residual > stopping_criterion and num_iterations < max_iterations:
        if solver == "BB":
            weights, gradient_convergence = BB_solver(weights, points, target_areas, domain_, block_size, periodicity_, 
                permute=permute, max_iter=gradient_max_iterations, tol=gradient_tol, disp=disp, num_cpus=num_cpus)
        elif solver == "MM":
            weights, gradient_convergence = MM_solver(weights, points, target_areas, domain_, block_size, periodicity_, 
                permute=permute, max_iter=gradient_max_iterations, tol=gradient_tol, disp=disp, num_cpus=num_cpus)
        else:
            print("Invalid solver provided, defaulting to MM.")
            compute_centroidal_laguerre(weights_init, points_init, target_areas, domain, periodicity, permute, 
                max_iterations, gradient_max_iterations, tol, gradient_tol, disp, fix_num_iterations, "MM", respect_bounds, num_cpus)

        volumes_centroids = compute_volumes_centroids(weights, 
            <double>domain_[0][0], <double>domain_[0][1], <double>domain_[1][0], <double>domain_[1][1], <double>domain_[2][0], <double>domain_[2][1], 
                <int>periodicity_[0], <int>periodicity_[1], <int>periodicity_[2], n, points[:, 0], points[:, 1], points[:, 2], <int>num_cpus)
        areas = np.array(volumes_centroids.first)
        centroids = np.reshape(volumes_centroids.second, (n, 3), order='F')
        centroids[:, 2] = 0.5

        if not fix_num_iterations:
            residual = np.sum(target_areas * np.sum(np.square(points - centroids), axis=1))

        points[:, 2] = 0.5
        previous_points = points
        points = centroids
        num_iterations += 1

    if permute:
        target_areas_ = np.array(function_definitions.permute_targets(areas, target_areas, n))
    else:
        target_areas_ = target_areas

    if periodicity[0]:
        previous_points[:, 0] = np.remainder(previous_points[:, 0], L1)
    if periodicity[1]:
        previous_points[:, 1] = np.remainder(previous_points[:, 1], L2)

    if (residual > stopping_criterion and not fix_num_iterations) or not gradient_convergence:
        warnings.warn("Algorithm did not converge in max_iterations iterations.")
        return {"areas": areas, "weights": weights, "points": previous_points[:, 0:2], 
            "target_areas": target_areas_, "convergence": False, "centroids": points[:, 0:2]}
    elif disp:
        print("Algorithm number of iterations: ", num_iterations)

    return {"areas": areas, "weights": weights, "points": previous_points[:, 0:2], 
        "target_areas": target_areas_, "convergence": True, "centroids": points[:, 0:2]}


@cython.wraparound(False)
def compute_voronoi(points, limits, dispersion, radii, periodic):
  """
Input arg formats:
  points = list of 3-vectors (lists or compatible class instances) of doubles,
    being the coordinates of the points to voronoi-tesselate.
  limits = 3-list of 2-lists, specifying the start and end sizes of the box the
    points are in.
  dispersion = max distance between two points that might be adjacent (sets
    voro++ block sizes.)
  radii (optional) = list of python floats as the sphere radii of the points,
    for radical (weighted) tessellation.
  periodic (optional) = 3-list of bools indicating x, y and z periodicity of 
    the system box.
  
Output format is a list of cells as follows:
  [ # list in same order as original points.
    {
      'volume' : 1.0,
      'vertices' : [[1.0, 2.0, 3.0], ...], # positions of vertices
      'faces' : [
        {
          'vertices' : [7,4,13, ...], # vertex ids in loop order
          'adjacent_cell' : 34 # *cell* id, negative if a wall
        }, ...]
      'original' : point[index] # the original instance from args
    },
    ... 
  ]
  
  NOTE: The class from items in input points list is reused for all 3-vector
  outputs. It must have a constructor which accepts a list of 3 python floats
  (python's list type does satisfy this requirement.)
  """
  cdef Py_ssize_t n = len(points), i, j
  cdef double *xs
  cdef double *ys
  cdef double *zs
  cdef double *rs
  cdef void** voronoi_cells
  
  vector_class = get_constructor(points[0])

  periodic = [1 if p else 0 for p in periodic]
  
  # we must make sure we have at least one block, or voro++ will segfault when
  # we look for cells.
  
  blocks = [
    max([1, int(math.floor((limits[0][1] - limits[0][0]) / dispersion))]),
    max([1, int(math.floor((limits[1][1] - limits[1][0]) / dispersion))]),
    max([1, int(math.floor((limits[2][1] - limits[2][0]) / dispersion))])
  ]
  
  # if no radii provided, we still run the radical routine, but with all the same small radius.
  if len(radii) != len(points):
    radii = [dispersion / 10.] * len(points)
  
  # build the container object
  cdef void* container = container_poly_create(
    <double>limits[0][0],
    <double>limits[0][1],
    <double>limits[1][0],
    <double>limits[1][1],
    <double>limits[2][0],
    <double>limits[2][1],
    <int>blocks[0],
    <int>blocks[1],
    <int>blocks[2],
    <int>periodic[0],
    <int>periodic[1],
    <int>periodic[2]
  )
  
  xs = <double*>malloc(sizeof(double) * n)
  ys = <double*>malloc(sizeof(double) * n)
  zs = <double*>malloc(sizeof(double) * n)
  rs = <double*>malloc(sizeof(double) * n)
  
  # initialise particle positions:
  for i in range(n):
    xs[i] = <double>points[i][0]
    ys[i] = <double>points[i][1]
    zs[i] = <double>points[i][2]
    rs[i] = <double>radii[i]
    
  # and add them to the container:
  put_particles(container, n, xs, ys, zs, rs)
    
  # now compute the tessellation:
  result = compute_voronoi_tessellation_par(container, n)
  voronoi_cells = result.first
  empty_indices = result.second
    
  # extract the Voronoi cells into python objects:
  py_cells = [{'original':p} for p in points]
  cdef vector[double] vertex_positions
  cdef void** lists = NULL
  cdef vector[int]* vptr = NULL
  for i in range(n):
    py_cells[i]['volume'] = float(cell_get_volume(voronoi_cells[i]))
    vertex_positions = cell_get_vertex_positions(voronoi_cells[i], xs[i], ys[i], zs[i])
    cell_vertices = []
    for j in range(<Py_ssize_t>(vertex_positions.size() // 3)):
      cell_vertices.append(vector_class([
        float(vertex_positions[3 * j]),
        float(vertex_positions[3 * j + 1]),
        float(vertex_positions[3 * j + 2])
      ]))
    py_cells[i]['vertices'] = cell_vertices
    
    lists = cell_get_faces(voronoi_cells[i])
    faces = []
    j = 0
    while lists[j] != NULL:
      face_vertices = []
      vptr = <vector[int]*>lists[j]
      for k in range(vptr.size() - 1):
        face_vertices.append(int(deref(vptr)[k]))
      faces.append({
        'adjacent_cell' : int(deref(vptr)[vptr.size() - 1]),
        'vertices' : face_vertices
      })
      del vptr
      j += 1
    free(lists)
    py_cells[i]['faces'] = faces
  
  for idx in empty_indices:
    py_cells[idx] = {}
    
  # finally, tidy up.
  dispose_all(container, voronoi_cells, n)
  free(xs)
  free(ys)
  free(zs)
  free(rs)
  return py_cells


@cython.wraparound(False)
def compute_voronoi_bounded(points, limits, dispersion, cells_to_compute, x_outside, y_outside, z_outside, radii, periodic):

  cdef Py_ssize_t n = len(points), i, j
  cdef double *xs
  cdef double *ys
  cdef double *zs
  cdef double *rs
  cdef void** voronoi_cells
  
  vector_class = get_constructor(points[0])

  periodic = [1 if p else 0 for p in periodic]
  
  # we must make sure we have at least one block, or voro++ will segfault when
  # we look for cells.
  
  blocks = [
    max([1, int(math.floor((limits[0][1] - limits[0][0]) / dispersion))]),
    max([1, int(math.floor((limits[1][1] - limits[1][0]) / dispersion))]),
    max([1, int(math.floor((limits[2][1] - limits[2][0]) / dispersion))])
  ]
  
  # if no radii provided, we still run the radical routine, but with all the same small radius.
  if len(radii) != len(points):
    radii = [dispersion / 10.] * len(points)
  
  # build the container object
  cdef void* container = container_poly_create(
    <double>limits[0][0],
    <double>limits[0][1],
    <double>limits[1][0],
    <double>limits[1][1],
    <double>limits[2][0],
    <double>limits[2][1],
    <int>blocks[0],
    <int>blocks[1],
    <int>blocks[2],
    <int>periodic[0],
    <int>periodic[1],
    <int>periodic[2]
  )
  
  xs = <double*>malloc(sizeof(double) * n)
  ys = <double*>malloc(sizeof(double) * n)
  zs = <double*>malloc(sizeof(double) * n)
  rs = <double*>malloc(sizeof(double) * n)
  
  # initialise particle positions:
  for i in range(n):
    xs[i] = <double>points[i][0]
    ys[i] = <double>points[i][1]
    zs[i] = <double>points[i][2]
    rs[i] = <double>radii[i]
    
  # and add them to the container:
  put_particles(container, n, xs, ys, zs, rs)
    
  # now compute the tessellation:
  result = compute_voronoi_tessellation_bounded(container, n, cells_to_compute, x_outside, y_outside, z_outside)
  voronoi_cells = result.first
  computed_indices = result.second
    
  # extract the Voronoi cells into python objects:
  py_cells = [{'original':p} for p in points]
  cdef vector[double] vertex_positions
  cdef void** lists = NULL
  cdef vector[int]* vptr = NULL
  for i in range(n):
    if computed_indices[i]:
      py_cells[i]['volume'] = float(cell_get_volume(voronoi_cells[i]))
      vertex_positions = cell_get_vertex_positions(voronoi_cells[i], xs[i], ys[i], zs[i])
      cell_vertices = []
      for j in range(<Py_ssize_t>(vertex_positions.size() // 3)):
        cell_vertices.append(vector_class([
          float(vertex_positions[3 * j]),
          float(vertex_positions[3 * j + 1]),
          float(vertex_positions[3 * j + 2])
        ]))
      py_cells[i]['vertices'] = cell_vertices
      
      lists = cell_get_faces(voronoi_cells[i])
      faces = []
      j = 0
      while lists[j] != NULL:
        face_vertices = []
        vptr = <vector[int]*>lists[j]
        for k in range(vptr.size() - 1):
          face_vertices.append(int(deref(vptr)[k]))
        faces.append({
          'adjacent_cell' : int(deref(vptr)[vptr.size() - 1]),
          'vertices' : face_vertices
        })
        del vptr
        j += 1
      free(lists)
      py_cells[i]['faces'] = faces
    else:
      py_cells[i] = {}

  # finally, tidy up.
  dispose_all(container, voronoi_cells, n)
  free(xs)
  free(ys)
  free(zs)
  free(rs)
  return py_cells


@cython.wraparound(False) 
def compute_voronoi_section(coeffs, offset, points, limits, dispersion, radii, periodic):

  cdef Py_ssize_t n = len(points), i, j
  cdef double *xs
  cdef double *ys
  cdef double *zs
  cdef double *rs
  cdef void** voronoi_cells
  
  vector_class = get_constructor(points[0])

  periodic = [1 if p else 0 for p in periodic]
  
  # we must make sure we have at least one block, or voro++ will segfault when
  # we look for cells.
  
  blocks = [
    max([1, int(math.floor((limits[0][1] - limits[0][0]) / dispersion))]),
    max([1, int(math.floor((limits[1][1] - limits[1][0]) / dispersion))]),
    max([1, int(math.floor((limits[2][1] - limits[2][0]) / dispersion))])
  ]
  
  # if no radii provided, we still run the radical routine, but with all the same small radius.
  if len(radii) != len(points):
    radii = [dispersion / 10.] * len(points)
  
  # build the container object
  cdef void* container = container_poly_create(
    <double>limits[0][0],
    <double>limits[0][1],
    <double>limits[1][0],
    <double>limits[1][1],
    <double>limits[2][0],
    <double>limits[2][1],
    <int>blocks[0],
    <int>blocks[1],
    <int>blocks[2],
    <int>periodic[0],
    <int>periodic[1],
    <int>periodic[2]
  )
  
  xs = <double*>malloc(sizeof(double) * n)
  ys = <double*>malloc(sizeof(double) * n)
  zs = <double*>malloc(sizeof(double) * n)
  rs = <double*>malloc(sizeof(double) * n)
  
  # initialise particle positions:
  for i in range(n):
    xs[i] = <double>points[i][0]
    ys[i] = <double>points[i][1]
    zs[i] = <double>points[i][2]
    rs[i] = <double>radii[i]
    
  # and add them to the container:
  put_particles(container, n, xs, ys, zs, rs)
    
  # now compute the tessellation:
  result = compute_section(container, n, <double>coeffs[0], <double>coeffs[1], <double>coeffs[2], 
    <double>offset[0], <double>offset[1], <double>offset[2])
  voronoi_cells = result.first
  section_indices = result.second
      
  # extract the Voronoi cells into python objects:
  cdef vector[double] vertex_positions
  cdef void** lists = NULL
  cdef vector[int]* vptr = NULL
  vertices_3d = []
  section_areas = np.zeros(len(section_indices))

  for i in range(len(section_indices)):
    vertex_positions = cell_get_vertex_positions(voronoi_cells[section_indices[i]], xs[section_indices[i]], ys[section_indices[i]], zs[section_indices[i]])
    cell_vertices = []
    for j in range(<Py_ssize_t>(vertex_positions.size() // 3)):
      cell_vertices.append(vector_class([
        float(vertex_positions[3 * j]),
        float(vertex_positions[3 * j + 1]),
        float(vertex_positions[3 * j + 2])
      ]))
    
    lists = cell_get_faces(voronoi_cells[section_indices[i]])
    
    faces_areas = cell_get_areas(voronoi_cells[section_indices[i]])
    j=0
    while lists[j] != NULL:
      face_vertices = []
      vptr = <vector[int]*>lists[j]
      for k in range(vptr.size() - 1):
        face_vertices.append(int(deref(vptr)[k]))
      if int(deref(vptr)[vptr.size() - 1]) == n+1:
        section_areas[i] = faces_areas[j]
        vertices_3d.append(np.array(cell_vertices)[np.array(face_vertices, dtype=np.intc)])
      del vptr
      j += 1
      
    free(lists)
  
  duplicates = set()
  idx_sort = np.argsort(section_areas)
  sorted_areas = section_areas[idx_sort]
  sorted_indices = idx_sort[idx_sort]

  vals, idx_start, count = np.unique(sorted_areas, return_counts=True, return_index=True)

  for i, first_occurrence in enumerate(idx_start):
    if count[i] > 1:
      for p in range(first_occurrence, first_occurrence + count[i]):
        for q in range(p+1, first_occurrence + count[i]):
          if vertices_3d[sorted_indices[p]].shape == vertices_3d[sorted_indices[q]].shape:
            if same_rows(vertices_3d[sorted_indices[p]], vertices_3d[sorted_indices[q]]):
              duplicates.add(sorted_indices[q])
  
  vertices_3d = [vertices_3d[j] for j in range(len(vertices_3d)) if j not in duplicates]
  section_indices_ = np.delete(section_indices, list(duplicates))
  section_areas_ = np.delete(section_areas, list(duplicates))

  if coeffs[0] == 0 and coeffs[1] == 0:
    cross_section_2d = [np.array(cell)[:, 0:2] for cell in vertices_3d]
  else:
    cross_section_2d = transform_2d(vertices_3d, coeffs, offset)
    
  # finally, tidy up.
  dispose_all(container, voronoi_cells, n)
  free(xs)
  free(ys)
  free(zs)
  free(rs)
  return {"3d": vertices_3d, "original_indices": section_indices_, "areas": section_areas_, "2d": cross_section_2d, "polygons_indices": section_indices_}


@cython.wraparound(False)
def compute_voronoi_section_bounded(coeffs, offset, points, limits, dispersion, cells_to_compute, x_outside, y_outside, z_outside, radii, periodic):

  cdef Py_ssize_t n = len(points), i, j
  cdef double *xs
  cdef double *ys
  cdef double *zs
  cdef double *rs
  cdef void** voronoi_cells
  
  vector_class = get_constructor(points[0])

  periodic = [1 if p else 0 for p in periodic]
  
  # we must make sure we have at least one block, or voro++ will segfault when
  # we look for cells.
  
  blocks = [
    max([1, int(math.floor((limits[0][1] - limits[0][0]) / dispersion))]),
    max([1, int(math.floor((limits[1][1] - limits[1][0]) / dispersion))]),
    max([1, int(math.floor((limits[2][1] - limits[2][0]) / dispersion))])
  ]
  
  # if no radii provided, we still run the radical routine, but with all the same small radius.
  if len(radii) != len(points):
    radii = [dispersion / 10.] * len(points)
  
  # build the container object
  cdef void* container = container_poly_create(
    <double>limits[0][0],
    <double>limits[0][1],
    <double>limits[1][0],
    <double>limits[1][1],
    <double>limits[2][0],
    <double>limits[2][1],
    <int>blocks[0],
    <int>blocks[1],
    <int>blocks[2],
    <int>periodic[0],
    <int>periodic[1],
    <int>periodic[2]
  )
  
  xs = <double*>malloc(sizeof(double) * n)
  ys = <double*>malloc(sizeof(double) * n)
  zs = <double*>malloc(sizeof(double) * n)
  rs = <double*>malloc(sizeof(double) * n)
  
  # initialise particle positions:
  for i in range(n):
    xs[i] = <double>points[i][0]
    ys[i] = <double>points[i][1]
    zs[i] = <double>points[i][2]
    rs[i] = <double>radii[i]
    
  # and add them to the container:
  put_particles(container, n, xs, ys, zs, rs)
    
  # now compute the tessellation:
  result = compute_section_bounded(container, n, <double>coeffs[0], <double>coeffs[1], <double>coeffs[2], 
    <double>offset[0], <double>offset[1], <double>offset[2], cells_to_compute, x_outside, y_outside, z_outside)
  voronoi_cells = result.first
  section_indices = result.second
      
  # extract the Voronoi cells into python objects:
  cdef vector[double] vertex_positions
  cdef void** lists = NULL
  cdef vector[int]* vptr = NULL
  vertices_3d = []
  section_areas = [] # np.zeros(len(section_indices))

  for i in range(len(section_indices)):
    vertex_positions = cell_get_vertex_positions(voronoi_cells[section_indices[i]], xs[section_indices[i]], ys[section_indices[i]], zs[section_indices[i]])
    cell_vertices = []
    for j in range(<Py_ssize_t>(vertex_positions.size() // 3)):
      cell_vertices.append(vector_class([
        float(vertex_positions[3 * j]),
        float(vertex_positions[3 * j + 1]),
        float(vertex_positions[3 * j + 2])
      ]))
    
    lists = cell_get_faces(voronoi_cells[section_indices[i]])
    
    faces_areas = cell_get_areas(voronoi_cells[section_indices[i]])
    j=0
    while lists[j] != NULL:
      face_vertices = []
      vptr = <vector[int]*>lists[j]
      for k in range(vptr.size() - 1):
        face_vertices.append(int(deref(vptr)[k]))
      if int(deref(vptr)[vptr.size() - 1]) == n+1:
        section_areas.append(faces_areas[j])
        vertices_3d.append(np.array(cell_vertices)[np.array(face_vertices, dtype=np.intc)])
      del vptr
      j += 1
      
    free(lists)
  
  section_areas = np.array(section_areas)
  duplicates = set()

  if len(section_areas) > 0:
    idx_sort = np.argsort(section_areas)
    sorted_areas = section_areas[idx_sort]
    sorted_indices = idx_sort[idx_sort]

    vals, idx_start, count = np.unique(sorted_areas, return_counts=True, return_index=True)

    pairs = set()

    for i, first_occurrence in enumerate(idx_start):
      if count[i] > 1:
        for p in range(first_occurrence, first_occurrence + count[i]):
          for q in range(p+1, first_occurrence + count[i]):
            if ((q, p) not in pairs) and ((p, q) not in pairs):
              if vertices_3d[sorted_indices[p]].shape == vertices_3d[sorted_indices[q]].shape:
                if same_rows(vertices_3d[sorted_indices[p]], vertices_3d[sorted_indices[q]]):
                  duplicates.add(sorted_indices[q])
              pairs.add((p, q))
  
  vertices_3d = [vertices_3d[j] for j in range(len(vertices_3d)) if j not in duplicates]
  section_indices_ = np.delete(section_indices, list(duplicates))
  section_areas_ = np.delete(section_areas, list(duplicates))

  if coeffs[0] == 0 and coeffs[1] == 0:
    cross_section_2d = [np.array(cell)[:, 0:2] for cell in vertices_3d]
  else:
    cross_section_2d = transform_2d(vertices_3d, coeffs, offset)

  # finally, tidy up.
  dispose_all(container, voronoi_cells, n)
  free(xs)
  free(ys)
  free(zs)
  free(rs)
  return {"3d": vertices_3d, "original_indices": section_indices_, "areas": section_areas_, "2d": cross_section_2d}


@cython.wraparound(False)
def voronoi_num_fragments(points, limits, dispersion, radii, periodic):
  """
Input arg formats:
  points = list of 3-vectors (lists or compatible class instances) of doubles,
    being the coordinates of the points to voronoi-tesselate.
  limits = 3-list of 2-lists, specifying the start and end sizes of the box the
    points are in.
  dispersion = max distance between two points that might be adjacent (sets
    voro++ block sizes.)
  radii (optional) = list of python floats as the sphere radii of the points,
    for radical (weighted) tessellation.
  periodic (optional) = 3-list of bools indicating x, y and z periodicity of 
    the system box.
  
Output:
List of length n, each element is a list of length 6
  """

  cdef Py_ssize_t n = len(points), i, j
  cdef double *xs
  cdef double *ys
  cdef double *zs
  cdef double *rs
  cdef vector[vector[bool]] result
  
  vector_class = get_constructor(points[0])

  periodic = [1 if p else 0 for p in periodic]
  
  # we must make sure we have at least one block, or voro++ will segfault when
  # we look for cells.
  
  blocks = [
    max([1, int(math.floor((limits[0][1] - limits[0][0]) / dispersion))]),
    max([1, int(math.floor((limits[1][1] - limits[1][0]) / dispersion))]),
    max([1, int(math.floor((limits[2][1] - limits[2][0]) / dispersion))])
  ]
  
  # if no radii provided, we still run the radical routine, but with all the same small radius.
  if len(radii) != len(points):
    radii = [dispersion / 10.] * len(points)
  
  # build the container object
  cdef void* container = container_poly_create(
    <double>limits[0][0],
    <double>limits[0][1],
    <double>limits[1][0],
    <double>limits[1][1],
    <double>limits[2][0],
    <double>limits[2][1],
    <int>blocks[0],
    <int>blocks[1],
    <int>blocks[2],
    <int>periodic[0],
    <int>periodic[1],
    <int>periodic[2]
  )
  
  xs = <double*>malloc(sizeof(double) * n)
  ys = <double*>malloc(sizeof(double) * n)
  zs = <double*>malloc(sizeof(double) * n)
  rs = <double*>malloc(sizeof(double) * n)
  
  # initialise particle positions:
  for i in range(n):
    xs[i] = <double>points[i][0]
    ys[i] = <double>points[i][1]
    zs[i] = <double>points[i][2]
    rs[i] = <double>radii[i]
  
  # and add them to the container:
  put_particles(container, n, xs, ys, zs, rs)
  
  # now compute the tessellation:
  result = compute_num_fragments(container, n)

  dispose_container(container)
  free(xs)
  free(ys)
  free(zs)
  free(rs)
  return result
