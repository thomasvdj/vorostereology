# cython: language_level=3
# distutils: language = c++
#
# voroplusplus.pyx : cython interface to voro++

from libcpp.vector cimport vector
from libcpp cimport bool as bool_t
from math import sqrt
import numpy as np
import warnings
from . import function_definitions
cimport cython
from scipy.sparse import csr_matrix


cdef extern from "vpp.h":
    void compute_centroids(const vector[double] &weights, 
      double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
      const vector[double] &x_, const vector[double] &y_, const vector[double] &z_, int nx_, int ny_, int nz_, double centroids[])
    void compute_volumes(const vector[double] &weights, 
      double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
      const vector[double] &x_, const vector[double] &y_, const vector[double] &z_, int nx_, int ny_, int nz_, double volumes[])
    void compute_surface_areas(const vector[double] &weights, 
      double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
      const vector[double] &x_, const vector[double] &y_, const vector[double] &z_, int nx_, int ny_, int nz_, double areas[])
    void compute_hessian(const vector[double] &weights, 
      double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
      const vector[double] &x_, const vector[double] &y_, const vector[double] &z_, int nx_, int ny_, int nz_,
      vector[int] &row_coordinates, vector[int] &column_coordinates, vector[double] &hessian_entries)
    void compute_voronoi_tessellation(const vector[double] &weights, 
      double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
      const vector[double] &x_, const vector[double] &y_, const vector[double] &z_, int nx_, int ny_, int nz_,
      vector[vector[vector[int]]] &vertices_indices, vector[vector[double]] &vertices, vector[vector[int]] &neighbors_indices, vector[bool_t] &computed_cells)
    void compute_voronoi_tessellation_bounded(const vector[double] &weights, 
        double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
        const vector[double] &x_, const vector[double] &y_, const vector[double] &z_, int nx_, int ny_, int nz_, 
        const vector[bool_t] &cells_to_compute, const vector[bool_t] &x_outside, const vector[bool_t] &y_outside, const vector[bool_t] &z_outside,
        vector[vector[vector[int]]] &vertices_indices, vector[vector[double]] &vertices, vector[vector[int]] &neighbors, vector[bool_t] &computed_cells)
    void compute_section(const vector[double] &weights, 
        double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
        const vector[double] &x_, const vector[double] &y_, const vector[double] &z_, int nx_, int ny_, int nz_, 
        double a, double b, double c, double x_0, double y_0, double z_0, vector[int] &section_indices, vector[double] &areas, vector[vector[double]] &section_vertices)
    void compute_section_bounded(const vector[double] &weights, 
        double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
        const vector[double] &x_, const vector[double] &y_, const vector[double] &z_, int nx_, int ny_, int nz_, 
        double a, double b, double c, double x_0, double y_0, double z_0, 
        const vector[bool_t] &cells_to_compute, const vector[bool_t] &x_outside, const vector[bool_t] &y_outside, const vector[bool_t] &z_outside, 
        vector[int] &section_indices, vector[double] &areas, vector[vector[double]] &section_vertices)
    void compute_num_fragments(const vector[double] &weights, 
      double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
      const vector[double] &x_, const vector[double] &y_, const vector[double] &z_, int nx_, int ny_, int nz_, vector[vector[bool_t]] &cell_sections)


cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr)
    void* malloc(size_t size)

# cdef extern from "numpy/arrayobject.h":
#     void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

import sys
import math

# np.import_array()
# ctypedef np.int32_t DTYPE_int32
# ctypedef np.float64_t DTYPE_double

def same_rows(a, b, tol=8):
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return np.all(np.any(rows_close, axis=-1), axis=-1) and np.all(np.any(rows_close, axis=0), axis=0)


# cdef data_to_numpy_array_with_spec(void * ptr, np.npy_intp N, int t):
#     cdef np.ndarray[DTYPE_int32, ndim=1] arr = np.PyArray_SimpleNewFromData(1, &N, t, ptr)
#     PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
#     return arr


def get_3d_volumes(weights, points, domain, periodic):
    n = weights.shape[0]
    l1, l2, l3 = domain[0][1] - domain[0][0], domain[1][1] - domain[1][0], domain[2][1] - domain[2][0]
    domain_volume = l1 * l2 * l3
    block_fraction = (0.2*n/domain_volume)**(1./3)
    blocks = max(round(block_fraction * l1), 1), max(round(block_fraction * l2), 1), max(round(block_fraction * l3), 1)

    #cdef vector[double] volumes = vector[double](n, 0.0)
    volumes = np.zeros(n)
    cdef double[::1] volumes_memview = volumes
    compute_volumes(weights, 
            <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
            <int>periodic, <int>periodic, <int>periodic, n, points[:, 0], points[:, 1], points[:, 2], <int>blocks[0], <int>blocks[1], <int>blocks[2], &volumes_memview[0])

    return volumes


def get_3d_surface_areas(weights, points, domain, periodic):
    n = weights.shape[0]
    l1, l2, l3 = domain[0][1] - domain[0][0], domain[1][1] - domain[1][0], domain[2][1] - domain[2][0]
    domain_volume = l1 * l2 * l3
    block_fraction = (0.2*n/domain_volume)**(1./3)
    blocks = max(round(block_fraction * l1), 1), max(round(block_fraction * l2), 1), max(round(block_fraction * l3), 1)

    #cdef vector[double] volumes = vector[double](n, 0.0)
    areas = np.zeros(n)
    cdef double[::1] areas_memview = areas
    compute_surface_areas(weights, 
            <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
            <int>periodic, <int>periodic, <int>periodic, n, points[:, 0], points[:, 1], points[:, 2], <int>blocks[0], <int>blocks[1], <int>blocks[2], &areas_memview[0])

    return areas


def get_3d_centroids(weights, points, domain, periodic):
    n = weights.shape[0]
    l1, l2, l3 = domain[0][1] - domain[0][0], domain[1][1] - domain[1][0], domain[2][1] - domain[2][0]
    domain_volume = l1 * l2 * l3
    block_fraction = (0.2*n/domain_volume)**(1./3)
    blocks = max(round(block_fraction * l1), 1), max(round(block_fraction * l2), 1), max(round(block_fraction * l3), 1)

    #cdef vector[double] centroids = vector[double](3*n, 0.0)
    centroids = np.zeros(n*3)
    cdef double[::1] centroids_memview = centroids
    compute_centroids(weights, 
            <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
            <int>periodic, <int>periodic, <int>periodic, n, points[:, 0], points[:, 1], points[:, 2], <int>blocks[0], <int>blocks[1], <int>blocks[2], &centroids_memview[0])
    return np.reshape(centroids, (n, 3), order='F')


def get_3d_hessian(weights, points, domain, periodic):
    n = weights.shape[0]
    l1, l2, l3 = domain[0][1] - domain[0][0], domain[1][1] - domain[1][0], domain[2][1] - domain[2][0]
    domain_volume = l1 * l2 * l3
    block_fraction = (0.2*n/domain_volume)**(1./3)
    blocks = max(round(block_fraction * l1), 1), max(round(block_fraction * l2), 1), max(round(block_fraction * l3), 1)

    cdef vector[int] row_coordinates
    cdef vector[int] column_coordinates
    cdef vector[double] hessian_entries
    
    compute_hessian(weights, 
            <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
            <int>periodic, <int>periodic, <int>periodic, n, points[:, 0], points[:, 1], points[:, 2], <int>blocks[0], <int>blocks[1], <int>blocks[2],
            row_coordinates, column_coordinates, hessian_entries)
    # This involves three fairly expensive copy operations, seems difficult to convert std::vector to numpy array without copying data
    # idea: compute number of neighbors simultaneously with cell volumes. Number of neighbors can be used to determine the size of row_coordinates, column_coordinates and hessian_entries
    # By knowing the number of neighbors of each cell we can fill these arrays in order, hence we can use the default structure used for csr matrices
    rows_np = np.asarray(row_coordinates)
    cols_np = np.asarray(column_coordinates)
    hess_np = np.asarray(hessian_entries)

    return csr_matrix((hess_np, (rows_np, cols_np)), shape=(n, n))


def section(weights, points, domain, periodic, coeffs, offset):
    n = weights.shape[0]
    l1, l2, l3 = domain[0][1] - domain[0][0], domain[1][1] - domain[1][0], domain[2][1] - domain[2][0]
    domain_volume = l1 * l2 * l3
    block_fraction = (0.2*n/domain_volume)**(1./3)
    blocks = max(round(block_fraction * l1), 1), max(round(block_fraction * l2), 1), max(round(block_fraction * l3), 1)

    cdef vector[double] areas
    cdef vector[int] section_indices
    cdef vector[vector[double]] section_vertices

    compute_section(weights, 
            <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
            <int>periodic, <int>periodic, <int>periodic, n, points[:, 0], points[:, 1], points[:, 2], <int>blocks[0], <int>blocks[1], <int>blocks[2],
            <double>coeffs[0], <double>coeffs[1], <double>coeffs[2], <double>offset[0], <double>offset[1], <double>offset[2], section_indices, areas, section_vertices)
    vertices_3d = [np.reshape(cell, (len(cell)//3, 3)) for cell in section_vertices]
    return section_indices, areas, vertices_3d


def section_bounded(weights, points, domain, periodic, coeffs, offset, cells_to_compute, x_outside, y_outside, z_outside):
    n = weights.shape[0]
    l1, l2, l3 = domain[0][1] - domain[0][0], domain[1][1] - domain[1][0], domain[2][1] - domain[2][0]
    domain_volume = l1 * l2 * l3
    block_fraction = (0.2*n/domain_volume)**(1./3)
    blocks = max(round(block_fraction * l1), 1), max(round(block_fraction * l2), 1), max(round(block_fraction * l3), 1)

    cdef vector[double] areas
    cdef vector[int] section_indices
    cdef vector[vector[double]] section_vertices

    compute_section_bounded(weights, 
            <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
            <int>periodic, <int>periodic, <int>periodic, n, points[:, 0], points[:, 1], points[:, 2], <int>blocks[0], <int>blocks[1], <int>blocks[2],
            <double>coeffs[0], <double>coeffs[1], <double>coeffs[2], <double>offset[0], <double>offset[1], <double>offset[2],
            cells_to_compute, x_outside, y_outside, z_outside, section_indices, areas, section_vertices)
    vertices_3d = [np.reshape(cell, (len(cell)//3, 3)) for cell in section_vertices]
    return section_indices, areas, vertices_3d


def voronoi_num_fragments(weights, points, domain, periodic):
    n = weights.shape[0]
    l1, l2, l3 = domain[0][1] - domain[0][0], domain[1][1] - domain[1][0], domain[2][1] - domain[2][0]
    domain_volume = l1 * l2 * l3
    block_fraction = (0.2*n/domain_volume)**(1./3)
    blocks = max(round(block_fraction * l1), 1), max(round(block_fraction * l2), 1), max(round(block_fraction * l3), 1)

    cdef vector[vector[bool_t]] cell_sections

    compute_num_fragments(weights, 
            <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
            <int>periodic, <int>periodic, <int>periodic, n, points[:, 0], points[:, 1], points[:, 2], <int>blocks[0], <int>blocks[1], <int>blocks[2], cell_sections)
    return np.array(cell_sections, dtype=bool)


# def cells_contain(weights, points, domain, periodic, pts_to_check):
#     n = weights.shape[0]
#     l1, l2, l3 = domain[0][1] - domain[0][0], domain[1][1] - domain[1][0], domain[2][1] - domain[2][0]
#     domain_volume = l1 * l2 * l3
#     block_fraction = (0.2*n/domain_volume)**(1./3)
#     blocks = max(round(block_fraction * l1), 1), max(round(block_fraction * l2), 1), max(round(block_fraction * l3), 1)

#     cdef vector[vector[vector[int]]] vertices_indices
#     cdef vector[vector[double]] vertices
#     cdef vector[vector[int]] neighbors
#     cdef vector[bool_t] computed_cells

#     compute_voronoi_tessellation(weights, 
#         <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
#         <int>periodic, <int>periodic, <int>periodic, n, points[:, 0], points[:, 1], points[:, 2], <int>blocks[0], <int>blocks[1], <int>blocks[2],
#         vertices_indices, vertices, neighbors, computed_cells)

#     cells = []
#     for i in range(n):
#         if computed_cells[i]:
            
            
#     return



def get_cells(weights, points, domain, periodic):
    n = weights.shape[0]
    l1, l2, l3 = domain[0][1] - domain[0][0], domain[1][1] - domain[1][0], domain[2][1] - domain[2][0]
    domain_volume = l1 * l2 * l3
    block_fraction = (0.2*n/domain_volume)**(1./3)
    blocks = max(round(block_fraction * l1), 1), max(round(block_fraction * l2), 1), max(round(block_fraction * l3), 1)

    cdef vector[vector[vector[int]]] vertices_indices
    cdef vector[vector[double]] vertices
    cdef vector[vector[int]] neighbors
    cdef vector[bool_t] computed_cells

    compute_voronoi_tessellation(weights, 
        <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
        <int>periodic, <int>periodic, <int>periodic, n, points[:, 0], points[:, 1], points[:, 2], <int>blocks[0], <int>blocks[1], <int>blocks[2],
        vertices_indices, vertices, neighbors, computed_cells)

    cells = []
    for i in range(n):
        if computed_cells[i]:
            cell_faces = [{'adjacent_cell': neighbors[i][j], 'vertices': vertices_indices[i][j]} for j in range(len(neighbors[i]))]
            cell_vertices = list(np.reshape(vertices[i], (len(vertices[i])//3, 3)))
            cells.append({'faces': cell_faces, 'vertices': cell_vertices})
        else:
            cells.append(dict())
    return cells


def get_cells_bounded(weights, points, domain, periodic, cells_to_compute, x_outside, y_outside, z_outside):
    n = weights.shape[0]
    l1, l2, l3 = domain[0][1] - domain[0][0], domain[1][1] - domain[1][0], domain[2][1] - domain[2][0]
    domain_volume = l1 * l2 * l3
    block_fraction = (0.2*n/domain_volume)**(1./3)
    blocks = max(round(block_fraction * l1), 1), max(round(block_fraction * l2), 1), max(round(block_fraction * l3), 1)

    cdef vector[vector[vector[int]]] vertices_indices
    cdef vector[vector[double]] vertices
    cdef vector[vector[int]] neighbors
    cdef vector[bool_t] computed_cells

    compute_voronoi_tessellation_bounded(weights, 
        <double>domain[0][0], <double>domain[0][1], <double>domain[1][0], <double>domain[1][1], <double>domain[2][0], <double>domain[2][1], 
        <int>periodic, <int>periodic, <int>periodic, n, points[:, 0], points[:, 1], points[:, 2], <int>blocks[0], <int>blocks[1], <int>blocks[2],
        cells_to_compute, x_outside, y_outside, z_outside, vertices_indices, vertices, neighbors, computed_cells)

    cells = []
    for i in range(n):
        if computed_cells[i]:
            cell_faces = [{'adjacent_cell': neighbors[i][j], 'vertices': vertices_indices[i][j]} for j in range(len(neighbors[i]))]
            cell_vertices = list(np.reshape(vertices[i], (len(vertices[i])//3, 3)))
            cells.append({'faces': cell_faces, 'vertices': cell_vertices})
        else:
            cells.append(dict())

    return cells

      

# def compute_centroidal_laguerre2d(weights_init, points_init, target_areas, domain, periodicity, permute, 
#     max_iterations, gradient_max_iterations, tol, gradient_tol, disp, fix_num_iterations, solver, respect_bounds,
#     num_cpus):
    
#     n = weights_init.shape[0]
#     cdef pair[vector[double], vector[double]] volumes_centroids
#     weights = weights_init
#     points = np.hstack([points_init, np.ones((weights.shape[0], 1))*0.5])
#     previous_points = np.hstack([points_init, np.ones((weights.shape[0], 1))*0.5])
#     L1 = domain[0][1] - domain[0][0]
#     L2 = domain[1][1] - domain[1][0]
#     L3 = 1
#     domain_ = domain + [[0, 1]]
#     Lengths = np.array([L1, L2, L3])
#     domain_volume = L1 * L2 * L3
#     block_fraction = (1.*n/domain_volume)**(1./3)
#     blocks = round(block_fraction * L1), round(block_fraction * L2), round(block_fraction * L3)
#     block_size = np.max(Lengths)*0.2
#     max_side = np.max(Lengths)
#     periodicity_ = periodicity + [False]

#     areas = np.zeros(n)

#     num_iterations = 0
#     stopping_criterion = tol * tol * domain_volume * max_side * max_side
#     residual = 10 * stopping_criterion
#     gradient_convergence = True

#     while residual > stopping_criterion and num_iterations < max_iterations:
#         if solver == "BB":
#             weights, gradient_convergence = BB_solver(weights, points, target_areas, domain_, block_size, periodicity_, 
#                 permute=permute, max_iter=gradient_max_iterations, tol=gradient_tol, disp=disp, num_cpus=num_cpus)
#         elif solver == "MM":
#             weights, gradient_convergence = MM_solver(weights, points, target_areas, domain_, block_size, periodicity_, 
#                 permute=permute, max_iter=gradient_max_iterations, tol=gradient_tol, disp=disp, num_cpus=num_cpus)
#         else:
#             print("Invalid solver provided, defaulting to MM.")
#             compute_centroidal_laguerre(weights_init, points_init, target_areas, domain, periodicity, permute, 
#                 max_iterations, gradient_max_iterations, tol, gradient_tol, disp, fix_num_iterations, "MM", respect_bounds, num_cpus)

#         volumes_centroids = compute_volumes_centroids(weights, 
#             <double>domain_[0][0], <double>domain_[0][1], <double>domain_[1][0], <double>domain_[1][1], <double>domain_[2][0], <double>domain_[2][1], 
#                 <int>periodicity_[0], <int>periodicity_[1], <int>periodicity_[2], n, points[:, 0], points[:, 1], points[:, 2], <int>num_cpus, <int>blocks[0], <int>blocks[1], <int>blocks[2])
#         areas = np.array(volumes_centroids.first)
#         centroids = np.reshape(volumes_centroids.second, (n, 3), order='F')
#         centroids[:, 2] = 0.5

#         if not fix_num_iterations:
#             residual = np.sum(target_areas * np.sum(np.square(points - centroids), axis=1))

#         points[:, 2] = 0.5
#         previous_points = points
#         points = centroids
#         num_iterations += 1

#     if permute:
#         target_areas_ = np.array(function_definitions.permute_targets(areas, target_areas, n))
#     else:
#         target_areas_ = target_areas

#     if periodicity[0]:
#         previous_points[:, 0] = np.remainder(previous_points[:, 0], L1)
#     if periodicity[1]:
#         previous_points[:, 1] = np.remainder(previous_points[:, 1], L2)

#     if (residual > stopping_criterion and not fix_num_iterations) or not gradient_convergence:
#         warnings.warn("Algorithm did not converge in max_iterations iterations.")
#         return {"areas": areas, "weights": weights, "points": previous_points[:, 0:2], 
#             "target_areas": target_areas_, "convergence": False, "centroids": points[:, 0:2]}
#     elif disp:
#         print("Algorithm number of iterations: ", num_iterations)

#     return {"areas": areas, "weights": weights, "points": previous_points[:, 0:2], 
#         "target_areas": target_areas_, "convergence": True, "centroids": points[:, 0:2]}



# @cython.wraparound(False)
# def compute_voronoi_section_bounded(coeffs, offset, points, limits, dispersion, cells_to_compute, x_outside, y_outside, z_outside, radii, periodic):
#   """
# Input arg formats:
#   points = list of 3-vectors (lists or compatible class instances) of doubles,
#     being the coordinates of the points to voronoi-tesselate.
#   limits = 3-list of 2-lists, specifying the start and end sizes of the box the
#     points are in.
#   dispersion = max distance between two points that might be adjacent (sets
#     voro++ block sizes.)
#   radii (optional) = list of python floats as the sphere radii of the points,
#     for radical (weighted) tessellation.
#   periodic (optional) = 3-list of bools indicating x, y and z periodicity of 
#     the system box.
  
# Output format is a list of cells as follows:
#   [ # list in same order as original points.
#     {
#       'volume' : 1.0,
#       'vertices' : [[1.0, 2.0, 3.0], ...], # positions of vertices
#       'adjacency' : [[1,3,4, ...], ...], # cell-vertices adjacent to i by index
#       'faces' : [
#         {
#           'vertices' : [7,4,13, ...], # vertex ids in loop order
#           'adjacent_cell' : 34 # *cell* id, negative if a wall
#         }, ...]
#       'original' : point[index] # the original instance from args
#     },
#     ... 
#   ]
  
#   NOTE: The class from items in input points list is reused for all 3-vector
#   outputs. It must have a constructor which accepts a list of 3 python floats
#   (python's list type does satisfy this requirement.)
#   """
#   cdef Py_ssize_t n = len(points), i, j
#   cdef double *xs
#   cdef double *ys
#   cdef double *zs
#   cdef double *rs
#   cdef void** voronoi_cells
  
#   vector_class = get_constructor(points[0])

#   periodic = [1 if p else 0 for p in periodic]
  
#   # we must make sure we have at least one block, or voro++ will segfault when
#   # we look for cells.
  
#   blocks = [
#     max([1, int(math.floor((limits[0][1] - limits[0][0]) / dispersion))]),
#     max([1, int(math.floor((limits[1][1] - limits[1][0]) / dispersion))]),
#     max([1, int(math.floor((limits[2][1] - limits[2][0]) / dispersion))])
#   ]
  
#   # if no radii provided, we still run the radical routine, but with all the same small radius.
#   if len(radii) != len(points):
#     radii = [dispersion / 10.] * len(points)
  
#   # build the container object
#   cdef void* container = container_poly_create(
#     <double>limits[0][0],
#     <double>limits[0][1],
#     <double>limits[1][0],
#     <double>limits[1][1],
#     <double>limits[2][0],
#     <double>limits[2][1],
#     <int>blocks[0],
#     <int>blocks[1],
#     <int>blocks[2],
#     <int>periodic[0],
#     <int>periodic[1],
#     <int>periodic[2]
#   )
  
#   xs = <double*>malloc(sizeof(double) * n)
#   ys = <double*>malloc(sizeof(double) * n)
#   zs = <double*>malloc(sizeof(double) * n)
#   rs = <double*>malloc(sizeof(double) * n)
  
#   # initialise particle positions:
#   for i in range(n):
#     xs[i] = <double>points[i][0]
#     ys[i] = <double>points[i][1]
#     zs[i] = <double>points[i][2]
#     rs[i] = <double>radii[i]
    
#   # and add them to the container:
#   put_particles(container, n, xs, ys, zs, rs)
    
#   # now compute the tessellation:
#   result = compute_section_bounded(container, n, <double>coeffs[0], <double>coeffs[1], <double>coeffs[2], 
#     <double>offset[0], <double>offset[1], <double>offset[2], cells_to_compute, x_outside, y_outside, z_outside)
#   voronoi_cells = result.first
#   section_indices = result.second
      
#   # extract the Voronoi cells into python objects:
#   cdef vector[double] vertex_positions
#   cdef void** lists = NULL
#   cdef vector[int]* vptr = NULL
#   vertices_3d = []
#   section_areas = [] # np.zeros(len(section_indices))

#   for i in range(len(section_indices)):
#     vertex_positions = cell_get_vertex_positions(voronoi_cells[section_indices[i]], xs[section_indices[i]], ys[section_indices[i]], zs[section_indices[i]])
#     cell_vertices = []
#     for j in range(<Py_ssize_t>(vertex_positions.size() // 3)):
#       cell_vertices.append(vector_class([
#         float(vertex_positions[3 * j]),
#         float(vertex_positions[3 * j + 1]),
#         float(vertex_positions[3 * j + 2])
#       ]))
    
#     lists = cell_get_faces(voronoi_cells[section_indices[i]])
    
#     faces_areas = cell_get_areas(voronoi_cells[section_indices[i]])
#     j=0
#     while lists[j] != NULL:
#       face_vertices = []
#       vptr = <vector[int]*>lists[j]
#       for k in range(vptr.size() - 1):
#         face_vertices.append(int(deref(vptr)[k]))
#       if int(deref(vptr)[vptr.size() - 1]) == n+1:
#         section_areas.append(faces_areas[j])
#         vertices_3d.append(np.array(cell_vertices)[np.array(face_vertices, dtype=np.intc)])
#       del vptr
#       j += 1
      
#     free(lists)
  
#   section_areas = np.array(section_areas)
#   duplicates = set()

#   if len(section_areas) > 0:
#     idx_sort = np.argsort(section_areas)
#     sorted_areas = section_areas[idx_sort]
#     sorted_indices = idx_sort[idx_sort]

#     vals, idx_start, count = np.unique(sorted_areas, return_counts=True, return_index=True)

#     pairs = set()

#     for i, first_occurrence in enumerate(idx_start):
#       if count[i] > 1:
#         for p in range(first_occurrence, first_occurrence + count[i]):
#           for q in range(p+1, first_occurrence + count[i]):
#             if ((q, p) not in pairs) and ((p, q) not in pairs):
#               if vertices_3d[sorted_indices[p]].shape == vertices_3d[sorted_indices[q]].shape:
#                 if same_rows(vertices_3d[sorted_indices[p]], vertices_3d[sorted_indices[q]]):
#                   duplicates.add(sorted_indices[q])
#               pairs.add((p, q))
  
#   vertices_3d = [vertices_3d[j] for j in range(len(vertices_3d)) if j not in duplicates]
#   section_indices_ = np.delete(section_indices, list(duplicates))
#   section_areas_ = np.delete(section_areas, list(duplicates))

#   if coeffs[0] == 0 and coeffs[1] == 0:
#     cross_section_2d = [np.array(cell)[:, 0:2] for cell in vertices_3d]
#   else:
#     cross_section_2d = transform_2d(vertices_3d, coeffs, offset)

#   # finally, tidy up.
#   dispose_all(container, voronoi_cells, n)
#   free(xs)
#   free(ys)
#   free(zs)
#   free(rs)
#   return {"3d": vertices_3d, "original_indices": section_indices_, "areas": section_areas_, "2d": cross_section_2d}
