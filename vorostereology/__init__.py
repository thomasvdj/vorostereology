from . import voroplusplus
from itertools import product
import numpy as np
import os


# Author: Thomas van der Jagt
# Note: The functions "compute_voronoi" and "compute_2d_voronoi" were originally written by Joe Jordan and modified
# to also support empty cells.


def compute_voronoi(points, limits, weights, periodic=None, respect_bounds=True):
    """
    Input arg formats:

    points = list of 3-vectors (lists or compatible class instances) of doubles,
        being the coordinates of the points to Voronoi-tessellate.
    limits = 3-list of 2-lists, specifying the start and end sizes of the box the
        points are in.
    weights (optional) = list of python floats as the weights of the points,
        for radical (weighted) tessellation.
    periodic (optional) = 3-list of bools indicating x, y and z periodicity of
        the system box.
    respect_bounds (optional) = bool, only used when periodic boundary conditions
        are used. If true it ensures that cells are fully within the domain.

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
    if periodic is None:
        periodic = [False, False, False]
    
    if respect_bounds and (periodic[0] or periodic[1] or periodic[2]):
        return compute_voronoi_bounded(points, limits, np.sqrt(weights - np.min(weights)), periodic)
    else:
        L1 = limits[0][1] - limits[0][0]
        L2 = limits[1][1] - limits[1][0]
        L3 = limits[2][1] - limits[2][0]
        Lengths = np.array([L1, L2, L3])
        block_size = np.max(Lengths) * 0.2

        return voroplusplus.compute_voronoi(points, limits, block_size, np.sqrt(weights - np.min(weights)), periodic)


def compute_voronoi_bounded(points, domain, radii, periodic=None):

    if periodic is None:
        periodic = [False, False, False]

    L1 = domain[0][1] - domain[0][0]
    L2 = domain[1][1] - domain[1][0]
    L3 = domain[2][1] - domain[2][0]
    Lengths = np.array([L1, L2, L3])
    block_size = np.max(Lengths) * 0.2
    n = len(points)

    section_locations = np.array(voroplusplus.voronoi_num_fragments(points, domain, dispersion=block_size,
                                                                    radii=radii, periodic=periodic), dtype=bool)
    max_intersections = np.max(np.sum(section_locations, axis=1))
    if max_intersections == 0:
        num_iterations = 0
    elif max_intersections == 1:
        num_iterations = 2
    elif max_intersections == 2:
        num_iterations = 4
    else:
        num_iterations = 8

    x_translations = np.zeros((n, num_iterations), dtype=np.int8)
    y_translations = np.zeros((n, num_iterations), dtype=np.int8)
    z_translations = np.zeros((n, num_iterations), dtype=np.int8)

    for i in range(n):
        x_col = section_locations[i, 0] or section_locations[i, 1]
        y_col = section_locations[i, 2] or section_locations[i, 3]
        z_col = section_locations[i, 4] or section_locations[i, 5]
        num_axis = int(x_col) + int(y_col) + int(z_col)
        if num_axis > 0:
            x_loc = [0, 0]
            y_loc = [0, 0]
            z_loc = [0, 0]
            if x_col:
                if section_locations[i, 0]:
                    x_loc = [0, 1]
                else:
                    x_loc = [0, -1]
            if y_col:
                if section_locations[i, 2]:
                    y_loc = [0, 1]
                else:
                    y_loc = [0, -1]
            if z_col:
                if section_locations[i, 4]:
                    z_loc = [0, 1]
                else:
                    z_loc = [0, -1]
            if num_axis == 1:
                if x_col:
                    x_translations[i, 1] = x_loc[1]
                elif y_col:
                    y_translations[i, 1] = y_loc[1]
                elif z_col:
                    z_translations[i, 1] = z_loc[1]
            elif num_axis == 2:
                if x_col and y_col:
                    combinations = list(product(x_loc, y_loc))
                    for j in range(4):
                        x_translations[i, j] = combinations[j][0]
                        y_translations[i, j] = combinations[j][1]
                elif x_col and z_col:
                    combinations = list(product(x_loc, z_loc))
                    for j in range(4):
                        x_translations[i, j] = combinations[j][0]
                        z_translations[i, j] = combinations[j][1]
                elif y_col and z_col:
                    combinations = list(product(y_loc, z_loc))
                    for j in range(4):
                        y_translations[i, j] = combinations[j][0]
                        z_translations[i, j] = combinations[j][1]
            elif num_axis == 3:
                combinations = list(product(x_loc, y_loc, z_loc))
                for j in range(8):
                    x_translations[i, j] = combinations[j][0]
                    y_translations[i, j] = combinations[j][1]
                    z_translations[i, j] = combinations[j][2]

    result = []
    first_iteration = True
    for j in range(num_iterations):
        if first_iteration:
            cells_to_compute = [True] * n
            first_iteration = False
            x_outside = x_translations[:, j] != 0
            y_outside = y_translations[:, j] != 0
            z_outside = z_translations[:, j] != 0
            result = voroplusplus.compute_voronoi_bounded(points, domain, block_size, cells_to_compute, x_outside,
                                                          y_outside, z_outside, radii, periodic)
        else:
            x_outside = x_translations[:, j] != 0
            y_outside = y_translations[:, j] != 0
            z_outside = z_translations[:, j] != 0
            cells_to_compute = list(np.logical_or(np.logical_or(x_outside, y_outside), z_outside))
            points_ = np.copy(points)
            points_[:, 0] = points_[:, 0] + x_translations[:, j] * L1
            points_[:, 1] = points_[:, 1] + y_translations[:, j] * L2
            points_[:, 2] = points_[:, 2] + z_translations[:, j] * L3
            res = voroplusplus.compute_voronoi_bounded(points_, domain, block_size, cells_to_compute, x_outside,
                                                       y_outside, z_outside, radii, periodic)

            for cell_idx, cell in enumerate(res):
                if len(cell) > 0:
                    if len(result[cell_idx]) > 0:
                        num_vertices = len(result[cell_idx]["vertices"])
                        result[cell_idx]["vertices"] += cell["vertices"]
                        result[cell_idx]["volume"] += cell["volume"]
                        new_faces = cell["faces"]
                        for k in range(len(new_faces)):
                            new_faces[k]["vertices"] = [m + num_vertices for m in new_faces[k]["vertices"]]
                        result[cell_idx]["faces"] += new_faces
                    else:
                        result[cell_idx]["original"] = points[cell_idx]
                        result[cell_idx]["vertices"] = cell["vertices"]
                        result[cell_idx]["volume"] = cell["volume"]
                        result[cell_idx]["faces"] = cell["faces"]

    return result


def compute_centroidal_laguerre(weights_init, points_init, target_volumes, domain, periodicity,
                                permute=False, max_iterations=100, gradient_max_iterations=None, tol=0.001,
                                gradient_tol=0.01, disp=False, fix_num_iterations=False, solver="MM",
                                respect_bounds=True, num_cpus=None):
    """
    Function for computing a regularised Laguerre-Voronoi diagram with a prescribed volume distribution in 3D.

    :arg weights_init: The initial weights used for the algorithm, a 1 dimensional numpy array of length n, where n
        is the number of cells.
    :arg points_init: The initial locations of the seeds/ generator points, a numpy array of shape (n, 3), where n
        is the number of cells.
    :arg target_volumes: The desired volumes for the cells in the Laguerre diagram, a 1 dimensional numpy array of
        length n, where n is the number of cells.
    :arg domain: The domain that is used: a 3-list of 2-lists, specifying the start and end sizes of the box the
        points are in. Only boxes are supported.
    :arg periodicity:  3-list of bools indicating x, y and z periodicity of the domain.
    :arg permute: A bool indicating whether to include a permutation step in the algorithm, which may in some cases
        speed up the algorithm. If set to true the final volumes of the cells will be a permutation of the
        :arg target_volumes: (up to the prescribed error tolerance of course).
    :arg max_iterations: An integer indicating the maximum number of iterations (Lloyd centering steps) of the
        algorithm.
    :arg tol: A float indicating the error tolerance for the distance between the final generator points locations
        and the centroids of the final Laguerre diagram.
    :arg gradient_max_iterations: An integer indicating the maximum number of iterations for the gradient method
        within each iteration of the algorithm.
    :arg gradient_tol: A float indicating the error tolerance regarding how close the final volumes of the cells
        need to be to the :arg target_volumes:.
    :arg disp: A bool indicating whether to print some information regarding the amount of iterations to the
        python console.
    :arg fix_num_iterations: A bool indicating whether to run a fixed number of iterations (Lloyd centering steps),
        if set to True the algorithm will ignore :arg tol: and simply run the algorithm for :arg max_iterations: steps.
    :arg solver: A string indicating the gradient method te be used to minimize the convex function -g. Options are:
        "BB": Barzilai-Borwein, "MM": Malitsky-Mishchenko, "BFGS": Broyden–Fletcher–Goldfarb–Shanno algorithm.

    The arguments :arg permute:, :arg max_iterations:, :arg gradient_max_iterations:, :arg tol:, :arg gradient_tol:,
        :arg disp: and :arg fix_num_iterations: are optional and are set to reasonable defaults.

    :return dict: A python dictionary is returned with the following keys and values:
        "volumes": A 1 dimensional numpy array of length n containing the volumes of the final Laguerre diagram.
        "weights": A 1 dimensional numpy array of length n containing the weights of the final Laguerre diagram.
        "points": A numpy array of shape (n, 3) containing the positions of the generator points of the final
            Laguerre diagram.
        "target_volumes": A copy of the provided :arg target_volumes:. However, if :arg permute: was set to True, then
            a permutation of :arg target_volumes: is given instead, which best aligns with the final volumes of the
            Laguerre diagram in the sense that it minimizes the gradient of the function that is minimized.
        "convergence": A bool indicating whether the algorithm converged. If False then "laguerre" will be an
            empty list.

    This functions implements Algorithm 2 as described in the paper 'Laguerre tessellations and polycrystalline
    microstructures: A fast algorithm for generating grains of given volumes' by D.P. Bourne, P.J.J. Kok, S.M. Roper,
    W.D.T. Spanjer.

    The gradient method that is used is the Barzilai-Borwein method and the convergence criterion regarding the
    closeness of the final generator points to the actual centroids that :arg tol: refers to is described in the
    paper "Fast methods for computing centroidal Laguerre tessellations for prescribed volume fractions with
    applications to microstructure generation of polycrystalline materials" by Jannick Kuhna, Matti Schneiderb,
    Petra Sonnweber-Ribica, Thomas Böhlke.

    All of the arrays "volumes", "weights", "points", "target_volumes" follow the same order. This means:
        At index j of "volumes" we find the volume of the j-th cell in the Laguerre diagram.
        At index j of "weights" we find the weight of the j-th cell in the Laguerre diagram.
        At index j of "points" we find the position of the generator point of the j-th cell in the Laguerre diagram.
        At index j of "target volumes" we find the desired volume of the j-th cell in the Laguerre diagram.
    """
    if gradient_max_iterations is None:
        gradient_max_iterations = weights_init.shape[0] * 10
    if num_cpus is None:
        num_cpus = os.cpu_count()
    return voroplusplus.compute_centroidal_laguerre(weights_init, points_init, target_volumes, domain, periodicity,
                                                    permute, max_iterations, gradient_max_iterations, tol, gradient_tol,
                                                    disp, fix_num_iterations, solver, respect_bounds, num_cpus)


def compute_centroidal_laguerre2d(weights_init, points_init, target_areas, domain, periodicity,
                                  permute=False, max_iterations=100, gradient_max_iterations=None, tol=0.001,
                                  gradient_tol=0.01, disp=False, fix_num_iterations=False, solver="MM",
                                  respect_bounds=True, num_cpus=None):
    """
    Function for computing a regularised Laguerre-Voronoi diagram with a prescribed area distribution in 2D.

    :arg weights_init: The initial weights used for the algorithm, a 1 dimensional numpy array of length n, where n
        is the number of cells.
    :arg points_init: The initial locations of the seeds/ generator points, a numpy array of shape (n, 2), where n
        is the number of cells.
    :arg target_areas: The desired areas for the cells in the Laguerre diagram, a 1 dimensional numpy array of
        length n, where n is the number of cells.
    :arg domain: The domain that is used: a 2-list of 2-lists, specifying the start and end sizes of the box the
        points are in. Only boxes are supported.
    :arg periodicity:  2-list of bools indicating x, y and z periodicity of the domain.
    :arg permute: A bool indicating whether to include a permutation step in the algorithm, which may in some cases
        speed up the algorithm. If set to true the final volumes of the cells will be a permutation of the
        :arg target_areas: (up to the prescribed error tolerance of course).
    :arg max_iterations: An integer indicating the maximum number of iterations (Lloyd centering steps) of the
        algorithm.
    :arg tol: A float indicating the error tolerance for the distance between the final generator points locations
        and the centroids of the final Laguerre diagram.
    :arg gradient_max_iterations: An integer indicating the maximum number of iterations for the gradient method
        within each iteration of the algorithm.
    :arg gradient_tol: A float indicating the error tolerance regarding how close the final volumes of the cells
        need to be to the :arg target_volumes:.
    :arg disp: A bool indicating whether to print some information regarding the amount of iterations to the
        python console.
    :arg fix_num_iterations: A bool indicating whether to run a fixed number of iterations (Lloyd centering steps),
        if set to True the algorithm will ignore :arg tol: and simply run the algorithm for :arg max_iterations: steps.
    :arg solver: A string indicating the gradient method te be used to minimize the convex function -g. Options are:
        "BB": Barzilai-Borwein, "MM": Malitsky-Mishchenko, "BFGS": Broyden–Fletcher–Goldfarb–Shanno algorithm.

    The arguments :arg permute:, :arg max_iterations:, :arg gradient_max_iterations:, :arg tol:, :arg gradient_tol:,
        :arg disp: and :arg fix_num_iterations: are optional and are set to reasonable defaults.

    :return dict: A python dictionary is returned with the following keys and values:
        "areas": A 1 dimensional numpy array of length n containing the volumes of the final Laguerre diagram.
        "weights": A 1 dimensional numpy array of length n containing the weights of the final Laguerre diagram.
        "points": A numpy array of shape (n, 2) containing the positions of the generator points of the final
            Laguerre diagram.
        "target_areas": A copy of the provided :arg target_areas:. However, if :arg permute: was set to True, then
            a permutation of :arg target_areas: is given instead, which best aligns with the final areas of the
            Laguerre diagram in the sense that it minimizes the gradient of the function that is minimized.
        "convergence": A bool indicating whether the algorithm converged. If False then "laguerre" will be an
            empty list.

    This functions implements Algorithm 2 as described in the paper 'Laguerre tessellations and polycrystalline
    microstructures: A fast algorithm for generating grains of given volumes' by D.P. Bourne, P.J.J. Kok, S.M. Roper,
    W.D.T. Spanjer.

    The gradient method that is used is the Barzilai-Borwein method and the convergence criterion regarding the
    closeness of the final generator points to the actual centroids that :arg tol: refers to is described in the
    paper "Fast methods for computing centroidal Laguerre tessellations for prescribed volume fractions with
    applications to microstructure generation of polycrystalline materials" by Jannick Kuhna, Matti Schneiderb,
    Petra Sonnweber-Ribica, Thomas Böhlke.

    All of the arrays "areas", "weights", "points", "target_volumes" follow the same order. This means:
        At index j of "areas" we find the area of the j-th cell in the Laguerre diagram.
        At index j of "weights" we find the weight of the j-th cell in the Laguerre diagram.
        At index j of "points" we find the position of the generator point of the j-th cell in the Laguerre diagram.
        At index j of "target volumes" we find the desired volume of the j-th cell in the Laguerre diagram.
    """
    if gradient_max_iterations is None:
        gradient_max_iterations = weights_init.shape[0] * 10
    if num_cpus is None:
        num_cpus = os.cpu_count()
    return voroplusplus.compute_centroidal_laguerre2d(weights_init, points_init, target_areas, domain, periodicity,
                                                      permute, max_iterations, gradient_max_iterations, tol,
                                                      gradient_tol, disp, fix_num_iterations, solver, respect_bounds,
                                                      num_cpus)


def compute_cross_section(coeffs, offset, points, domain, weights, periodic=None, respect_bounds=True):
    """
    Function for computing a cross section of a Laguerre-Voronoi diagram.

    :arg cells: The output of a call to the function compute_voronoi
    :arg coeffs: The coefficients of the plane used for the cross section, a 1 dimensional numpy array of length 3
    :arg offset: The offset of the plane used for the cross section, a 1 dimensional numpy array of length 3

    :return dict: A python dictionary is returned with the following keys and values:
        "original_indices": A python list containing the indices of the cells which are present in the cross section.
            These indices refer to the index of a cell in the input argument :arg cells:.
        "3d": A python list of numpy arrays of shape (k, 3) where k is the number of corners (vertices) of the
            polygon of that cell (the polygon is the result of performing the cross section). Keep in mind that k is
            not the same for each cell. Each row of each array represents a corner (vertex) of the polygon in 3d
            coordinates. The corners are ordered in a clockwise fashion.
        "2d": A python list of numpy arrays of shape (k, 2) where k is the number of corners (vertices) of the
            polygon of that cell (the polygon is the result of performing the cross section). Keep in mind that k is
            not the same for each cell. Each row of each array represents a corner (vertex) of the polygon in 2d
            coordinates. The corners are ordered in a clockwise fashion. Similar to "3d" but now the obtained corners
            of the polygons are transformed into a 2d plane.
        "areas": A 1 dimensional numpy array containing the areas of the cells which are present in the cross section.

    The plane that is used for computing the cross section is mathematically represented by points (x, y, z) which
    satisfy:
        coeffs[0]*(x - offset[0]) + coeffs[1]*(y - offset[1]) + coeffs[2]*(z - offset[2]) = 0

    The relation to the usual notation should be clear:
        a*(x - x_0) + b*(y - y_0) + c*(z - z_0) = 0

    All of the arrays "original_indices", "3d", "2d" and "areas" follow the same order. This means:
        At index j of "original_indices" we find the index of the j-th cell in the cross section w.r.t. :arg cells:.
        At index j of "3d" we find the corners of the j-th cell in the cross section in 3d space.
        At index j of "2d" we find the corners of the j-th cell in the cross section in 2d space.
        At index j of "areas" we find the area of the j-th cell in the cross section.
    """
    if periodic is None:
        periodic = [False, False, False]
    L1 = domain[0][1] - domain[0][0]
    L2 = domain[1][1] - domain[1][0]
    L3 = domain[2][1] - domain[2][0]
    Lengths = np.array([L1, L2, L3])
    block_size = np.max(Lengths) * 0.2

    if respect_bounds and (periodic[0] or periodic[1] or periodic[2]):
        return compute_cross_section_bounded(coeffs, offset, points, domain, weights, periodic)
    else:
        return voroplusplus.compute_voronoi_section(coeffs, offset, points, domain, dispersion=block_size,
                                                    radii=np.sqrt(weights - np.min(weights)), periodic=periodic)


def compute_cross_section_bounded(coeffs, offset, points, domain, weights, periodic=None):
    """
    Function for computing a cross section of a Laguerre-Voronoi diagram.

    :arg cells: The output of a call to the function compute_voronoi
    :arg coeffs: The coefficients of the plane used for the cross section, a 1 dimensional numpy array of length 3
    :arg offset: The offset of the plane used for the cross section, a 1 dimensional numpy array of length 3

    :return dict: A python dictionary is returned with the following keys and values:
        "original_indices": A python list containing the indices of the cells which are present in the cross section.
            These indices refer to the index of a cell in the input argument :arg cells:.
        "3d": A python list of numpy arrays of shape (k, 3) where k is the number of corners (vertices) of the
            polygon of that cell (the polygon is the result of performing the cross section). Keep in mind that k is
            not the same for each cell. Each row of each array represents a corner (vertex) of the polygon in 3d
            coordinates. The corners are ordered in a clockwise fashion.
        "2d": A python list of numpy arrays of shape (k, 2) where k is the number of corners (vertices) of the
            polygon of that cell (the polygon is the result of performing the cross section). Keep in mind that k is
            not the same for each cell. Each row of each array represents a corner (vertex) of the polygon in 2d
            coordinates. The corners are ordered in a clockwise fashion. Similar to "3d" but now the obtained corners
            of the polygons are transformed into a 2d plane.
        "areas": A 1 dimensional numpy array containing the areas of the cells which are present in the cross section.

    The plane that is used for computing the cross section is mathematically represented by points (x, y, z) which
    satisfy:
        coeffs[0]*(x - offset[0]) + coeffs[1]*(y - offset[1]) + coeffs[2]*(z - offset[2]) = 0

    The relation to the usual notation should be clear:
        a*(x - x_0) + b*(y - y_0) + c*(z - z_0) = 0

    All of the arrays "original_indices", "3d", "2d" and "areas" follow the same order. This means:
        At index j of "original_indices" we find the index of the j-th cell, wich is associated with points[j] and radii[j]
        At index j of "3d" we find the corners of the j-th cell in the cross section in 3d space.
        At index j of "2d" we find the corners of the j-th cell in the cross section in 2d space.
        At index j of "areas" we find the area of the j-th cell in the cross section.
    """
    if periodic is None:
        periodic = [False, False, False]

    L1 = domain[0][1] - domain[0][0]
    L2 = domain[1][1] - domain[1][0]
    L3 = domain[2][1] - domain[2][0]
    Lengths = np.array([L1, L2, L3])
    block_size = np.max(Lengths) * 0.2
    n = len(points)

    section_locations = np.array(voroplusplus.voronoi_num_fragments(points, domain, dispersion=block_size,
                                                                    radii=np.sqrt(weights - np.min(weights)),
                                                                    periodic=periodic), dtype=bool)
    max_intersections = np.max(np.sum(section_locations, axis=1))
    if max_intersections == 0:
        num_iterations = 0
    elif max_intersections == 1:
        num_iterations = 2
    elif max_intersections == 2:
        num_iterations = 4
    else:
        num_iterations = 8

    x_translations = np.zeros((n, num_iterations), dtype=np.int8)
    y_translations = np.zeros((n, num_iterations), dtype=np.int8)
    z_translations = np.zeros((n, num_iterations), dtype=np.int8)

    for i in range(n):
        x_col = section_locations[i, 0] or section_locations[i, 1]
        y_col = section_locations[i, 2] or section_locations[i, 3]
        z_col = section_locations[i, 4] or section_locations[i, 5]
        num_axis = int(x_col) + int(y_col) + int(z_col)
        if num_axis > 0:
            x_loc = [0, 0]
            y_loc = [0, 0]
            z_loc = [0, 0]
            if x_col:
                if section_locations[i, 0]:
                    x_loc = [0, 1]
                else:
                    x_loc = [0, -1]
            if y_col:
                if section_locations[i, 2]:
                    y_loc = [0, 1]
                else:
                    y_loc = [0, -1]
            if z_col:
                if section_locations[i, 4]:
                    z_loc = [0, 1]
                else:
                    z_loc = [0, -1]
            if num_axis == 1:
                if x_col:
                    x_translations[i, 1] = x_loc[1]
                elif y_col:
                    y_translations[i, 1] = y_loc[1]
                elif z_col:
                    z_translations[i, 1] = z_loc[1]
            elif num_axis == 2:
                if x_col and y_col:
                    combinations = list(product(x_loc, y_loc))
                    for j in range(4):
                        x_translations[i, j] = combinations[j][0]
                        y_translations[i, j] = combinations[j][1]
                elif x_col and z_col:
                    combinations = list(product(x_loc, z_loc))
                    for j in range(4):
                        x_translations[i, j] = combinations[j][0]
                        z_translations[i, j] = combinations[j][1]
                elif y_col and z_col:
                    combinations = list(product(y_loc, z_loc))
                    for j in range(4):
                        y_translations[i, j] = combinations[j][0]
                        z_translations[i, j] = combinations[j][1]
            elif num_axis == 3:
                combinations = list(product(x_loc, y_loc, z_loc))
                for j in range(8):
                    x_translations[i, j] = combinations[j][0]
                    y_translations[i, j] = combinations[j][1]
                    z_translations[i, j] = combinations[j][2]

    areas_dict = {}
    polygon_indices = []
    polygons_3d = []
    polygons_2d = []
    first_iteration = True
    for j in range(num_iterations):
        if first_iteration:
            cells_to_compute = [True] * n
            first_iteration = False
            x_outside = x_translations[:, j] != 0
            y_outside = y_translations[:, j] != 0
            z_outside = z_translations[:, j] != 0
            result = voroplusplus.compute_voronoi_section_bounded(coeffs, offset, points, domain, block_size,
                                                                  cells_to_compute, x_outside, y_outside, z_outside,
                                                                  np.sqrt(weights - np.min(weights)), periodic)
            for list_ix, ix in enumerate(result["original_indices"]):
                areas_dict[ix] = result["areas"][list_ix]
            polygon_indices = list(result["original_indices"])
            polygons_2d = result["2d"]
            polygons_3d = result["3d"]
        else:
            x_outside = x_translations[:, j] != 0
            y_outside = y_translations[:, j] != 0
            z_outside = z_translations[:, j] != 0
            cells_to_compute = list(np.logical_or(np.logical_or(x_outside, y_outside), z_outside))
            points_ = np.copy(points)
            points_[:, 0] = points_[:, 0] + x_translations[:, j] * L1
            points_[:, 1] = points_[:, 1] + y_translations[:, j] * L2
            points_[:, 2] = points_[:, 2] + z_translations[:, j] * L3
            res = voroplusplus.compute_voronoi_section_bounded(coeffs, offset, points_, domain, block_size,
                                                               cells_to_compute, x_outside, y_outside, z_outside,
                                                               np.sqrt(weights - np.min(weights)), periodic)
            for list_ix, ix in enumerate(res["original_indices"]):
                if ix in areas_dict:
                    areas_dict[ix] += res["areas"][list_ix]
                else:
                    areas_dict[ix] = res["areas"][list_ix]
            polygon_indices += list(res["original_indices"])
            polygons_2d += res["2d"]
            polygons_3d += res["3d"]
    areas_temp = areas_dict.items()
    areas = [item[1] for item in areas_temp]
    original_indices = [item[0] for item in areas_temp]
    return {"areas": areas, "original_indices": original_indices, "2d": polygons_2d, "3d": polygons_3d,
            "polygons_indices": polygon_indices}


def compute_2d_voronoi(points, limits, weights, periodic=None, z_height=0.5, respect_bounds=True):
    """
    Input arg formats:
    points = list of 2-vectors (lists or compatible class instances) of doubles,
        being the coordinates of the points to Voronoi-tessellate.
    limits = 2-list of 2-lists, specifying the start and end sizes of the box the
        points are in.
    weights = list of python floats as the weights of the points,
        for radical (weighted) tessellation.
    periodic (optional) = 2-list of bools indicating x and y periodicity of
        the system box.
    z_height = a suitable system-size dimension value (if this is particularly different to the
        other system lengths, voro++ will be very inefficient.)
    respect_bounds (optional) = bool, only used when periodic boundary conditions
        are used. If true it ensures that cells are fully within the domain.

    Output format is a list of cells as follows:
        [ # list in same order as original points.
            {
                'volume' : 1.0, # in fact, in 2D, this is the area.
                'vertices' : [[1.0, 2.0], ...], # positions of vertices
                'faces' : [
                    {
                        'vertices' : [7,4], # vertex ids, always 2 for a 2D cell edge.
                        'adjacent_cell' : 34 # *cell* id, negative if a wall
                    }, ...]
                'original' : point[index] # the original instance from args
            },
            ...
        ]
    """
    if periodic is None:
        periodic = [False, False]

    vector_class = voroplusplus.get_constructor(points[0])
    points = [list(p) for p in points]
    points3d = [p[:] + [0.] for p in points]
    limits3d = [l[:] for l in limits] + [[-z_height, +z_height]]
    periodic = periodic + [False]

    py_cells3d = compute_voronoi(points3d, limits3d, weights, periodic, respect_bounds)

    # we assume that each cell is a prism, and so the 2D solution for each cell contains
    # half of the vertices from the 3D solution. We verify this assumption by asserting
    # that each cell has a face adjacent to both -5 and -6, and that they don't share
    # any vertices. We simply take the -5 cell, and ignore the z components.

    py_cells = []
    depth = z_height * 2

    for p3d in py_cells3d:
        if len(p3d) > 0:
            faces_to = [f['adjacent_cell'] for f in p3d['faces']]
            assert(-5 in faces_to and -6 in faces_to)
            indices = [i for i, x in enumerate(faces_to) if x == -5]
            vertices_to_keep = []
            for ix in indices:
                vertices_to_keep += p3d['faces'][ix]['vertices']

            faces2d = []
            for f in p3d['faces']:
                if f['adjacent_cell'] == -5 or f['adjacent_cell'] == -6:
                    continue
                faces2d.append({
                    'adjacent_cell': f['adjacent_cell'],
                    'vertices': [vertices_to_keep.index(vid) for vid in f['vertices'] if vid in vertices_to_keep]
                })

            py_cells.append({
                'faces': faces2d,
                'original': vector_class(p3d['original'][:-1]),
                'vertices': [vector_class(p3d['vertices'][v][:-1]) for v in vertices_to_keep],
                'volume': p3d['volume'] / depth
            })

        else:
            py_cells.append({})

    return py_cells


def plot_2d_helper(laguerre):
    polygons = []
    areas = []
    for cell in laguerre:
        vertex_counts = {}
        if "faces" in cell:
            for face in cell["faces"]:
                for j in range(2):
                    if face["vertices"][j] in vertex_counts:
                        vertex_counts[face["vertices"][j]] += 1
                    else:
                        vertex_counts[face["vertices"][j]] = 1
                if 1 not in set(vertex_counts.values()):
                    polygon_vertices = sorted(list(vertex_counts.keys()))
                    polygons.append(np.array(cell["vertices"])[polygon_vertices])
                    areas.append(cell["volume"])
                    vertex_counts = {}
    return polygons, np.array(areas)
