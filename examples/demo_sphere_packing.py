# Author: Thomas van der Jagt
# Demo: Compute largest inscribed spheres in (approximately) centroidal Laguerre diagram

import numpy as np
from scipy.optimize import linprog
import vorostereology as vs
from math import pi
# NOTE: plotting requires packages not part of the dependencies.
# Install via:
# pip install vtk
# pip install mayavi
from mayavi import mlab
from tvtk.api import tvtk


def sphere_packing(laguerre, domain, points, weights, periodicity):
    L1 = domain[0][1] - domain[0][0]
    L2 = domain[1][1] - domain[1][0]
    L3 = domain[2][1] - domain[2][0]
    lengths = np.array([L1, L2, L3])

    periodic = periodicity[0] or periodicity[1] or periodicity[2]
    n = points.shape[0]
    centers = np.zeros((n, 3))
    r = np.zeros(n)
    c = np.zeros(4)
    c[3] = -1
    if periodic:
        bounds = [(domain[0][0] - L1, domain[0][1] + L1), (domain[1][0] - L2, domain[1][1] + L2),
                  (domain[2][0] - L3, domain[2][1] + L3), (0, None)]
    else:
        bounds = [(domain[0][0], domain[0][1]), (domain[1][0], domain[1][1]), (domain[2][0], domain[2][1]), (0, None)]

    for idx, cell in enumerate(laguerre):
        k = len(cell["faces"])
        A = np.zeros((k, 4))
        b = np.zeros(k)
        for face_idx, face in enumerate(cell["faces"]):
            face_vertices = np.array(cell['vertices'])[np.array(face['vertices'])]
            neighbor_idx = face["adjacent_cell"]
            if neighbor_idx >= 0:
                dist = np.linalg.norm(points[idx] - points[neighbor_idx])
                ijk = np.zeros(3, dtype=np.intc)

                if periodic:
                    for i_ in [-1, 0, 1]:
                        for j_ in [-1, 0, 1]:
                            for k_ in [-1, 0, 1]:
                                ijk_ = np.array([i_, j_, k_], dtype=np.intc)
                                temp_dist = np.linalg.norm(points[idx] - points[neighbor_idx] + ijk_ * lengths)
                                if temp_dist < dist:
                                    dist = temp_dist
                                    ijk = ijk_

                neighbor_point = points[neighbor_idx] - ijk * lengths
                A[face_idx, 0:3] = 2 * (neighbor_point - points[idx])
                A[face_idx, 3] = np.linalg.norm(A[face_idx, 0:3])
                b[face_idx] = weights[idx] - weights[neighbor_idx] - np.sum(np.square(points[idx])) + \
                              np.sum(np.square(neighbor_point))
            # We will only end up in this case if we do not have a periodic boundary, and the face of the cell under
            # considerations is caused by the domain
            else:
                a = np.zeros(3, dtype=np.intc)
                if np.allclose(face_vertices[:, 0], domain[0][0]):
                    a[0] = -1
                    b[face_idx] = domain[0][0]
                elif np.allclose(face_vertices[:, 0], domain[0][1]):
                    a[0] = 1
                    b[face_idx] = domain[0][1]
                elif np.allclose(face_vertices[:, 1], domain[1][0]):
                    a[1] = -1
                    b[face_idx] = domain[1][0]
                elif np.allclose(face_vertices[:, 1], domain[1][1]):
                    a[1] = 1
                    b[face_idx] = domain[1][1]
                elif np.allclose(face_vertices[:, 2], domain[2][0]):
                    a[2] = -1
                    b[face_idx] = domain[2][0]
                elif np.allclose(face_vertices[:, 2], domain[2][1]):
                    a[2] = 1
                    b[face_idx] = domain[2][1]
                A[face_idx, 0:3] = a
                A[face_idx, 3] = np.linalg.norm(A[face_idx, 0:3])

        opt_res = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, method="revised simplex")
        if not opt_res.success:
            print("Warning: simplex algorithm returning non-optimal solution for cell:", idx)
        centers[idx] = opt_res.x[0:3]
        r[idx] = opt_res.x[3]
    return centers, r


# ######################################## Define/ initialize parameters ##############################################

n = 100  # number of cells
periodicity = [False, False, False]  # make the domain non-periodic in the x,y,z coordinates
np.random.seed(1)  # make results reproducible with a fixed random seed
sigma = 0.8
mu = -1*sigma*sigma/2
target_volumes = np.random.lognormal(mean=mu, sigma=sigma, size=n)  # take a sample from lognormal distribution

# Total volume of the domain (a cube) should be equal to sum of all volumes
side_length = np.sum(target_volumes) ** (1./3)
domain = [[0, side_length], [0, side_length], [0, side_length]]

points = np.random.uniform(low=0, high=side_length, size=(n, 3))
weights = np.zeros(n)

# ######################################### compute Laguerre diagram ##################################################

# Compute a Laguerre diagram with the desired volume distribution
res = vs.compute_centroidal_laguerre(weights, points, target_volumes, domain, periodicity, disp=True)

cells = vs.compute_voronoi(res["points"], domain, res["weights"], periodicity)
volumes = res["volumes"]

centers, r = sphere_packing(cells, domain, res["points"], res["weights"], periodicity)
sphere_volumes = (4./3)*pi*np.power(r, 3)

print("Packing percentage: ", 100*np.sum(sphere_volumes)/np.sum(res["volumes"]))

# mayavi uses diameter instead of radius
# scale_factor=1 prevents mayavi from applying auto-scaling to the size of the sphere
# a high resolution makes the spheres look more round, instead of looking like polyhedra
mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(1200, 1200))

counter = 0
vertices = []
faces = []
vols = []
for cell_idx, cell in enumerate(cells):
    vols += [volumes[cell_idx]] * len(cell["vertices"])
    vertices += cell["vertices"]
    faces += [[j + counter for j in face["vertices"]] for face in cell["faces"]]
    counter += len(cell["vertices"])

mesh = tvtk.PolyData(points=vertices, polys=faces)
mesh.point_data.scalars = vols
mesh.point_data.scalars.name = 'Volume'

mlab.pipeline.surface(mesh, color=(0, 0, 0), representation="wireframe")
mlab.points3d(centers[:, 0], centers[:, 1], centers[:, 2], 2*r, scale_factor=1, resolution=20, colormap="viridis")
mlab.view(distance=20, azimuth=300, elevation=60)
mlab.show()
