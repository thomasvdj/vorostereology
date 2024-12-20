import numpy as np
import vorostereology as vs
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d as m3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib as mpl
import time


# ######################################## Define/ initialize parameters ##############################################

n = 5000  # number of cells
rng = np.random.default_rng()
sigma = 0.8
mu = -1*sigma*sigma/2
target_volumes = rng.lognormal(mean=mu, sigma=sigma, size=n)  # take a sample from lognormal distribution
target_volumes = target_volumes/np.sum(target_volumes)

total_volume = np.sum(target_volumes)  # Total volume of the domain (a cube) should be equal to sum of all volumes
side_length = 1
domain = [[0, side_length], [0, side_length], [0, side_length]]

# ######################################### compute Laguerre diagram ##################################################

# Compute a Laguerre diagram with the desired volume distribution
start = time.time()
res, _ = vs.compute_centroidal_laguerre3d(target_volumes, periodic=True, disp=True, solver="DN")
end = time.time()
print("Time elapsed", end-start)

cells = res.get_cells()
volumes = res.get_volumes()

# ########################################### Compute cross section ###################################################

# Take a horizontal cross section at z = side_length / 2
coeffs = np.array([0.0, 0.0, 1.0])
offset = np.array([0.5, 0.5, 0.5])
cross_section = res.compute_section(coeffs, offset)

# ############################################## Visualize results ####################################################

# Make a 3d plot of the Laguerre diagram
cmap = mpl.cm.viridis
norm = mpl.colors.LogNorm(vmin=np.min(volumes), vmax=np.max(volumes))
scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

fig = plt.figure(figsize=(4, 4))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

for cell_idx, cell in enumerate(cells):
    for facet_idx, facet in enumerate(cell['faces']):
        if facet["adjacent_cell"] < 0:
            poly = np.array(cell['vertices'])[np.array(facet['vertices'])]
            in_top = facet["adjacent_cell"] in {-5, -6} and np.min(poly[:, 2] > 0.5*side_length)
            in_front = facet["adjacent_cell"] in {-3, -4} and np.max(poly[:, 1] < 0.5*side_length)
            in_right = facet["adjacent_cell"] in {-1, -2} and np.min(poly[:, 0] > 0.5*side_length)
            if in_top or in_front or in_right:
                polygon = m3.art3d.Poly3DCollection([poly])
                polygon.set_color(scalarMap.to_rgba(volumes[cell_idx]))
                polygon.set_edgecolor('k')
                ax.add_collection3d(polygon)

ax.set_xlim3d(domain[0])
ax.set_ylim3d(domain[1])
ax.set_zlim3d(domain[2])
ax.set_box_aspect((1, 1, 1))
ax.set_axis_off()
plt.show(block=False)

# Make a plot of the cross section
fig2, ax2 = plt.subplots(figsize=(4, 4))
# the original indices of the cells in the cross section can be used to give them the same color as in the 3d plot
colors_2d = scalarMap.to_rgba(volumes[cross_section['polygon_indices']])
coll = PolyCollection(cross_section['2d'], array=None, facecolors=colors_2d, edgecolors='k')
ax2.add_collection(coll)
ax2.axis("equal")
ax2.set_axis_off()
plt.show(block=False)

# Make a histogram plot of the cross section area distribution
plt.figure()
plt.hist(cross_section["areas"], bins=50, density=True, ec='black', linewidth=0.2)
plt.title("Cross section area distribution")
plt.show(block=False)

print("Number of observed areas:", len(cross_section["areas"]))

# Make a histogram plot of the distribution of the weights
plt.figure()
plt.hist(res.weights, bins=50, density=True, ec='black', linewidth=0.2)
plt.title("Distribution of the weights")
plt.show(block=False)

# Make a histogram plot of the distribution of the volumes
plt.figure()
plt.hist(volumes, bins=50, density=True, ec='black', linewidth=0.2)
plt.title("Distribution of the volumes")
plt.show()
