import vorostereology as vs
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d as m3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection


# ######################################## Define/ initialize parameters ##############################################

n = 100  # number of generator points
domain = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]  # unit cube
rng = np.random.default_rng()

# initialize generator points and weights
points = rng.uniform(size=(n, 3), low=0.0, high=1.0)
weights = np.zeros(n)

# ######################################### compute Laguerre diagram ##################################################

# Generate a voronoi diagram
lag = vs.Laguerre3D(points, weights, domain, False)
cells = lag.get_cells()

# ########################################### Compute cross section ###################################################

coeffs = np.array([0.0, -0.5, 1.0])
offset = np.array([0.5, 0.5, 0.7])
cross_section = lag.compute_section(coeffs, offset)

# ############################################## Visualize results ####################################################

# Make a 3d plot of the voronoi diagram with the cross seciton
fig = plt.figure(figsize=(4, 4))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

for cell_idx, cell in enumerate(cells):
    for facet_idx, facet in enumerate(cell['faces']):
        idx = np.array(facet['vertices'])
        polygon = m3.art3d.Poly3DCollection([np.array(cell['vertices'])[idx]])
        polygon.set_edgecolor('k')
        polygon.set_alpha(0.1)
        ax.add_collection3d(polygon)

for cell in cross_section['3d']:
    polygon = m3.art3d.Poly3DCollection([cell])
    polygon.set_color("red")
    polygon.set_edgecolor('k')
    ax.add_collection3d(polygon)

ax.set_xlim3d(domain[0])
ax.set_ylim3d(domain[1])
ax.set_zlim3d(domain[2])
ax.set_box_aspect((1, 1, 1))
ax.set_axis_off()
plt.show(block=False)

# Make a 2d plot of the cross section

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

coll = PolyCollection(cross_section['2d'], facecolors="red", edgecolors='k')
ax2.add_collection(coll)
ax2.axis("equal")
ax2.set_axis_off()
fig2.tight_layout()
plt.show()
