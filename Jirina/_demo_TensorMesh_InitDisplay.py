from SimPEG import Mesh
import matplotlib.pyplot as plt
import numpy as np

# different init of TensorMesh class
# every mesh displayed in separate window

# --------------------------- #
# 2D mesh - init and display  #
# --------------------------- #

# structured 2D mesh
# number of cells in each direction, implicit range, x-centered, y-negative
me1 = Mesh.TensorMesh([10, 10], x0='CN')
me1.plotGrid()
plt.title(s='Rectangular mesh - one')
plt.show()

# structured 2D mesh
# number of cells and mesh origin
hnum = [10, 10]
hpos = [20, 20]
me2 = Mesh.TensorMesh(hnum, hpos )
me2.plotGrid()
plt.title(s='Rectangular mesh - two')
plt.show()

# structured 2D mesh
# number and size of cell in each direction
hp = [(1, 20)]
me3 = Mesh.TensorMesh([hp, hp])
me3.plotGrid()
plt.title(s='Rectangular mesh - three')
plt.show()

# structured 2D mesh with padding cells
# number and size of cell in each section - used for all the directions
hp = [ (1, 10, -1.25), (1, 20), (1, 10, 1.25)]
hpos = [20, 20]
me4 = Mesh.TensorMesh([hp, hp], x0='CC')
me4.plotGrid()
plt.title(s='Rectangular mesh - four')
plt.show()

# structured 2D mesh - size of every cell defined
# size of cell each cell in each direction defined by values in arrays hx, hy
hx = np.ones(12)
hx[1]=3
hy = np.ones(7)
hy[3] = 0.5
me5 = Mesh.TensorMesh([hx, hy])
me5.plotGrid()
plt.title(s='Rectangular mesh - five')
plt.show()

# --------------------------- #
# 3D mesh - init and display  #
# --------------------------- #

# structured 3D mesh
# size of cell each cell in each direction defined by values in arrays hx, hy
hx = np.ones(12)
hy = np.ones(12)
hz = np.ones(8)
me11 = Mesh.TensorMesh([hx, hy, hz])
me11.plotGrid()
plt.title(s='3D mesh - eleven')
plt.show()

# structured 3D mesh with padding cells
# number and size of cell in each section
hxy = [ (1, 5, -1.5), (1, 10), (1, 5, 1.5)]
hz = [ (2, 2, -1.25), (2, 3), (2, 2, 1.25)]
me12 = Mesh.TensorMesh([hxy, hxy, hz])
me12.plotGrid()
plt.title(s='3D mesh - twelve')
plt.show()
