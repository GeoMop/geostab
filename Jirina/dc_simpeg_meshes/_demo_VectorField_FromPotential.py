from SimPEG import Mesh
from SimPEG import Utils
import matplotlib.pyplot as plt
import numpy as np

# direct calculation
# calculation of vector field
#
# given potential values
# given conductivity values
#
# vector field calculation
# "source" calculation

# mesh
hx = 2*np.ones(50)
hy = 2*np.ones(30)
mesh = Mesh.TensorMesh([hx, hy])
mesh.plotGrid()
plt.title(s='Mesh')

cellCenters = mesh.gridCC
# potential values
print("Cell numbers")
print(mesh.vnC)
print(mesh.nCx)
print(mesh.nCy)
p = np.zeros(mesh.vnC)
for r in range (mesh.nCx) :
    for s in range (mesh.nCy) :
        index = s * mesh.nCx + r
        x = cellCenters[index, 0]
        y = cellCenters[index, 1]
        p[r, s] = x
print("values of potential in cell centers")
print(p)

# conductivity values
ka = 1
kb = 2
k = np.zeros(mesh.vnC)
for r in range (mesh.nCx) :
    for s in range(mesh.nCy):
        k[r, s] = ka
        sa = mesh.nCy / 4
        sb = 3 * mesh.nCy / 4 - 1
        if s > sa :
            k[r, s] = ka + (s - sa) * (kb - ka) / (sb - sa)
        if s > sb :
            k[r, s] = kb
print("values of conductivity in cell centers")
print(k)

# k, p matrix conversion to get necessary shape
p = Mesh.utils.mkvc(p)
k = Mesh.utils.mkvc(k)
kop = Utils.sdiag(mesh.aveF2CC.T * k)

# operators
grad = mesh.cellGrad
div = mesh.faceDiv

# vectors calculation
v = - kop * grad * p
print("vectors")
print(v)

# sources calculation
q = div * v
print("div(v)")
print(q)

# input values and results images creation
im = mesh.plotImage(p)
plt.colorbar(im[0] )
plt.title("Potential values")

im = mesh.plotImage(k)
plt.colorbar(im[0] )
plt.title("Conductivity values")

im = mesh.plotImage(q)
plt.colorbar(im[0] )
plt.title("Div v")

im = mesh.plotImage(v, 'F', view='vec')
plt.colorbar(im[0] )
plt.title("Vector field")

# plot display
plt.show()