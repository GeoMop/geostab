from SimPEG import Mesh
from SimPEG import Utils
import matplotlib.pyplot as plt
import numpy as np
from pymatsolver import Solver as solv

# direct calculation
# calculation of vector field
#
# given sources
# given constant conductivity
#
# potential calculation
# vector field calculation

hx = np.ones(60)
hy = np.ones(40)
mesh = Mesh.TensorMesh([hx, hy])
mesh.plotGrid()
plt.title(s='Mesh')

print("Cell numbers")
print(mesh.vnC)
print(mesh.nCx)
print(mesh.nCy)

# sources
pos = round(mesh.nCy / 4)
q = np.zeros(mesh.vnC)

q[pos, pos] = 1
q[[pos, pos, pos+1, pos+1], [pos, pos+1, pos, pos+1]] = [1, 1, 1, 1]
q[[mesh.nCx-pos-1,mesh.nCx-pos-1 ,mesh.nCx-pos-1-1 , mesh.nCx-pos-1-1], [mesh.nCy-pos-1, mesh.nCy-pos-1-1, mesh.nCy-pos-1, mesh.nCy-pos-1-1]] = [-1 ,-1 ,-1 ,-1]
print("sources within the area")
print(q)

# conductivity values
kk = 5
kk2 = kk / 100
kk3 = kk * 100
k = kk * np.ones(mesh.vnC)
posx = round(mesh.nCx / 2)
posy = round(mesh.nCy / 2)
rangex = round(posx / 2)
rangexp = round(rangex /2)
rangey = round(posy / 2)
rangeyp = round(rangey /2)
for i in range (rangex) :
    for j in range(rangey):
        k[posx-rangexp+i, posy-rangeyp+j] = kk2
posx = round(5)
posy = round(mesh.nCy - 5)
rangex = round(mesh.nCx / 4)
rangexp = 0
rangey = round(mesh.nCy / 4)
rangeyp = rangey
for i in range (rangex) :
    for j in range(rangey):
        k[posx-rangexp+i, posy-rangeyp+j] = kk3
print("conductivity")
print(k)

q = Mesh.utils.mkvc(q)
k = Mesh.utils.mkvc(k)
kop = Utils.sdiag(mesh.aveF2CC.T * k)

grad = mesh.cellGrad
div = mesh.faceDiv
matA = div * kop * grad
invA = solv(matA)

p = invA * q
v = kop * grad * p
print("vectors")
print(v)

im = mesh.plotImage(p)
plt.colorbar(im[0] )
plt.title("Potential values")

im = mesh.plotImage(np.log10(k))
plt.colorbar(im[0] )
plt.title("Conductivity values")

im = mesh.plotImage(q)
plt.colorbar(im[0] )
plt.title("Sources")

im = mesh.plotImage(v, 'F', view='vec', pcolorOpts={'alpha':0.8})
plt.colorbar(im[0] )
plt.title("Vector field")

# plot display
plt.show()