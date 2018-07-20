from SimPEG import Mesh
from SimPEG import Utils
import matplotlib.pyplot as plt
import numpy as np
from pymatsolver import Solver as solv

# direct calculation
# calculation of vector field on 3d mesh
#
# given sources
# given conductivity
#
# potential calculation
# vector field calculation

# mesh
hx = np.ones(30)
hy = np.ones(20)
hz = np.ones(10)
mesh = Mesh.TensorMesh([hx, hy, hz])
mesh.plotGrid()
plt.title(s='Mesh')

# sources
q = np.zeros(mesh.vnC)
posx = round(mesh.nCx / 3)
posy = round(mesh.nCy / 2)
posz = round(mesh.nCz / 2)
q[[posx, mesh.nCx-posx], [posy, posy], [posz, posz]]=[-1, 1]
q = Mesh.utils.mkvc(q)

# conductivity
# vodivost konstantni, hodnota 1 v celem objemu
k = np.ones(mesh.vnC)
k = Mesh.utils.mkvc(k)
kop = Utils.sdiag(mesh.aveF2CC.T * k)

# operator, solution
grad = mesh.cellGrad
div = mesh.faceDiv
matA = div * kop * grad
invA = solv(matA)
p = invA * q
v = grad*p

# vector field display
fig, axes = plt.subplots(1, 4, figsize=(22, 4))
im = mesh.plotSlice(v, 'F', ax=axes[0],  normal='X', ind=posx,  view='vec', grid=True, pcolorOpts={'alpha':0.4})
plt.colorbar(im[0], ax=axes[0])
im = mesh.plotSlice(v, 'F', ax=axes[1],  normal='X', ind=mesh.nCx-posx,  view='vec', grid=True, pcolorOpts={'alpha':0.4})
plt.colorbar(im[0], ax=axes[1])
im = mesh.plotSlice(v, 'F', ax=axes[2],  normal='Y',  view='vec', grid=True,  pcolorOpts={'alpha':0.4})
plt.colorbar(im[0], ax=axes[2])
im = mesh.plotSlice(v, 'F', ax=axes[3],  normal='Z',  view='vec', grid=True,  pcolorOpts={'alpha':0.4})
plt.colorbar(im[0], ax=axes[3])

# vector field display - selected slices  to Z normal
fig, axes = plt.subplots(1, 3, figsize=(22, 4))
im = mesh.plotSlice(v, 'F', ax=axes[0],  normal='Z', ind=posz-2,  view='vec', grid=True,  pcolorOpts={'alpha':0.4})
plt.colorbar(im[0], ax=axes[0])
im = mesh.plotSlice(v, 'F', ax=axes[1],  normal='Z', ind=posz,  view='vec', grid=True,  pcolorOpts={'alpha':0.4})
plt.colorbar(im[0], ax=axes[1])
im = mesh.plotSlice(v, 'F', ax=axes[2],  normal='Z', ind=posz+2,  view='vec', grid=True,  pcolorOpts={'alpha':0.4})
plt.colorbar(im[0], ax=axes[2])

# potential display - selected slices
fig, axes = plt.subplots(1, 4, figsize=(22, 4))
im = mesh.plotSlice(p, ax=axes[0], normal='X', ind=posx, grid=True, pcolorOpts={'alpha':0.8})
plt.colorbar(im[0], ax=axes[0])
im = mesh.plotSlice(p, ax=axes[1], normal='X', ind=mesh.nCx-posx, grid=True, pcolorOpts={'alpha':0.8})
plt.colorbar(im[0], ax=axes[1])
im = mesh.plotSlice(p, ax=axes[2], normal='Y', grid=True, pcolorOpts={'alpha':0.8})
plt.colorbar(im[0], ax=axes[2])
im = mesh.plotSlice(p, ax=axes[3], normal='Z', grid=True, pcolorOpts={'alpha':0.8})
plt.colorbar(im[0], ax=axes[3])

# figures display
plt.show()