from SimPEG import Mesh
import matplotlib.pyplot as plt
import numpy as np

# init of TensorMesh class

# --------------------------- #
# display values on 2D mesh   #
# --------------------------- #

# 2D mesh init
# given number and size of cell in each direction
hpx = [(2, 10)]
hpy = [(3, 10)]
mesh = Mesh.TensorMesh([hpx, hpy])

# creation of an array/matrix with values
nn = mesh.nC      # number of cells in mesh
#print(nn)
nx = mesh.nCx     # number of cells in x direction
#print(nx)
ny = mesh.nCy     # number of cells in y direction
#print(ny)
a = mesh.gridCC   # array with cell center coordinates
#print('gridCC')
#print(a)
valn = np.zeros(nn)
for i in range(nn) :
    valn[i] = i
valx = np.zeros(nn)
for i in range(nn) :
    valx[i] = a[i, 0]
valy = np.zeros(nn)
for i in range(nn) :
    valy[i] = a[i, 1]

data_v = []
for i in range(nx) :
    radek = []
    for j in range(ny) :
        index = i * nx + j
        x = a[index, 0] - 10
        y = a[index, 1] - 15
        cislo = x * x + y * y
        radek.append(cislo)
    data_v.append(radek)
mat = np.array(data_v)


# mesh and data display
# all in one window
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
mesh.plotGrid(axes[0])
# v must be a numpy array
imn = mesh.plotImage(v=valn, ax=axes[1])
imx = mesh.plotImage(v=valx, ax=axes[2])
imy = mesh.plotImage(v=valy, ax=axes[3])
im_mat = mesh.plotImage(v=mat, ax=axes[4])
plt.colorbar(imn[0], ax=axes[1])
plt.colorbar(imx[0], ax=axes[2])
plt.colorbar(imy[0], ax=axes[3])
plt.colorbar(im_mat[0], ax=axes[4])
axes[0].set_title('Mesh')
axes[1].set_title('Cell order')
axes[2].set_title('Cell X')
axes[3].set_title('Cell Y')
axes[4].set_title('Result')
plt.show()

