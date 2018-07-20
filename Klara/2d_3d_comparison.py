from SimPEG import (
    Mesh, Maps, Utils,
    DataMisfit, Regularization, Optimization,
    InvProblem, Directives, Inversion
)
from SimPEG.EM.Static import DC, Utils as DCUtils
import numpy as np
import matplotlib.pyplot as plt
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver
    
# Reproducible science
np.random.seed(12345)

# Meshes:
# Cells size
csx, csz =  1, 0.5
# Number of core cells in each direction
ncx, ncz = 41, 31
# Number of padding cells to add in each direction
npad = 7
# Vectors of cell lengthts in each direction
hx = [(csx, npad, -1.5), (csx, ncx), (csx, npad, 1.5)]
hz = [(csz, npad, -1.5), (csz, ncz)]
# Create mesh
mesh_2d = Mesh.TensorMesh([hx, hz], x0="CN")
#mesh_2d.x0[1] = mesh_2d.x0[1] + csz / 2.

# Cell sizes
csx, csy, csz = 1., 1., 0.5
# Number of core cells in each direction
ncx, ncy, ncz = 41, 1, 31
# Number of padding cells to add in each direction
npad = 7
# Vectors of cell lengths in each direction with padding
hx = [(csx, npad, -1.5), (csx, ncx), (csx, npad, 1.5)]
hy = [(csy, npad, -1.5), (csy, ncy), (csy, npad, 1.5)]
hz = [(csz, npad, -1.5), (csz, ncz)]
# Create mesh and center it
mesh_3d = Mesh.TensorMesh([hx, hy, hz], x0="CCN")

# Model:
ln_sigback = -5.
mtrue_2 = ln_sigback * np.ones(mesh_2d.nC)
mtrue_3 = ln_sigback * np.ones(mesh_3d.nC)

# Survey:
# Setup a Dipole-Dipole Survey
xmin, xmax = -20., 20.
ymin, ymax = -9.,-9.,
zmin, zmax = mesh_2d.vectorCCy[-1],  mesh_2d.vectorCCy[-1]
endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
survey_2 = DCUtils.gen_DCIPsurvey(endl, "dipole-dipole", dim=mesh_2d.dim,
                                a=10, b=10, n=1, d2flag='2D') 
                                
# Line 1
xmin, xmax = -20., 20.
ymin, ymax = 0.,0.
zmin, zmax = -9., -9.
endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
survey_3 = DCUtils.gen_DCIPsurvey(endl, "dipole-dipole", dim=mesh_3d.dim,
                                  a=10, b=10, n=1) 
                                  
# Active cells:
xmin, xmax = -30., 30
ymin, ymax = -20., 0.
xyzlim = np.r_[[[xmin, xmax], [ymin, ymax]]]
actind2, meshCore2 = Utils.meshutils.ExtractCoreMesh(xyzlim, mesh_2d)
expmap2 = Maps.ExpMap(mesh_2d)
mapactive = Maps.InjectActiveCells(mesh=mesh_2d, indActive=actind2,
                                    valInactive=-6.) 
mapping2 = expmap2 * mapactive
                                    
ymin, ymax = -1., 1.
zmin, zmax = -20., 0.
xyzlim = np.r_[[[xmin, xmax], [ymin, ymax], [zmin, zmax]]]
actind3, meshCore3 = Utils.meshutils.ExtractCoreMesh(xyzlim, mesh_3d)
expmap3 = Maps.ExpMap(mesh_3d)
mapactive = Maps.InjectActiveCells(mesh=mesh_3d, indActive=actind3,
                                    valInactive=-6.)
mapping3 = expmap3 * mapactive  

# Setup forward problem:
problem_2d = DC.Problem3D_CC(mesh_2d, sigmaMap=mapping2)
problem_2d.pair(survey_2)
problem_2d.Solver = Solver
survey_2.dpred(mtrue_2[actind2])
survey_2.makeSyntheticData(mtrue_2[actind2], std=0.05, force=True)

problem_3d = DC.Problem3D_CC(mesh_3d, sigmaMap=mapping3)
problem_3d.pair(survey_3)
problem_3d.Solver = Solver
survey_3.dpred(mtrue_3[actind3])
survey_3.makeSyntheticData(mtrue_3[actind3], std=0.05, force=True)

# Plotting:
# rescaling:
k = survey_3.dobs.sum()/survey_2.dobs.sum()
plt.plot(k*survey_2.dobs,'r*')
plt.plot(survey_3.dobs,'^b')
plt.show()


                                                                                                                                                                                                                                                                                                                                                      