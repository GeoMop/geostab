import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics.traveltime import Refraction

mpl.rcParams['image.cmap'] = 'inferno_r'

#######

# Acquisition parameters
tol = 1E-22
world_width = 30
ttm_length = 160.0
ttm_start = 40.
sensor_spacing = 5
shot_spacing = 20.
points_xpos = np.arange(0, ttm_length + sensor_spacing, sensor_spacing)
sensors_all_xpos = [0., shot_spacing]
for (i, xpos) in zip (range(len(points_xpos)), points_xpos) :
    if (xpos >= ttm_start - tol) and (xpos <= ttm_length - ttm_start + tol):
        sensors_all_xpos.append(xpos)
sensors_all_xpos.append(ttm_length - shot_spacing)
sensors_all_xpos.append(ttm_length)
print(sensors_all_xpos)

sens_id = []
shots_id = [0,1]
for (i, xpos) in zip (range(len(sensors_all_xpos)),sensors_all_xpos) :
    if (xpos >= ttm_start - tol) and (xpos <= ttm_length - ttm_start + tol):
        sens_id.append(i)
    if ((i-2) % 5 == 0) :
        shots_id.append(i)
shots_id.append(len(sensors_all_xpos) -2)
shots_id.append(len(sensors_all_xpos) -1)
print(sens_id)
print(shots_id)

sensors = np.zeros((len(sensors_all_xpos), 2))  # two boreholes
sensors[:, 0] = np.hstack([sensors_all_xpos]) # x
sensors[:, 1] = 0  # y
print()
print("sensors")
print(sensors)

########

# Create forward model and mesh
world = mt.createRectangle(start=[0, 0], end=[ttm_length, -world_width], marker=0)
# c0 = mt.createCircle(pos=(ttm_start + 60.0, -7), radius=7, segments=25, marker=1)
c0 = mt.createRectangle(start=[0, -2], end = [ttm_length, -4], marker=1)
# c1 = mt.createCircle(pos=(ttm_start + 120.0, -10.0), radius=10, segments=25, marker=2)
c1 = mt.createRectangle(start=[0, -4], end = [ttm_length, -world_width], marker=2)
geom = mt.mergePLC([world, c0, c1])

for sen in sensors:
    geom.createNode(sen)

mesh_fwd = mt.createMesh(geom, quality=34, area=1.0)
model = np.array([1000., 2000., 5000.])[mesh_fwd.cellMarkers()]
print("model")
print(model)
pg.show(mesh_fwd, model, label="Velocity (m/s)", nLevs=3, logScale=False, fitView = True)
plt.show()

########

# Create inversion mesh

refinement = 2.0
x = np.arange(0, ttm_length + refinement , refinement )
y = -np.arange(0, world_width   + refinement , refinement )
# print(x)
# print(y)
mesh = pg.createMesh2D(x, y)

ax, _ = pg.show(mesh, hold=True)
ax.plot(sensors[:, 0], sensors[:, 1], "ro")

########

from itertools import product
rays = list(product(shots_id, sens_id))
print()
print(len(rays))
print(rays)
rays2 = []
for ray in rays :
    if (ray[0] == ray[1]):
        pass
    else :
        rays2.append(ray)
rays = rays2
print(len(rays))
print(rays)
rayOld = 0
rays2 = []
print("vypis dvojic dle 'shots'")
for ray in rays :
    if ray[0] == rayOld :
        rays2.append(ray)
    else:
        print(len(rays2))
        print(rays2)
        rayOld = ray[0]
        rays2=[ray]
print(len(rays2))
print(rays2)

# Empty container
scheme = pg.DataContainer()

# Add sensors
for sen in sensors:
    scheme.createSensor(sen)

# Add measurements
rays = np.array(rays)
scheme.resize(len(rays))
scheme.add("s", rays[:, 0])
scheme.add("g", rays[:, 1])
scheme.add("valid", np.ones(len(rays)))
scheme.registerSensorIndex("s")
scheme.registerSensorIndex("g")

########

# Forward simulation

tt = Refraction()
mesh_fwd.createSecondaryNodes(5)
data = tt.simulate(mesh=mesh_fwd, scheme=scheme, slowness=1. / model, noisify='true',
                   noiseLevel=5.0, noiseAbs=1e-3)

########

times = data.get("t")
print("data of simulation")
print(data)
print("travel time array")
print(times)
print("travel time values")
for t in times :
    print(t)

########



########

# Inversion

ttinv = Refraction()
ttinv.setData(data)  # Set previously simulated data
ttinv.setMesh(mesh, secNodes=5)
ttinv.showData()
ttinv.showVA()
plt.show()
#invmodel = ttinv.invert(lam=1100, vtop=3000, vbottom=1000, zWeight=1.0)
invmodel = ttinv.invert(zWeight=0.5)
print("chi^2 = %.2f" % ttinv.inv.getChi2())  # Look at the data fit

########

# Visualization

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 7), sharex=True, sharey=True)
ax1.set_title("True model")
ax2.set_title("Inversion result")

pg.show(mesh_fwd, model, ax=ax1, showMesh=True, label="Velocity (m/s)",
        logScale=False, nLevs=3)

for ax in (ax1, ax2):
    ax.plot(sensors[:, 0], sensors[:, 1], "wo")

ttinv.showResult(ax=ax2)
ttinv.showRayPaths(ax=ax2, color="0.8", alpha=0.3)
fig.tight_layout()

fig, ax = plt.subplots()
ttinv.showCoverage(ax=ax, cMap="Greens")
ttinv.showRayPaths(ax=ax, color="k", alpha=0.3)
ttinv.showResultAndFit()
ax.plot(sensors[:, 0], sensors[:, 1], "ko")

plt.show()