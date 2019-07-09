import pybert as pb
import pygimli as pg


# compute k, err, rhoa
data = pb.DataContainerERT("lhp2.dat")
data.set("k", pg.geometricFactors(data))
data.set("err", pb.Resistivity.estimateError(data, absoluteUError=0.0001, relativeError=0.03))
data.set("rhoa", data("u") / data("i") * data("k"))

# mesh
mesh = pg.load("oblast.bms")
#mesh = pg.meshtools.createParaMesh2DGrid(data.sensorPositions(), paraDX=0.25, paraDZ=0.25, nLayers=20)

# inversion
res = pb.Resistivity(data)
res.setMesh(mesh)
#res.setMesh(mesh, refine=False)
res.invert()
res.showResult()
#res.saveResult("in_python_res")

pg.wait()
