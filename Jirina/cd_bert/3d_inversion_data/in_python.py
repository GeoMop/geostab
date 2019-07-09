import pybert as pb
import pygimli as pg


# compute k, err, rhoa
data = pb.DataContainerERT("ldp2.dat")
data.set("k", pg.geometricFactors(data))
data.set("err", pb.Resistivity.estimateError(data, absoluteUError=0.0001, relativeError=0.03))
data.set("rhoa", data("u") / data("i") * data("k"))

# mesh
mesh = pg.load("bukov.bms")

# inversion
res = pb.Resistivity(data)
res.setMesh(mesh)
#res.setMesh(mesh, refine=False)
res.invert()
#res.showResult()
res.saveResult("in_python_res") # ve verzi 2.2.9 spadne, ale vysledky ulozi

#pg.wait()
