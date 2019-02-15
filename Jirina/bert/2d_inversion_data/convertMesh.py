
from pygimli.meshtools import readGmsh
mesh = readGmsh('oblast.msh', verbose = True)
mesh.save('oblast.bms')
