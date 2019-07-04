
from pygimli.meshtools import readGmsh
mesh = readGmsh('bukov.msh', verbose = True)
mesh.save('bukov.bms')
