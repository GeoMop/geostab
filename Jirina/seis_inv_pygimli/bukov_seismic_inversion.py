
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics.traveltime import Refraction
import itertools

mpl.rcParams['image.cmap'] = 'inferno_r'

class BukovSeismicDataInversion() :
    # path to the measurement data file
    dataDir = "seis_data_bukov"
    dataFileName = "bukov_traveltime_data.dat"
    srcDir = os.path.dirname(os.path.abspath(__file__))
    dataDirPath = os.path.join(srcDir, "..", dataDir)
    dataFilePath = os.path.abspath(os.path.join(dataDirPath, dataFileName))
    resultBaseName = "bukov_result_all"
    outFilePath = os.path.abspath(os.path.join(dataDirPath, resultBaseName))

    # meshPolygonVertexes= [[0.0, 0.0], [0.0, 24.0], [56.0, 24.0], [56.0, 0.0]]
    # [1.0, 22.0], [54.0, 22.]
    # [-0.375, 24.625], [54.0, 24.625]
    # [-1.75, 27.25], [54.0, 27.25]
    leftTop = [-0.375, 24.625]
    rightTop = [54.0,leftTop[1]]
    meshPolygonVertexes = [[54.0, 18.0], [54.0, 16.0], [54.0, 14.0], [54.0, 12.0], [54.0, 10.0],
                       [54.0, 8.0], [54.0, 6.0], [54.0, 4.0], [54.0, 2.0], [54.0, 0.0],
                       [52, 0.0], [50, 0.0], [48, 0.0], [46, 0.0], [44, 0.0],
                       [42, 0.0], [40, 0.0], [38, 0.0], [36, 0.0], [34, 0.0],
                       [32, 0.0], [30, 0.0], [28, 0.0], [26, 0.0], [24, 0.0],
                       [22, 0.0], [20, 0.0], [18, 0.0], [16, 0.0], [14, 0.0],
                       [12.0, 1.0], [11.0833, 2.75], [10.1667, 4.5], [9.25, 6.25], [8.3333, 8.0],
                       [7.4167, 9.75], [6.5, 11.5], [5.5833, 13.25], [4.6667, 15.0], [3.75, 16.75],
                       [2.8333, 18.5], [1.9167, 20.25], [1.0, 22.0], leftTop, rightTop]
    rightBottom2 = [52.0, 2.0]
    rightTop2 = [52.0, rightTop[1]]
    meshPolygonRight = meshPolygonVertexes[0:11] + [rightBottom2, rightTop2, rightTop]
    leftBottom2 = [16, 2.0]
    meshPolygonBottom = meshPolygonVertexes[10:29] + [leftBottom2, rightBottom2]
    leftTop2 = [leftTop[0] + 2., leftTop[1]]
    meshPolygonLeft = meshPolygonVertexes[28:] + [leftTop2, leftBottom2]
    meshPolygonCenter = [leftBottom2, leftTop2, rightTop2, rightBottom2]
    geophonesCoord = [[54.0, 18.0], [54.0, 16.0], [54.0, 14.0], [54.0, 12.0], [54.0, 10.0],
                       [54.0, 8.0], [54.0, 6.0], [54.0, 4.0], [54.0, 2.0], [54.0, 0.0],
                       [52, 0.0], [50, 0.0], [48, 0.0], [46, 0.0], [44, 0.0],
                       [42, 0.0], [40, 0.0], [38, 0.0], [36, 0.0], [34, 0.0],
                       [32, 0.0], [30, 0.0], [28, 0.0], [26, 0.0], [24, 0.0],
                       [22, 0.0], [20, 0.0], [18, 0.0], [16, 0.0], [14, 0.0],
                       [12.0, 1.0], [11.0833, 2.75], [10.1667, 4.5], [9.25, 6.25], [8.3333, 8.0],
                       [7.4167, 9.75], [6.5, 11.5], [5.5833, 13.25], [4.6667, 15.0], [3.75, 16.75],
                       [2.8333, 18.5], [1.9167, 20.25], [1.0, 22.0]]
    shots1Coord = geophonesCoord[0:10]
    shots2Coord = geophonesCoord[30:43]
    geophonesIds1 = list(range(10, 43))
    geophonesIds2 = list(range(0, 30))
    shotsIds1 = list(range(10))
    shotsIds2 = list(range(30,43))
    regularisation_lam = 10


    # Refraction object
    travelTime = None


    def __init__(self, dataDirPath=None, dataFileName=None, resultBaseName=None):
        """
        BukovSeismicDataInversion object initialisation
        """
        if dataDirPath != None :
            self.dataDirPath = dataDirPath
        if dataFileName != None :
            self.dataFilePath = os.path.abspath(os.path.join(self.dataDirPath, dataFileName))
        if resultBaseName != None :
            self.resultBaseName = resultBaseName
        self.outFilePath = os.path.abspath(os.path.join(self.dataDirPath, resultBaseName))

        pass

    def info(self) :
        print("BukovSeismicDataInversion class info method")
        pass

    def initRefractionObject(self):
        self.travelTime = Refraction()
        pass

    def loadDataFromFile(self):
        self.travelTime.loadData(self.dataFilePath)
        pass

    def prepareMesh(self):
        geom = mt.createPolygon(self.meshPolygonVertexes, isClosed=True, marker=0)
        self.mesh = mt.createMesh(geom, quality = 34.0, area = 0.2, smooth = (1, 10))
        self.startModel = np.array([1. / 2000.])[self.mesh.cellMarkers()]
        print(self.mesh.cellMarkers())
        print("setting of mesh")
        self.travelTime.setMesh(self.mesh)
        pass

    def doInversion(self):
        #self.travelTime.useFMM(True)
        print()
        print("regularisation param.", self.regularisation_lam)
        self.invResult = self.travelTime.invert(zWeight=2.0, lam = self.regularisation_lam, useGradient=False, startModel=self.startModel)
        self.outFilePath = self.outFilePath + '_lam' + "%03d" % (self.regularisation_lam)
        #print(self.invResult)
        pass

    def showResults(self):
        # self.travelTime.showMesh()
        self.travelTime.getOffset()
        print(min(self.invResult), max(self.invResult))
        # self.travelTime.showData()
        # self.travelTime.showVA()
        self.travelTime.showRayPaths()
        self.travelTime.showCoverage()
        #self.travelTime.showResult(useCoverage=False, rays=False, nLevs=6)
        self.travelTime.showResult(useCoverage=False, rays=False, nLevs=9, cMin=3000, cMax=7000, showelectrodes=1)
        self.travelTime.saveFigures(name=self.outFilePath)
        #plt.show()
        pass

    def forwardSimulation(self):
        sensors = np.array(self.geophonesCoord)
        # print("sensor positions", sensors.shape)
        # print(sensors)
        rays1 = list(itertools.product(self.shotsIds1, self.geophonesIds1))
        rays2 = list(itertools.product(self.shotsIds2, self.geophonesIds2))
        rays = rays1 + rays2
        rays = np.array(rays)
        scheme = pg.DataContainer()
        for sen in sensors :
            scheme.createSensor(sen)
        scheme.resize(len(rays))
        scheme.add("s", rays[:, 0])
        scheme.add("g", rays[:, 1])
        scheme.add("valid", np.ones(len(rays)))
        scheme.registerSensorIndex("s")
        scheme.registerSensorIndex("g")
        model = np.array(self.invResult)
        traveltimeSimulation = Refraction()
        data = traveltimeSimulation.simulate(mesh=self.mesh, scheme=scheme, slowness=1. / model, nosify=False)
        traveltimes = data.get('t')
        matrix = np.zeros((43, 23))
        for ray, t in zip(rays, traveltimes) :
            t = t * 1000.
            print(ray, t)
            if ray[0] < 10 :
                matrix[ray[1], ray[0]] = t
            else:
                matrix[ray[1], ray[0] - 20] = t
        # print(matrix)
        filename = self.outFilePath + '-values.csv'
        with open(filename,'w') as fout:
            fout.write(';')
            for i in range(23):
                fout.write(';%d' % (i+1))
            fout.write('\n')
            fout.write(';')
            for i in range(23):
                fout.write(';%d' % (i + 1))
            fout.write('\n')
            for i in range(matrix.shape[0]):
                fout.write('%d;' % (i+1))
                for j in range(matrix.shape[1]):
                    fout.write(';%1.2f' % (matrix[i, j]))
                fout.write('\n')
        print("backward simulation results written into file")
        pass

    def processMeasurementData(self):
        self.initRefractionObject()
        self.loadDataFromFile()
        self.prepareMesh()
        self.doInversion()
        self.showResults()
        self.forwardSimulation()
        pass

    pass # BukovSeismicDataInversion class



def main() :
    print(BukovSeismicDataInversion.geophonesCoord)
    print(BukovSeismicDataInversion.shots1Coord)
    print(BukovSeismicDataInversion.shots2Coord)
    print(BukovSeismicDataInversion.geophonesIds1)
    print(BukovSeismicDataInversion.geophonesIds2)
    print(BukovSeismicDataInversion.shotsIds1)
    print(BukovSeismicDataInversion.shotsIds2)

    dataFileName = 'bukov_traveltime_data.dat'
    resultBaseName = 'bukov_inv_out'
    # BukovSeismicDataInversion.regularisation_lam = 2
    bukovdatainv = BukovSeismicDataInversion(dataFileName=dataFileName, resultBaseName=resultBaseName)
    bukovdatainv.processMeasurementData()

    plt.show()
    pass


if __name__ == "__main__" :
    main()