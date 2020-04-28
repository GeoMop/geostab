
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics.traveltime import Refraction

mpl.rcParams['image.cmap'] = 'inferno_r'

class BukovSeismicDataConversion() :
    """
    Conversion obtained data (exported to csv format)
    to the <<DAT>> format
    Suited to the Bukov measurement fix configuration - according to the scheme
    """
    # implicit files location
    # path to the measurement data file
    dataDir = "seis_data_bukov"
    # dataInFilename = "data_traveltime_data.csv"
    dataInFilename = "bukov_values_measured_2.csv"
    dataOutFilename = "bukov_traveltime_data.dat"
    srcdir = os.path.dirname(os.path.abspath(__file__))
    dataInFilepath = os.path.abspath(os.path.join(srcdir, "..", dataDir, dataInFilename))
    dataOutFilepath = os.path.abspath(os.path.join(srcdir, "..", dataDir, dataOutFilename))
    # implicit data separator
    datSeparator = ';'

    def __init__(self, inFilePath=None, outFilePath=None):
        """
        BukovSeismicDataConversion object initialisation
        """
        if inFilePath != None : self.dataInFilepath = inFilePath
        if outFilePath != None: self.dataOutFilepath = outFilePath
        pass

    def readMeassuredData(self):
        with open(self.dataInFilepath, 'r') as fin:
            obsahIn = fin.read()
            pass
        self.radky = obsahIn.split()
        self.radkyDatStr = []
        for radek in self.radky :
            self.radkyDatStr.append(radek.split(self.datSeparator))
            pass
        # for rd in self.radkyDatStr :
        #     print(rd)
        pass

    def generateGeofonPoits(self):
        self.geophonePoints = []
        for i in range(2, len(self.radkyDatStr)) :
            self.geophonePoints.append((self.radkyDatStr[i][0],self.radkyDatStr[i][1] ))
            pass
        for ib in range(10):
            x = 54.
            y = 18. - 2 * ib
            pointData = (self.geophonePoints[ib][0], self.geophonePoints[ib][1], x, y)
            self.geophonePoints[ib] = pointData
        ib = 10
        for x in range(52, 12, -2):
            y = 0.
            pointData = (self.geophonePoints[ib][0], self.geophonePoints[ib][1], x, y)
            self.geophonePoints[ib] = pointData
            ib = ib + 1
        x1 = 12.
        y1 = 1.
        x2 = 1.
        y2 = 22.
        pocet = 13
        dx = (x2 - x1) / (pocet - 1)
        dy = (y2 - y1) / (pocet - 1)
        for i in range(pocet):
            x = x1 + dx * i
            y = y1 + dy * i
            pointData = (self.geophonePoints[ib][0], self.geophonePoints[ib][1], x, y)
            self.geophonePoints[ib] = pointData
            ib = ib + 1
        for point in self.geophonePoints:
            print(point)
        pass

    def writeData(self):
        with open(self.dataOutFilepath, 'w') as fout:
            fout.write("%d # shot/geophone points\n" % (len(self.geophonePoints)))
            fout.write("#x	y\n")
            for point in self.geophonePoints:
                fout.write("%1.4f %1.4f\n" % (point[2], point[3]))
            # shotNumStart = 1
            # shotNumEnd = 10
            # shotNumStart = 11
            # shotNumEnd = 23
            pocetDat = 0
            pocetNul = 0
            # pocet dat
            for i in range(2, len(self.radkyDatStr)) :
                for j in range(2, len(self.radkyDatStr[i])) :
                    # shotNum = self.radkyDatStr[i][0]
                    # if (shotNum >= shotNumStart) and (shotNum <= shotNumEnd) :
                        if float(self.radkyDatStr[i][j]) > 0.0 :
                            pocetDat = pocetDat + 1
                        else :
                            pocetNul = pocetNul + 1
            print(pocetNul, pocetDat)
            shotsId = []
            for metraz in range(2, len(self.radkyDatStr[1])):
                for point in self.geophonePoints:
                    print(metraz, point[1])
                    if self.radkyDatStr[1][metraz] == point[1] :
                        shotsId.append(int(point[0]))
            print(shotsId)
            fout.write("%d # measurements\n" % (pocetDat))
            fout.write("#s\tg\tt\n")
            print(len(shotsId))
            print(len(self.geophonePoints))
            print(self.geophonePoints)
            for shotIndex in range(len(shotsId)) :
                for geophoneIndex in range(len(self.geophonePoints)) :
                    sl = shotIndex + 2
                    ra = geophoneIndex + 2
                    value = float(self.radkyDatStr[ra][sl]) / 1000.
                    if value > 0.0:
                        fout.write("%s %s %1.6f\n" % (self.geophonePoints[shotsId[shotIndex] - 1][0],
                                                        self.geophonePoints[geophoneIndex][0], value))
                    pass
                pass
            coords = []
            for point in self.geophonePoints:
                coords.append([point[2], point[3]])
            print(coords)
        pass

    def doConversion(self):
        if self.dataInFilepath == None:
            raise ValueError('Input file path not specified')
        if self.dataOutFilepath == None:
            raise ValueError('Output file path not specified')
        if not os.path.exists(self.dataInFilepath ):
            raise ValueError('Input file path not exists')
        if not os.path.exists(self.dataInFilepath):
            raise ValueError('Output file path not exists')
        self.readMeassuredData()
        self.generateGeofonPoits()
        self.writeData()

def main() :
    # using default file location - see class data
    dataprep = BukovSeismicDataConversion()
    dataprep.doConversion()
    pass


if __name__ == "__main__" :
    main()