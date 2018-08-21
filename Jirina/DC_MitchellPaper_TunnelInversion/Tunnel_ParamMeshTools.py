"""

"""
from SimPEG import Mesh
from SimPEG import Utils
from copy import deepcopy
import numpy


class SimpleTunnelSurveyImplicitParams():
    """
    Implicit parameter value for mesh, core mesh, tunnel, survey
    """
    # tunnel x size - mesh x size
    # tunnel y, z centered
    # tunnel y, z size
    tunnelSizeY = 3
    tunnelSizeZ = 3

    # survey length
    surveyLength = 50.

    # size of core mesh cells in each direction
    cellSizeX, cellSizeY, cellSizeZ = 1., 1., 1.
    # number of core mesh cells in each direction
    cellNumY, cellNumZ = 41, 41
    # number of padding cells in all directions
    padNum = 6
    # factor for padding cells extension in all directions
    padFactor = 2.0

    # discontinuity size
    disSizeX, disSizeY, disSizeZ = 5., 5., 5.
    disTunnelDistance = 3.

    # environment conductivity values - log mapping
    log_sigmaTunnel = -8
    log_sigmaRock = -4
    log_sigmaDis = -2

    def __init__(self):
        """
        Instance initialisation
        """
        self.recalculateDependentParams()

    def recalculateDependentParams(self):
        """
        Setting values of dependent parameters
        :return: None
        """
        self._recalculateMeshDependentParams()
        self._recalculateCoreMeshDependentParams()
        self._recalculateDicontinuityDependentParams()
        pass

    def _recalculateMeshDependentParams(self):
        """
        Setting values of dependent parameters of the master mesh
        :return: None
        """
        self.cellNumX = round(self.surveyLength / self.cellSizeX) + 2 * round(self.surveyLength / 10) + 1
        xSize = (2 * (self.cellSizeX * self.padFactor) * (self.padFactor ** (self.padNum) - 1) / (self.padFactor - 1)) + \
                self.cellSizeX * self.cellNumX

        # origin of x axis at the beginning of the survey
        self.meshXOrigin = - xSize / 2 + self.surveyLength / 2
        # mesh parameters
        self.hx = [(self.cellSizeX, self.padNum, -self.padFactor),
              (self.cellSizeX, self.cellNumX),
              (self.cellSizeX, self.padNum, self.padFactor)]
        self.hy = [(self.cellSizeY, self.padNum, -self.padFactor),
              (self.cellSizeY, self.cellNumY),
              (self.cellSizeY, self.padNum, self.padFactor)]
        self.hz = [(self.cellSizeZ, self.padNum, -self.padFactor),
              (self.cellSizeZ, self.cellNumZ),
              (self.cellSizeZ, self.padNum, self.padFactor)]
        pass

    def _recalculateCoreMeshDependentParams(self):
        """
        Setting values of dependent parameters of the core mesh
        :return: None
        """
        # core mesh extent
        coreXSize = self.cellSizeX * self.cellNumX
        xlap = (coreXSize - self.surveyLength) / 2.
        self.xmin, self.xmax = -xlap, self.surveyLength + xlap
        coreYHalfSize = (self.cellSizeY * self.cellNumY) / 2. + self.padFactor / 4.
        self.ymin, self.ymax = -coreYHalfSize, coreYHalfSize
        coreZHalfSize = (self.cellSizeZ * self.cellNumZ) / 2. + self.padFactor / 4.
        self.zmin, self.zmax = -coreZHalfSize, coreZHalfSize
        pass

    def _recalculateDicontinuityDependentParams(self):
        self.disCenterX = self.surveyLength / 2.
        self.disCenterY = 0
        self.disCenterZ = self.disTunnelDistance + self.tunnelSizeZ / 2. + self.disSizeZ / 2.

    def _recalculateSurveyDependentParams(self):
        """
        Setting values of dependent parameters of the survey
        :return:
        """
        pass

    def getSingleArrayEnds(self):
        """
        Provides an array with survey end points position
        Survey - on the ceiling of the tunnel (in the center of the ceiling)
        :return: numpy array
        """
        xmin, xmax = 0., self.surveyLength
        ymin, ymax = 0., 0.
        zmin, zmax = self.tunnelSizeZ / 2.0 + self.cellSizeZ / 2., self.tunnelSizeZ / 2.0 + self.cellSizeZ / 2.
        #zmin, zmax = self.tunnelSizeZ / 2., self.tunnelSizeZ / 2.0
        return numpy.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        pass

    def getSingleArrayTunnelCelingEnds(self):
        """
        Provides an array with survey end points position
        Survey - on the ceiling of the tunnel (in the center of the ceiling)
        :return: numpy array
        """
        xmin, xmax = 0., self.surveyLength
        ymin, ymax = 0., 0.
        #zmin, zmax = self.tunnelSizeZ / 2.0 + self.cellSizeZ / 10., self.tunnelSizeZ / 2.0 + self.cellSizeZ / 10.
        zmin, zmax = self.tunnelSizeZ / 2., self.tunnelSizeZ / 2.0
        return numpy.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        pass

    def getSingleArrayLeftSideEnds(self):
        """
        Provides an array with survey end points position
        Survey - on the left side of the tunnel (in the center of the wall)
        :return: numpy array
        """
        xmin, xmax = 0., self.surveyLength
        ymin, ymax = self.tunnelSizeY /2., self.tunnelSizeY /2.
        zmin, zmax = 0., 0.
        return numpy.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        pass

    def getSingleArrayRightSideEnds(self):
        """
        Provides an array with survey end points position
        Survey - on the right side of the tunnel (in the center of the wall)
        :return: numpy array
        """
        xmin, xmax = 0., self.surveyLength
        ymin, ymax = -self.tunnelSizeY /2., -self.tunnelSizeY /2.
        zmin, zmax = 0., 0.
        return numpy.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        pass

    def getSingleArrayFloorEnds(self):
        """
        Provides an array with survey end points position
        Survey - on the floor of the tunnel (in the center of the floor)
        :return: numpy array
        """
        xmin, xmax = 0., self.surveyLength
        ymin, ymax = 0., 0.
        zmin, zmax = -self.tunnelSizeZ / 2., -self.tunnelSizeZ / 2.
        return numpy.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        pass

    def getDiscontinuityXYPlainPoints(self, z):
        dzs2 = self.disSizeZ / 2.
        if (z >= self.disCenterZ - dzs2) and (z <= self.disCenterZ + dzs2):
            dxs2 = self.disSizeX / 2.
            px = numpy.array([self.disCenterX - dxs2, self.disCenterX - dxs2,
                              self.disCenterX + dxs2, self.disCenterX + dxs2, self.disCenterX - dxs2])
            dys2 = self.disSizeY / 2.
            py = numpy.array([self.disCenterY + dys2, self.disCenterY - dys2,
                              self.disCenterY - dys2, self.disCenterY + dys2, self.disCenterY + dys2])
            pz = numpy.array([z, z, z, z, z])
            points = numpy.vstack([px, py, pz]).T
            return points
        else:
            return numpy.array([])
        pass

    def getDiscontinuityXZPlainPoints(self, y):
        dys2 = self.disSizeY / 2.
        if (y >= self.disCenterY - dys2) and (y <= self.disCenterY + dys2):
            dxs2 = self.disSizeX / 2.
            px = numpy.array([self.disCenterX - dxs2, self.disCenterX - dxs2,
                              self.disCenterX + dxs2, self.disCenterX + dxs2, self.disCenterX - dxs2])
            py = numpy.array([y, y, y, y, y])
            dzs2 = self.disSizeZ / 2.
            pz = numpy.array([self.disCenterZ + dzs2, self.disCenterZ - dzs2,
                              self.disCenterZ - dzs2, self.disCenterZ + dzs2, self.disCenterZ + dzs2])
            points = numpy.vstack([px, py, pz]).T
            return points
        else:
            return numpy.array([])
        pass

    def getDiscontinuityYZPlainPoints(self, x):
        dxs2 = self.disSizeX / 2.
        if (x >= self.disCenterX - dxs2) and (x <= self.disCenterX + dxs2) :
            px = numpy.array([x, x, x, x, x])
            dys2 = self.disSizeY / 2.
            py = numpy.array([self.disCenterY - dys2, self.disCenterY - dys2,
                              self.disCenterY + dys2, self.disCenterY + dys2, self.disCenterY - dys2])
            dzs2 = self.disSizeZ / 2.
            pz = numpy.array([self.disCenterZ + dzs2, self.disCenterZ - dzs2,
                              self.disCenterZ - dzs2, self.disCenterZ + dzs2, self.disCenterZ + dzs2])
            points = numpy.vstack([px, py, pz]).T
            return points
        else:
            return numpy.array([])
        pass

class SimpleTunnelMesh(Mesh.TensorMesh):
    """
    SimpleTunnelMesh represents tunnel and its surroundings.
    """

    tunnelParams : SimpleTunnelSurveyImplicitParams

    def __init__(self, params) :
        """
        SimpleTunnelMesh instance initialisation
        :param params:
        """
        # parameters check
        assert type(params) in [SimpleTunnelSurveyImplicitParams], 'params must be a SimpleTunnelParams, not {}'.format(type(params))
        self.tunnelParams = deepcopy(params)
        xOrigin = self.tunnelParams.meshXOrigin
        hx, hy, hz = self.tunnelParams.hx, self.tunnelParams.hy, self.tunnelParams.hz
        Mesh.TensorMesh.__init__(self, [hx, hy, hz], x0=[xOrigin, 'C', 'C'])
        pass

    def getCoreMesh(self) :
        """
        Extracts and provides core mesh
        :return: indices of core mesh, core mesh
        """
        xyzlim = numpy.array([[self.tunnelParams.xmin, self.tunnelParams.xmax],
                              [self.tunnelParams.ymin, self.tunnelParams.ymax],
                              [self.tunnelParams.zmin, self.tunnelParams.zmax]])
        actind, coreMesh = Utils.meshutils.ExtractCoreMesh(xyzlim, self)
        return actind, coreMesh

    def getGivenModelConductivities(self):
        """
        Creates a numpy array with given conductivity values
        :return: numpy array
        """
        # rock background conductivity
        givenModel = self.tunnelParams.log_sigmaRock * numpy.ones(self.nC)
        # tunnel conductivity
        tunnelCellsIndices = (abs(self.gridCC[:, 1]) <= self.tunnelParams.tunnelSizeY / 2.) & \
                             (abs(self.gridCC[:, 2]) <= self.tunnelParams.tunnelSizeZ / 2.)
        # discontinuity - conductive blok - conductivity
        givenModel[tunnelCellsIndices] = self.tunnelParams.log_sigmaTunnel * numpy.ones_like(givenModel[tunnelCellsIndices])
        disCellsIndices = (abs(self.gridCC[:, 0] - self.tunnelParams.disCenterX) <= self.tunnelParams.disSizeX / 2.) & \
                          (abs(self.gridCC[:, 1] - self.tunnelParams.disCenterY) <= self.tunnelParams.disSizeY / 2.) & \
                          (abs(self.gridCC[:, 2] - self.tunnelParams.disCenterZ) <= self.tunnelParams.disSizeZ / 2.)
        givenModel[disCellsIndices] = self.tunnelParams.log_sigmaDis * numpy.ones_like(givenModel[disCellsIndices])
        return givenModel
        pass

    def makeTunnelCellsInactive(self, currentActiveCells):
        """
        Marks tunnel cells False in the currentActiveCells structure
        :param currentActiveCells:
        :return: numpy array
        """
        tunnelIndices = ((abs(self.gridCC[:, 1]) < self.tunnelParams.tunnelSizeY / 2.) &
                   (abs(self.gridCC[:, 2]) < self.tunnelParams.tunnelSizeZ / 2.))
        tunnelFalse = True * numpy.ones_like(currentActiveCells)
        tunnelFalse = numpy.full(currentActiveCells.shape, True)
        tunnelFalse[tunnelIndices] = False & numpy.ones_like(tunnelFalse[tunnelIndices])
        currentActiveCells = currentActiveCells & tunnelFalse
        return currentActiveCells
        pass



# testing routines
def _runAsMain():
    print("-- Tunnel_ParamMeshTools test code start")
    # implicit parameters
    tunnelParams = SimpleTunnelSurveyImplicitParams()
    tunnelMesh = SimpleTunnelMesh(tunnelParams)
    coreMeshIndices, tunnelCoreMesh = tunnelMesh.getCoreMesh()
    #print(help(tunnelMesh))
    import matplotlib.pyplot as plt
    # tunnelMesh.plotGrid()
    tunnelCoreMesh.plotGrid()
    #print(tunnelCoreMesh.gridCC)

    # changed parameters
    tunnelParams = SimpleTunnelSurveyImplicitParams()
    tunnelParams.cellNumY, tunnelParams.cellNumZ = 11, 11
    tunnelParams.padNum = 2
    tunnelParams.surveyLength = 10
    tunnelParams.recalculateDependentParams()
    tunnelMesh = SimpleTunnelMesh(tunnelParams)
    coreMeshIndices, tunnelCoreMesh = tunnelMesh.getCoreMesh()
    # tunnelMesh.plotGrid()
    # tunnelCoreMesh.plotGrid()

    model = tunnelMesh.getGivenModelConductivities()
    print(tunnelParams.cellNumX, tunnelParams.cellNumY, tunnelParams.cellNumZ)
    print(tunnelMesh.nC)
    print(model.shape)
    print(tunnelCoreMesh.nC)
    print(model[coreMeshIndices].shape)
    activeCellIndices = tunnelMesh.makeTunnelCellsInactive(coreMeshIndices)
    print(model[activeCellIndices].shape)

    print("Discontinuity rectangle points")
    print(tunnelParams.getDiscontinuityYZPlainPoints(tunnelParams.disCenterX))
    print(tunnelParams.getDiscontinuityXYPlainPoints(tunnelParams.disCenterZ))
    print(tunnelParams.getDiscontinuityXZPlainPoints(tunnelParams.disCenterY))

    # clim = [(model[coreMeshIndices]).min(), (model[coreMeshIndices]).max()]
    # clim = [-8., -2.]
    # fig, ax = plt.subplots(3, 4, figsize=(16, 8))
    # ax = Utils.mkvc(ax)
    # for ii in range(12):
    #     sl_ind = ii + int((tunnelMesh.tunnelParams.cellNumX - 12) / 2) #int(tunnelCoreMesh.nCx)
    #     ax_ind = ii
    #     tunnelCoreMesh.plotSlice(((model[coreMeshIndices])), ax=ax[ax_ind], normal='X', clim=clim, grid=True, ind=sl_ind)
    #     ax[ax_ind].set_aspect('equal')
    #     ax[ax_ind].set_title('Conductivity, x=' + str(tunnelCoreMesh.vectorCCx[sl_ind]))
    #
    # fig, ax = plt.subplots(3, 4, figsize=(16, 8))
    # ax = Utils.mkvc(ax)
    # for ii in range(12):
    #     sl_ind = ii + int((tunnelMesh.tunnelParams.cellNumY - 12) / 2)  # int(tunnelCoreMesh.nCx)
    #     ax_ind = ii
    #     tunnelCoreMesh.plotSlice(((model[coreMeshIndices])), ax=ax[ax_ind], normal='Y', clim=clim, grid=True,
    #                              ind=sl_ind)
    #     ax[ax_ind].set_aspect('equal')
    #     ax[ax_ind].set_title('Conductivity, y=' + str(tunnelCoreMesh.vectorCCy[sl_ind]))
    # # given model data - ok

    plt.show()
    print("-- Tunnel_ParamMeshTools test code stop")


if __name__ == "__main__":
    _runAsMain()