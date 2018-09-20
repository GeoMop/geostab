from scipy.linalg._flapack import slamch

from Jirina.DC_MitchellPaper_TunnelInversion import Tunnel_ParamMeshTools
from Jirina.DC_MitchellPaper_TunnelInversion import Tunnel_SurveyTools
import matplotlib.pyplot as plt
import numpy
from pymatsolver import Pardiso as Solver
from SimPEG.EM.Static import DC, Utils as DCUtils
from SimPEG import Maps
from SimPEG import Utils
from SimPEG import (DataMisfit, Regularization, Optimization, InvProblem, Directives, Inversion)


class SingleLinearArraySurveyInversion:
    tunnelParams = Tunnel_ParamMeshTools.SimpleTunnelSurveyImplicitParams
    title = 'Single linear tunnel ceiling array'
    singleArray = True
    arrayNum = 1
    surveyType = None

    def __init__(self):
        self.tunnelParams = Tunnel_ParamMeshTools.SimpleTunnelSurveyImplicitParams()
        self.arrayNum = 1
        pass

    def setup(self):
        # meshes
        self._setupMeshes()
        # rock environment conductivity
        self._setupConductivities()
        # survey
        self._setupSurvey()
        # problem
        self._setupDCProblem()
        # mapping, pair, solve, synthetic measurement data
        self._setupSyntheticMeasurementData()

        pass

    def _setupMeshes(self):
        self.mesh = Tunnel_ParamMeshTools.SimpleTunnelMesh(self.tunnelParams)
        self.coreMeshCellIndices, self.coreMesh = self.mesh.getCoreMesh()
        self.activeCellIndices = self.mesh.makeTunnelCellsInactive(self.coreMeshCellIndices)
        pass

    def _setupConductivities(self):
        self.givenModelCond = self.mesh.getGivenModelConductivities()
        self.givenModelOnCoreMesh = self.givenModelCond[self.coreMeshCellIndices]
        pass

    def _setupSurvey(self):
        endPointsPosition = self.tunnelParams.getSingleArrayTunnelCelingEnds()
        #self.survey = DCUtils.gen_DCIPsurvey(endPointsPosition, "dipole-dipole", dim=self.mesh.dim, a = 5, b = 5, n = 8)
        self.allSurveyPoints, self.survey =  Tunnel_SurveyTools.getLinearArraySurvey(endPointsPosition, 5, self.surveyType)
        pass

    def _setupDCProblem(self):
        expMap = Maps.ExpMap(self.mesh)
        inactiveCellIndices = numpy.logical_not(self.activeCellIndices)
        inactiveCellData = self.givenModelCond[inactiveCellIndices]
        mapActive = Maps.InjectActiveCells(mesh=self.mesh, indActive=self.activeCellIndices,
                                           valInactive=inactiveCellData)
        self.mapping = expMap * mapActive
        self.DCproblem = DC.Problem3D_CC(self.mesh, sigmaMap=self.mapping)
        self.DCproblem.pair(self.survey)
        self.DCproblem.Solver = Solver
        pass

    def _setupSyntheticMeasurementData(self):
        self.survey.dpred(self.givenModelCond[self.activeCellIndices])
        self.dobs = self.survey.makeSyntheticData(self.givenModelCond[self.activeCellIndices], std=0.00, force=True)
        pass


    def solve(self):
        # initial values/model
        m0 = numpy.median(-4) * numpy.ones(self.mapping.nP)
        # Data Misfit
        dataMisfit = DataMisfit.l2_DataMisfit(self.survey)
        # Regularization
        regT = Regularization.Simple(self.mesh, indActive=self.activeCellIndices, alpha_s=1e-6, alpha_x=1., alpha_y=1., alpha_z=1.)
        # Optimization Scheme
        opt = Optimization.InexactGaussNewton(maxIter=10)
        # Form the problem
        opt.remember('xc')
        invProb = InvProblem.BaseInvProblem(dataMisfit, regT, opt)
        # Directives for Inversions
        beta = Directives.BetaEstimate_ByEig(beta0_ratio=0.5e+1)
        Target = Directives.TargetMisfit()
        betaSched = Directives.BetaSchedule(coolingFactor=5., coolingRate=2)
        inversion = Inversion.BaseInversion(invProb, directiveList=[beta, Target, betaSched])
        # Run Inversion
        self.invModelOnActiveCells = inversion.run(m0)
        self.invModelOnAllCells = self.givenModelCond * numpy.ones_like(self.givenModelCond)
        self.invModelOnAllCells[self.activeCellIndices] = self.invModelOnActiveCells
        self.invModelOnCoreCells = self.invModelOnAllCells[self.coreMeshCellIndices]
        pass

    def plotSingleArrayAparentResistivity(self):
        if self.singleArray :
            survey2D = DCUtils.convertObs_DC3D_to_2D(self.survey, numpy.ones(self.survey.nSrc), 'Xloc')
            survey2D.dobs = numpy.hstack(self.survey.dobs)
            fig = plt.figure(figsize=(5, 8), num = self.title + ': Apparent resistivity')
            ax1 = plt.subplot(2, 1, 1, aspect="equal")
            clim = [self.mesh.tunnelParams.log_sigmaRock - 1, self.mesh.tunnelParams.log_sigmaDis]
            slice = int(self.coreMesh.nCy / 2)
            dat1 = self.coreMesh.plotSlice(((self.givenModelOnCoreMesh)),
                                           ax=ax1, normal='Y', clim=clim, grid=False, ind=slice)
            ax1.scatter(self.allSurveyPoints[:, 0], self.allSurveyPoints[:, 2], s=5, c='w')

            ax2 = plt.subplot(2, 1, 2, aspect="equal")
            dat2 = DCUtils.plot_pseudoSection(
                survey2D, ax2, survey_type='dipole-dipole', data_type='appConductivity',
                space_type='whole-space', scale="log", data_location=True
            )
            #plt.gca().set_aspect('equal', adjustable='box')
            plt.xlim([self.coreMesh.vectorNx[0], self.coreMesh.vectorNx[-1]])
            plt.ylim([-25, 0])
            ax2.scatter(self.allSurveyPoints[:, 0], self.allSurveyPoints[:, 2]* (-1), s=5, c='k')
        else:
            pass
        pass

    def plotModelDataSliceCenter(self, model, mesh, clim, winTitle):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), num = winTitle)
        ax = Utils.mkvc(ax)
        slice = int(mesh.nCx / 2)
        dat = mesh.plotSlice(((model)), ax=ax[0], normal='X', clim=clim, grid=False, ind=slice)
        ax[0].set_aspect('equal')
        ax[0].set_title('Conductivity, x=' + str(mesh.vectorCCx[slice]))
        rectPoints = self.tunnelParams.getDiscontinuityYZPlainPoints(mesh.vectorCCx[slice])
        ax[0].plot(rectPoints[:, 1], rectPoints[:, 2], 'k--')
        num = round(self.allSurveyPoints.shape[0] / self.arrayNum)
        for ii in range(self.arrayNum):
            ax[0].scatter(self.allSurveyPoints[ii*num, 1], self.allSurveyPoints[ii*num, 2], s=10, c='w')

        slice = int(mesh.nCy / 2)
        mesh.plotSlice(((model)), ax=ax[1], normal='Y', clim=clim, grid=False, ind=slice)
        ax[1].set_aspect('equal')
        ax[1].set_title('Conductivity, y=' + str(mesh.vectorCCy[slice]))
        rectPoints = self.tunnelParams.getDiscontinuityXZPlainPoints(mesh.vectorCCy[slice])
        ax[1].plot(rectPoints[:, 0], rectPoints[:, 2], 'k--')
        ax[1].scatter(self.allSurveyPoints[:, 0], self.allSurveyPoints[:, 2], s=5, c='w')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cb = plt.colorbar(dat[0], ax=cbar_ax)
        cb.set_label('ln conductivity')
        cbar_ax.axis('off')
        pass

    def plot(self):
        # plot initial conductivity
        clim = [self.mesh.tunnelParams.log_sigmaRock - 1, self.mesh.tunnelParams.log_sigmaDis]
        self.plotModelDataSliceCenter(self.givenModelOnCoreMesh, self.coreMesh, clim, self.title + ': True conductivity')
        # plot inversion problem results
        #min = (self.invModelOnActiveCells).min()
        min = (self.invModelOnActiveCells).min()
        max = (self.invModelOnActiveCells).max()
        clim = [min - 0.1, max]
        self.plotModelDataSliceCenter(self.invModelOnCoreCells, self.coreMesh, clim, self.title + ': Inversion results')
        pass

    def setupSolvePlotTheTask(self):
        self.setup()
        self.plotSingleArrayAparentResistivity()
        self.solve()
        self.plot()

class FourLinearArraySurveyInversion(SingleLinearArraySurveyInversion):
    def __init__(self):
        SingleLinearArraySurveyInversion.__init__(self)
        self.linearArray = False
        self.title = "Four linear tunnel wall arrays"
        self.arrayNum = 4
        pass

    def _setupSurvey(self):
        epp = self.tunnelParams.getSingleArrayTunnelCelingEnds()
        p1, s1 = Tunnel_SurveyTools.getLinearArraySurvey(epp, 5, self.surveyType)
        epp = self.tunnelParams.getSingleArrayFloorEnds()
        p2, s2 = Tunnel_SurveyTools.getLinearArraySurvey(epp, 5, self.surveyType)
        epp = self.tunnelParams.getSingleArrayRightSideEnds()
        p3, s3 = Tunnel_SurveyTools.getLinearArraySurvey(epp, 5, self.surveyType)
        epp = self.tunnelParams.getSingleArrayLeftSideEnds()
        p4, s4 = Tunnel_SurveyTools.getLinearArraySurvey(epp, 5, self.surveyType)
        self.allSurveyPoints = numpy.vstack((p1, p2, p3, p4))
        self.survey = DC.Survey(s1.srcList + s2.srcList + s3.srcList + s4.srcList)
        pass

    def plotSingleArrayAparentResistivity(self):
        pass


def _runAsMain():
    print("-- Tunnel_SurveyInversion test code start")
    # single linear array - different distance
    SingleLinearArraySurveyInversion.surveyType = 1
    slaInversion = SingleLinearArraySurveyInversion()
    slaInversion.title = slaInversion.title + ', '+ str(slaInversion.tunnelParams.disTunnelDistance) + ' distance'
    slaInversion.setupSolvePlotTheTask()

    slaInversion = SingleLinearArraySurveyInversion()
    slaInversion.tunnelParams.disTunnelDistance = 5
    slaInversion.tunnelParams.recalculateDependentParams()
    slaInversion.title = slaInversion.title + ', ' + str(slaInversion.tunnelParams.disTunnelDistance) + ' distance'
    slaInversion.setupSolvePlotTheTask()

    slaInversion = SingleLinearArraySurveyInversion()
    slaInversion.tunnelParams.disTunnelDistance = 7
    slaInversion.tunnelParams.recalculateDependentParams()
    slaInversion.title = slaInversion.title + ', ' + str(slaInversion.tunnelParams.disTunnelDistance) + ' distance'
    slaInversion.setupSolvePlotTheTask()

    # four linear array - different distance
    slaInversion = FourLinearArraySurveyInversion()
    slaInversion.title = slaInversion.title + ', ' + str(slaInversion.tunnelParams.disTunnelDistance) + ' distance'
    slaInversion.setupSolvePlotTheTask()

    slaInversion = FourLinearArraySurveyInversion()
    slaInversion.tunnelParams.disTunnelDistance = 5
    slaInversion.tunnelParams.recalculateDependentParams()
    slaInversion.title = slaInversion.title + ', ' + str(slaInversion.tunnelParams.disTunnelDistance) + ' distance'
    slaInversion.setupSolvePlotTheTask()

    slaInversion = FourLinearArraySurveyInversion()
    slaInversion.tunnelParams.disTunnelDistance = 7
    slaInversion.tunnelParams.recalculateDependentParams()
    slaInversion.title = slaInversion.title + ', ' + str(slaInversion.tunnelParams.disTunnelDistance) + ' distance'
    slaInversion.setupSolvePlotTheTask()

    plt.show()
    print("-- Tunnel_SurveyInversion test code stop")


if __name__ == "__main__":
    _runAsMain()