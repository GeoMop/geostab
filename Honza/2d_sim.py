"""
2D DC inversion of Dipole Dipole array
======================================

This is an example for 2D DC Inversion. The model consists of 2 cylinders,
one conductive, the other one resistive compared to the background.

We restrain the inversion to the Core Mesh through the use an Active Cells
mapping that we combine with an exponential mapping to invert
in log conductivity space. Here mapping,  :math:`\\mathcal{M}`,
indicates transformation of our model to a different space:

.. math::
    \\sigma = \\mathcal{M}(\\mathbf{m})

Following example will show you how user can implement a 2D DC inversion.
"""

from SimPEG import (
    Mesh, Maps, Utils,
    DataMisfit, Regularization, Optimization,
    InvProblem, Directives, Inversion
)
from SimPEG.EM.Static import DC, Utils as DCUtils
import numpy as np
import matplotlib.pyplot as plt
import survey_factory

#try:
#    from pymatsolver import Pardiso as Solver
#except ImportError:
from SimPEG import SolverLU as Solver

# Reproducible science
np.random.seed(12345)







class SynteticInv:
    def setup_geometry(self):
        """
        Define geometry and mesh.
        :return: None
        """
        # Mesh dimensions X,Z
        x_size, z_size = dim = [ 30, 15 ]
        # Number of core cells in each direction
        #ncx, ncz = 123, 41
        ncx, ncz = nc = [60, 20]
        csx, csz = step = np.array(dim)/np.array(nc)

        # Number of padding cells to add in each direction
        npad = 12
        # Vectors of cell lengthts in each direction
        hx = [(csx, npad, -1.5), (csx, ncx), (csx, npad, 1.5)]
        hz = [(csz, npad, -1.5), (csz, ncz)]
        # Create mesh
        mesh = Mesh.TensorMesh([hx, hz], x0="CN")
        # Vertical shift of the mesh by half of the mesh step
        mesh.x0[1] = mesh.x0[1] + csz / 2.
        self.mesh = mesh

        xmin, xmax = -x_size/2, +x_size/2
        zmin, zmax = -z_size, 0.
        xyzlim = np.r_[[[xmin, xmax], [zmin, zmax]]]

        # Extract submesh (CoreMesh) where the conductivity lives.
        self.actind, self.meshCore = Utils.meshutils.ExtractCoreMesh(xyzlim, self.mesh)


    def set_syntetic_conductivity(self):
        """
        Model of conductivity field.
        Constant backgorund with one more conductive and one less conductive cylinder (full 2D case).
        :return: None
        """
        # 2-cylinders Model Creation
        ############################

        # Spheres parameters
        # Left, Conductive sphere
        x0, z0, r0 = c0 = [-6., -5., 3.]
        # Right, Resistive sphere
        x1, z1, r1 = c1 = [6., -5., 3.]
        self.cylinders = [c0, c1]

        self.ln_sigback = ln_sigback = -5.
        ln_sigc = -3.
        ln_sigr = -6.

        # Set whole vector to background values
        mtrue = ln_sigback * np.ones(self.mesh.nC)

        # Set cells with center in Conductive sphere
        csph = (np.sqrt((self.mesh.gridCC[:, 1] - z0) **
                        2. + (self.mesh.gridCC[:, 0] - x0) ** 2.)) < r0
        mtrue[csph] = ln_sigc * np.ones_like(mtrue[csph])

        # Set cells with center in Resistive sphere
        rsph = (np.sqrt((self.mesh.gridCC[:, 1] - z1) **
                        2. + (self.mesh.gridCC[:, 0] - x1) ** 2.)) < r1
        mtrue[rsph] = ln_sigr * np.ones_like(mtrue[rsph])

        self.mtrue = Utils.mkvc(mtrue)




    def setup_measurement(self):

        # Manual setup of Dipole-Dipole Survey
        n_points = 20

        self.probe_points = survey_factory.probe_points([-15, 0], [15, 0], n_points)
        #self.survey = survey_factory.compose_1d_survey(self.probe_points, survey_factory.schlumberger_full(n_points))
        self.survey = survey_factory.compose_1d_survey(self.probe_points, survey_factory.schlumberger_inv_scheme(n_points))
        # xmin, xmax = -15., 15.
        # ymin, ymax = 0., 0.
        # zmin, zmax = self.mesh.vectorCCy[-1], self.mesh.vectorCCy[-1]
        # endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        # self.survey = DCUtils.gen_DCIPsurvey(endl, "dipole-dipole", dim=self.mesh.dim,
        #                                a=1, b=1, n=n_points, d2flag='2D')

    def setup(self):
        self.setup_geometry()
        self.set_syntetic_conductivity()
        self.setup_measurement()

        # Setup Problem with exponential mapping and Active cells only in the core mesh
        expmap = Maps.ExpMap(self.mesh)
        mapactive = Maps.InjectActiveCells(mesh=self.mesh, indActive=self.actind,
                                           valInactive=-5.)
        self.mapping = expmap * mapactive
        problem = DC.Problem3D_CC(self.mesh, sigmaMap=self.mapping)
        problem.pair(self.survey)
        problem.Solver = Solver

        # Compute prediction using the forward model and true conductivity data.
        # survey.dpred(mtrue[actind])
        # Make synthetic data adding a noise to the prediction.
        # In fact prediction is computed again.
        self.survey.makeSyntheticData(self.mtrue[self.actind], std=0.05, force=True)

    def solve(self):
        # Tikhonov Inversion
        ####################

        # Initial model values
        m0 = np.median(self.ln_sigback) * np.ones(self.mapping.nP)

        # Misfit functional
        dmis = DataMisfit.l2_DataMisfit(self.survey)
        # Regularization functional
        regT = Regularization.Simple(self.mesh, indActive=self.actind)

        # Personal preference for this solver with a Jacobi preconditioner
        opt = Optimization.ProjectedGNCG(maxIter=20, lower=-10, upper=10,
                                         maxIterLS=20, maxIterCG=30, tolCG=1e-4)

        # Optimization class keeps value of 'xc'. Seems to be solution for the model parameters
        opt.remember('xc')
        invProb = InvProblem.BaseInvProblem(dmis, regT, opt)

        # Options for the inversion algorithm in particular selection of Beta weight for regularization.

        # How to choose initial estimate for beta
        beta = Directives.BetaEstimate_ByEig(beta0_ratio=1.)
        Target = Directives.TargetMisfit()
        # Beta changing algorithm.
        betaSched = Directives.BetaSchedule(coolingFactor=5., coolingRate=2)
        # Change model weights, seems sensitivity of conductivity ?? Not sure.
        updateSensW = Directives.UpdateSensitivityWeights(threshold=1e-3)
        # Use Jacobi preconditioner ( the only available).
        update_Jacobi = Directives.UpdatePreconditioner()

        inv = Inversion.BaseInversion(invProb, directiveList=[beta, Target,
                                                              betaSched, updateSensW,
                                                              update_Jacobi])

        self.minv = inv.run(m0)

        # minv = mtrue[actind]
        # Final Plot
        ############


    @staticmethod
    def getCylinderPoints(xc, yc, r):
        """
        Function to plot cylinder border
        :param xc, yc: Cylinder center.
        :param r: radius
        """
        points = [(r*np.cos(phi)+xc, r*np.sin(phi)+yc) for phi in np.arange(0, 2*np.pi, 0.1)]
        return np.array(points)

        # xLocOrig1 = np.arange(-r, r + r / 10., r / 10.)
        # xLocOrig2 = np.arange(r, -r - r / 10., -r / 10.)
        # # Top half of cylinder
        # zLoc1 = np.sqrt(-xLocOrig1 ** 2. + r ** 2.) + zc
        # # Bottom half of cylinder
        # zLoc2 = -np.sqrt(-xLocOrig2 ** 2. + r ** 2.) + zc
        # # Shift from x = 0 to xc
        # xLoc1 = xLocOrig1 + xc * np.ones_like(xLocOrig1)
        # xLoc2 = xLocOrig2 + xc * np.ones_like(xLocOrig2)
        #
        # topHalf = np.vstack([xLoc1, zLoc1]).T
        # topHalf = topHalf[0:-1, :]
        # bottomHalf = np.vstack([xLoc2, zLoc2]).T
        # bottomHalf = bottomHalf[0:-1, :]
        #
        # cylinderPoints = np.vstack([topHalf, bottomHalf])
        # cylinderPoints = np.vstack([cylinderPoints, topHalf[0, :]])
        # return cylinderPoints


    def plot_conductivity(self, ax):
        clim = [(self.mtrue[self.actind]).min(), (self.mtrue[self.actind]).max()]

        dat = self.meshCore.plotImage(((self.mtrue[self.actind])), ax=ax[0], clim=clim)
        ax[0].set_title('Ground Truth')
        ax[0].set_aspect('equal')

        self.meshCore.plotImage((self.minv), ax=ax[1], clim=clim)
        ax[1].set_aspect('equal')
        ax[1].set_title('Inverted Model')
        ax[1].plot(self.probe_points[:, 0], self.probe_points[:, 1], 'ro')

        for cyl in self.cylinders:
            cyl_points = self.getCylinderPoints(*cyl)
            for sub_ax in ax[0:2]:
                sub_ax.plot(cyl_points[:, 0], cyl_points[:, 1], 'k--')
        return dat

    def plot_survey(self, ax):
        """
        Plot survey scheme to ax Axis.
        """
        survey = self.survey
        for i, cc in enumerate(survey.srcList):
            X, Y = zip(*cc.loc)
            ax.plot(X, np.array(Y) + i, 'ro', ms=1)
            for pp in cc.rxList:
                col = 'bo'
                for loc in pp.locs:
                    for x, y in loc:
                        ax.plot(x, y + i, col, ms=1)
                    col = 'go'
                # ax.plot(X[0], Y[0] + i, 'b^', ms = 1)
                # ax.plot(X[1], Y[1] + i, 'bv', ms=1)


    def plot_results(self):
        fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        ax = Utils.mkvc(ax)

        dat = self.plot_conductivity(ax)
        self.plot_survey(ax[2])


        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cb = plt.colorbar(dat[0], ax=cbar_ax)
        cb.set_label('ln conductivity')

        cbar_ax.axis('off')

        plt.show()


def main():
    inversion = SynteticInv()
    inversion.setup()
    inversion.solve()
    inversion.plot_results()

if __name__ == "__main__":
    main()