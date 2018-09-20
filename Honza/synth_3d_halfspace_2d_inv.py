"""
3D DC inversion of linear Dipole Dipole array
======================================

This is an example for 3D DC Inversion. The model consists of 2 cylinders,
one conductive, the other one resistive compared to the background.

Forward model is 3D, conductivity is considerd also full 3D (regularized by background conductivity).
Variant is 2D conductivity field extrapolated as constant is direction of the Y axis.

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
import inversion_tools as inv
import shapes as shp
import logging
import sys

# try:
#    from pymatsolver import Pardiso as Solver
# except ImportError:
from SimPEG import SolverLU
from SimPEG import SolverBiCG
# Reproducible science
np.random.seed(12345)

problem_name="synth_3d_half_space_2d_inv"

class SynteticInv:
    def __init__(self):

        # Whole mesh.
        self.mesh = None

        # Core mesh for conductivity
        self.mesh_core = None
        # indicate cells of whole mesh that are part of the core mesh
        self.actind = None

    @inv.time_func
    def setup_geometry(self):
        """
        Define geometry and mesh.
        :return: None
        """
        # Mesh dimensions X,Z
        x_size, y_size, z_size = dim = [30, 30, 15]
        # Number of core cells in each direction
        ncx, ncy, ncz = nc = [30, 30, 30]
        csx, csy, csz = step = np.array(dim) / np.array(nc)

        # Number of padding cells to add in each direction
        npad = 12
        # Vectors of cell lengthts in each direction
        hx = [(csx, npad, -1.5), (csx, ncx), (csx, npad, 1.5)]
        hy = [(csy, npad, -1.5), (csy, ncy), (csy, npad, 1.5)]
        hz = [(csz, npad, -1.5), (csz, ncz)]
        # Create mesh
        mesh = Mesh.TensorMesh([hx, hy, hz], x0="CCN")
        # Vertical shift of the mesh by half of the mesh step
        mesh.x0[2] = mesh.x0[2] + csz / 2.
        self.mesh = mesh

        x_bounds = (-x_size / 2, + x_size / 2)
        y_bounds = (-y_size / 2, + y_size / 2)
        z_bounds = (-z_size, 0.)
        self.core_mesh_limits = xyzlim = np.r_[[x_bounds, y_bounds, z_bounds]]

        # Extract submesh (CoreMesh) where the conductivity lives.
        self.actind, self.mesh_core = Utils.meshutils.ExtractCoreMesh(xyzlim, self.mesh)
        mesh_XZ = Mesh.TensorMesh([hx, hz], x0="CN")
        mesh_XZ.x0[1] = mesh_XZ.x0[1] + csz / 2.
        _, self.mesh_core_XZ = Utils.meshutils.ExtractCoreMesh(np.r_[[x_bounds, z_bounds]], mesh_XZ)
        assert self.mesh_core_XZ.nCx == self.mesh_core.nCx
        assert self.mesh_core_XZ.nCy == self.mesh_core.nCz

    def set_syntetic_conductivity(self):
        """
        Model of conductivity field.
        Constant backgorund with one more conductive and one less conductive cylinder (full 2D case).
        :return: None
        """
        # 2-cylinders Model Creation
        ############################

        # Cylinders, conductive on left resistive on  right
        self.cyl_x_pos = 6
        self.cyl_z_pos = -5
        self.cyl_radius = 3
        cyl_y_extent = 100
        self.cylinders = [shp.Cylinder.from_axis((-self.cyl_x_pos, -cyl_y_extent, self.cyl_z_pos),
                                                 (-self.cyl_x_pos, cyl_y_extent, self.cyl_z_pos), self.cyl_radius),
                          shp.Cylinder.from_axis((self.cyl_x_pos, -cyl_y_extent, self.cyl_z_pos),
                                                 (self.cyl_x_pos, cyl_y_extent, self.cyl_z_pos), self.cyl_radius)]

        self.ln_sigback = ln_sigback = -5.
        ln_sigc = -3.
        ln_sigr = -6.

        # Set whole vector to background values
        mtrue = ln_sigback * np.ones(self.mesh_core.nCx*self.mesh_core.nCz)

        # Set cells with center in Conductive sphere
        points = self.mesh_core.gridCC
        y_points = points[:, 1]
        y_abs_min = y_points[np.argmin(np.abs(y_points))]
        xz_points = points[np.where(points[:,1] == y_abs_min)]
        assert len(xz_points) == len(mtrue)
        csph = self.cylinders[0].inside(xz_points)
        mtrue[csph] = ln_sigc

        # Set cells with center in Resistive sphere
        rsph = self.cylinders[1].inside(xz_points)
        mtrue[rsph] = ln_sigr
        self.mtrue = mtrue

    @inv.time_func
    def setup_measurement(self):

        # Manual setup of Dipole-Dipole Survey
        n_points = 20

        self.probe_points = inv.PointSet.line([-15, 0, 0], [15, 0, 0], n_points)
        # self.survey = survey_factory.compose_1d_survey(self.probe_points, survey_factory.schlumberger_full(n_points))
        self.survey = inv.Survey(self.probe_points)
        # self.survey.schlumberger_inv_scheme()
        # self.survey.full_per_cable()
        self.survey.marching_cc_pair()
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
        extrude = Maps.Surject2Dto3D(self.mesh_core, normal='Y')
        mapactive = Maps.InjectActiveCells(mesh=self.mesh, indActive=self.actind,
                                           valInactive=-5.)
        self.mapping = expmap * mapactive * extrude
        assert self.mapping.nP == self.mesh_core.nCx * self.mesh_core.nCz
        problem = DC.Problem3D_CC(self.mesh, sigmaMap=self.mapping)
        problem.pair(self.survey.simpeg_survey)
        problem.Solver = SolverLU
        #problem.Solver = SolverBiCG

        # Compute prediction using the forward model and true conductivity data.
        # survey.dpred(mtrue[actind])
        # Make synthetic data adding a noise to the prediction.
        # In fact prediction is computed again.
        self.survey.simpeg_survey.makeSyntheticData(self.mtrue, std=0.05, force=True)

    @inv.time_func
    def solve(self):
        # Tikhonov Inversion
        ####################

        # Initial model values
        m0 = np.median(self.ln_sigback) * np.ones(self.mapping.nP)


        # Misfit functional
        dmis = DataMisfit.l2_DataMisfit(self.survey.simpeg_survey)
        # Regularization functional
        regT = Regularization.Simple(self.mesh_core_XZ)

        # Personal preference for this solver with a Jacobi preconditioner
        opt = Optimization.ProjectedGNCG(maxIter=10, lower=-10, upper=10,
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

    def plot_conductivity(self, ax):

        probe_points = self.probe_points.cables[0]
        clim = [(self.mtrue[self.actind]).min(), (self.mtrue[self.actind]).max()]

        # dat = self.mesh_core.plotImage(((self.mtrue[self.actind])), ax=ax[0], clim=clim)
        # ax[0].set_title('Ground Truth')
        # ax[0].set_aspect('equal')
        y_slices = [-10, -5, 0, 5, 10]

        axi = ax[0]
        dat = self.mesh_core.plotImage((self.minv), ax=axi, clim=clim, vType='CC')
        axi.set_aspect('equal')
        axi.set_title('Inverted Model')
        axi.plot(probe_points[:, 0], probe_points[:, 1], 'ro')

        # plane = [0, -1, 0, 0]
        # for cyl in self.cylinders:
        #     cyl_points = cyl.contour(100, plane)
        #     for sub_ax in ax[0:2]:
        #         sub_ax.plot(cyl_points[:, 0], cyl_points[:, 1], 'k--')
        return dat

    def plot_slice(self, axis, value, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(10, 10))

        # data plot
        axis_char = 'XYZ'[axis]
        t = (value - self.core_mesh_limits[axis][0]) / (self.core_mesh_limits[axis][1] - self.core_mesh_limits[axis][0])
        axis_idx = int(np.round(t * self.mesh_core.vnC[axis]))
        field = self.minv
        print(field.min(), field.max())
        # color_opt = dict(norm=plt.Normalize(field.min(), field.max()))
        plot = self.mesh_core.plotSlice(field,
                                        ax=ax, normal=axis_char, ind=axis_idx,
                                        clim=(field.min(), field.max()))
        # annotation of slice
        ax.annotate(axis_char + '=' + str(value),
                    xy=(1, 0), xycoords='axes fraction',
                    xytext=(-20, 20), textcoords='offset pixels',
                    horizontalalignment='right',
                    verticalalignment='bottom')
        # measurement points

    def plot_3d_slices(self, x_slices, y_slices, z_slices):
        slice_lists = [x_slices, y_slices, z_slices]
        max_len = max([len(l) for l in slice_lists])
        fig, axes = plt.subplots(3, max_len, figsize=(50, 30))
        for i_axis, s_list in enumerate(slice_lists):
            for i, slice_val in enumerate(s_list):
                self.plot_slice(i_axis, slice_val, ax=axes[i_axis, i])

        fig.savefig(problem_name+".pdf")
        plt.show()

    def plot_results(self):
        # self.survey.print_summary()

        #x_slices = [-self.cyl_x_pos, 0, self.cyl_x_pos]
        #z_slices = [self.cyl_z_pos - self.cyl_radius, self.cyl_z_pos, self.cyl_z_pos + self.cyl_radius]
        #y_slices = [-self.cyl_radius, 0, self.cyl_radius]
        #self.plot_3d_slices(x_slices, y_slices, z_slices)

        # Nx_fig =  np.sqrt( len(y_slices) )
        # Ny_fig = len(y_slices) / Nx_fig

        fig, ax = plt.subplots(1, 1, figsize=(20, 10))

        # dat = self.plot_conductivity(ax)
        # self.survey.plot_measurements(ax[2])

        probe_points = self.probe_points.cables[0]

        clim = [self.minv.min(), self.minv.max()]

        dat = self.mesh_core_XZ.plotImage(self.minv, ax=ax, clim=clim, vType='CC')
        ax.set_title('XZ conductivity inv')
        ax.set_aspect('equal')

        # axi = ax[0]
        # dat = self.mesh_core.plotImage((self.minv), ax=axi, clim=clim, vType='CC')
        # axi.set_aspect('equal')
        # axi.set_title('Inverted Model')
        ax.plot(probe_points[:, 0], probe_points[:, 1], 'ro')

        # plane = [0, -1, 0, 0]
        # for cyl in self.cylinders:
        #     cyl_points = cyl.contour(100, plane)
        #     for sub_ax in ax[0:2]:
        #         sub_ax.plot(cyl_points[:, 0], cyl_points[:, 1], 'k--')
        # return dat
        #
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # cb = plt.colorbar(dat, ax=cbar_ax)
        # cb.set_label('ln conductivity')
        #
        # cbar_ax.axis('off')
        #
        fig.savefig(problem_name+".pdf")
        plt.show()

    def save_inv(self):
        np.save(problem_name, self.minv, allow_pickle=False)

    @inv.time_func
    def load_inv(self):
        self.minv = np.load(problem_name+".npy", allow_pickle=False)


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    inversion = SynteticInv()
    inversion.setup()
    inversion.solve()
    inversion.save_inv()
    inversion.load_inv()
    inversion.plot_results()


if __name__ == "__main__":
    main()