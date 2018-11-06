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


TODO:
- function to display measurement, app res.
- problem with extremely high conductivity in corners, big difference in conductivity -> very bad conditioning
- increasing difference regularization decrease conductivity range, doesn't improve SolverWarning, less smooth conductivity field

- higher resolution and some gap over measurements provides something meaningful
- not sure if the structure is just artifact of measuring tunnel and iron frame in floor, need 3d model

"""

from SimPEG import (
    Mesh, Maps, Utils,
    DataMisfit, Regularization, Optimization,
    InvProblem, Directives, Inversion
)
from SimPEG.EM.Static import DC, Utils as DCUtils
import numpy as np
import matplotlib.pyplot as plt
import src.inversion_tools as inv
import logging
import sys
import pandas as pd


import os
src_dir = os.path.dirname(os.path.abspath(__file__))

#try:
#    from pymatsolver import Pardiso as Solver
#except ImportError:
from SimPEG import SolverLU as Solver

# Reproducible science
#np.random.seed(12345)







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
        x_size, z_size = dim = [ 50, 20 ]
        # Number of core cells in each direction
        #ncx, ncz = 123, 41
        ncx, ncz = nc = [100, 40]
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

        xmin, xmax = -x_size/2., x_size/2
        zmin, zmax = -z_size, 0.
        self.xyzlim = np.r_[[[xmin, xmax], [zmin, zmax]]]

        # Extract submesh (CoreMesh) where the conductivity lives.
        self.actind, self.mesh_core = Utils.meshutils.ExtractCoreMesh(self.xyzlim, self.mesh)


    def set_real_data_survey(self):
        """
        Model of conductivity field.
        Constant backgorund with one more conductive and one less conductive cylinder (full 2D case).
        :return: None
        """

        file_in = os.path.join(src_dir, '..', 'data', 'ert_mereni', 'komora_1','ldp.2dm')

        raw_df = pd.read_csv(file_in, skiprows=18, header=None, sep='\t')
        # remove last empty column
        raw_df = raw_df.dropna(axis=1)
        # name columns
        raw_df.columns = ['ca', 'cb', 'pa', 'pb', 'array', 'I', 'V', 'EP', 'AppRes', 'std']

        # remove wrong readings
        raw_df = raw_df.drop(raw_df[raw_df['V'] < 0].index)
        raw_df['std'] = raw_df['std'] * 0.01     # input error is in percent
        # I [mA], V [mV]
        #measured_df = raw_df.iloc[:, [0,1,2,3,5,6,7,9]]
        #print(measured_df.iloc[:,0])
        #raw_df['idx'] = raw_df['ca']*100 + raw_df['cb']


        def od(a, b):
            return 1.0 / (raw_df[b] - raw_df[a])

        raw_df['k'] = 1.0 / (od('ca', 'pa') - od('pa', 'cb') - od('ca', 'pb') + od('pb', 'cb'))
        raw_df['res'] = 2* np.pi * raw_df['k'] * raw_df['V'] / raw_df['I']
        raw_df['U_norm'] = raw_df['V'] / raw_df['I']

        eps = 0.005     # rounding error of input data
        I = raw_df['I']
        V = raw_df['V']
        # ???
        raw_df['round_rel_err'] = np.abs(2 * (V + I) * eps / ( I**2 - eps**2) * I / V )
        #raw_df['res_min'] = 2 * np.pi * raw_df['k'] * (raw_df['V'] - 0.01) / (raw_df['I'] + 0.01) /raw_df['AppRes']
        #raw_df['res_max'] = 2 * np.pi * raw_df['k'] * (raw_df['V'] + 0.01) / (raw_df['I'] - 0.01) /raw_df['AppRes']
        raw_df['app_to_res'] = raw_df['AppRes'] / raw_df['res']
        print("Measured to computed apparent resistivity: min={} max={}", np.min(raw_df['app_to_res']), np.max(raw_df['app_to_res']))
        # math up to +- 0.5 percent

        # TODO: ? meaning of std and EP
        # TODO: compute and use error of V / I (normalizing to I=1) from known 0.01 precision of the given values
        # (V + a)/(I + b) ~ V/I + a/V - b/I
        # for a=b=e:
        # (V + a)/(I + b) ~ V/I + e (1/V - 1/I)
        #print(raw_df.iloc[127, :])
        raw_df.plot(y=['app_to_res', 'std'])
        plt.show()



        xlim = self.xyzlim[0]
        n_points = max( [np.max(raw_df['ca']), np.max(raw_df['cb']), np.max(raw_df['pa']), np.max(raw_df['pb']) ] ) + 1
        print(n_points)
        self.probe_points = inv.PointSet.line([xlim[0], 0], [xlim[1], 0], n_points)
        #self.survey = survey_factory.compose_1d_survey(self.probe_points, survey_factory.schlumberger_full(n_points))
        self.survey = inv.Survey(self.probe_points)
        self.survey.read_scheme(raw_df, ['ca', 'cb', 'pa', 'pb'])
        self.survey.read_data(raw_df, ['U_norm', 'std'])



    @inv.time_func
    def setup_measurement(self):

        # Manual setup of Dipole-Dipole Survey
        n_points = 20

        self.probe_points = inv.PointSet.line([-15, 0], [15, 0], n_points)
        #self.survey = survey_factory.compose_1d_survey(self.probe_points, survey_factory.schlumberger_full(n_points))
        self.survey = inv.Survey(self.probe_points)
        #self.survey.schlumberger_inv_scheme()
        #self.survey.full_per_cable()
        self.survey.marching_cc_pair()
        # xmin, xmax = -15., 15.
        # ymin, ymax = 0., 0.
        # zmin, zmax = self.mesh.vectorCCy[-1], self.mesh.vectorCCy[-1]
        # endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        # self.survey = DCUtils.gen_DCIPsurvey(endl, "dipole-dipole", dim=self.mesh.dim,
        #                                a=1, b=1, n=n_points, d2flag='2D')

    def setup(self):
        self.setup_geometry()
        self.set_real_data_survey()
        #self.set_syntetic_conductivity()
        #self.setup_measurement()

        self.ln_sigback = np.log(self.survey.median_apparent_conductivity())
        print("ln cond med: ", self.ln_sigback)


        # Setup Problem with exponential mapping and Active cells only in the core mesh
        expmap = Maps.ExpMap(self.mesh)
        mapactive = Maps.InjectActiveCells(mesh=self.mesh, indActive=self.actind,
                                           valInactive=self.ln_sigback)
        self.mapping = expmap * mapactive
        problem = DC.Problem3D_CC(self.mesh, sigmaMap=self.mapping)
        problem.pair(self.survey.simpeg_survey)
        problem.Solver = Solver

        # Compute prediction using the forward model and true conductivity data.
        # survey.dpred(mtrue[actind])
        # Make synthetic data adding a noise to the prediction.
        # In fact prediction is computed again.
        # self.survey.simpeg_survey.makeSyntheticData(self.mtrue[self.actind], std=0.05, force=True)


    @inv.time_func
    def solve(self):
        # Tikhonov Inversion
        ####################

        # Initial model values
        m0 = np.median(self.ln_sigback) * np.ones(self.mapping.nP)
        m0 += np.random.randn(m0.size)

        # Misfit functional
        dmis = DataMisfit.l2_DataMisfit(self.survey.simpeg_survey)
        # Regularization functional
        regT = Regularization.Simple(self.mesh,
                                     alpha_s=10.0,
                                     alpha_x=10.0,
                                     alpha_y=10.0,
                                     alpha_z=10.0, indActive=self.actind)

        # Personal preference for this solver with a Jacobi preconditioner
        opt = Optimization.ProjectedGNCG(maxIter=10, tolX=1,
                                         maxIterCG=30)
        #opt = Optimization.ProjectedGradient(maxIter=100, tolX=1e-2,
        #                                 maxIterLS=20, maxIterCG=30, tolCG=1e-4)

        opt.printers.append(Optimization.IterationPrinters.iterationLS)
        #print(opt.printersLS)

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
        clim = [self.minv.min(), self.minv.max()]

        dat = self.mesh_core.plotImage((self.minv), ax=ax, clim=clim)
        #ax.set_aspect('equal')
        ax.set_title('Inverted Model')
        return dat


    def plot_results(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax = Utils.mkvc(ax)

        dat = self.plot_conductivity(ax[0])
        self.survey.plot_measurements(ax[1])
        self.survey.print_summary()


        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cb = plt.colorbar(dat[0], ax=cbar_ax)
        cb.set_label('ln conductivity')

        cbar_ax.axis('off')

        plt.show()

    def save_inv(self):
        np.save("real_inv_cond", self.minv, allow_pickle=False)

    def load_inv(self):
        self.minv =  np.load("real_inv_cond.npy", allow_pickle=False)


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