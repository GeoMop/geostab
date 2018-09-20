from SimPEG.EM.Static import DC
import numpy as np
import bisect
import time
from functools import wraps
import logging

"""
TODO:
- check that produced SimPEG Survey have no performance drawback, compare tim if we generate Survey by native tools or 
  by our self. Higher time of the full scheme is due to 10 times higher number of Tx pairs (16 vs. 163)
  
- Optimal scheme for 1d cable, 2d forward model. Select measurements with maximal sensitivity to the changes in conductivity parameters,
  i.e.  derivative of   dP / (dC =  1)  by a parameter of a conductivity model.
  Algo:
  - for every conductivity parameter C
    - for every Tx pair (Ca, Cb) compute derivative of potential field by C
        - find Rx electrode with minimal sensitivity set to be Pa
        - find given number of electrodes with maximal sensitivity to be Pb
        - set quality of (Ca, Cb) pair to average of sensitivity of Pb - Pa, for selected Pb points
    - sort (Ca, Cb) pairs according to the quality metric
    - use first N pairs to perform inversion for syntetic test with C parameter set to 1 and other zero
    - increase N until we get different inversion, should by also close to best, including more measurements should
      decrease inversion quality   
"""
class PointSet:

    @staticmethod
    def line(start, end, n_points):
        """
        Make single cable PointSet.
        :param start: np.array [x,y,z] of the first point.
        :param end: np.array [x,y,z] of the last point.
        :param n_points: Number of points on line.
        :return: np.array of the points, shape: (n_points, 3)
        """
        ps = PointSet()
        ps.add_cable(np.array([np.linspace(start[dim], end[dim], n_points) for dim in range(len(start))]).T)
        return ps

    def __init__(self):
        self.cables = []
        self.cable_start = []

    def add_cable(self, poly_line, n_points=None):
        """
        Add a numbered cable located at given polyline.
        :param poly_line: List of XYZ points or np.array of shape (N, 3).
        :param n_points: Number of equidistant points generated on the polyline.
                        None - use polyline points.
        :return: Index of the cable.
        """
        poly_line = np.array(poly_line)
        if n_points is None:
            points = poly_line
        else:
            segments = poly_line[1:, :] - poly_line[:-1, :]
            seg_len = np.linalg.norm(segments, axis=1)
            cum_seg_len = np.cumsum(seg_len)
            length = np.sum(seg_len)
            t_list = np.linspace(0, length, n_points)
            end_pt = 1
            points = []
            for t in t_list:
                while t > cum_seg_len[end_pt]:
                    end_pt += 1
                ta = cum_seg_len[end_pt-1]
                tb = cum_seg_len[end_pt]
                tt = (t - ta) / (tb - ta)   # Map segment parameter to (0,1)
                loc = poly_line[end_pt] * tt + poly_line[end_pt-1] * (1 - tt)
                points.append(loc)

        self._append_cable(points)
        return len(self.cables) - 1

    def _append_cable(self, points):
        if self.cables:
            start = self.cable_start[-1] + len(self.cables[-1])
        else:
            start = 0
        self.cables.append(points)
        self.cable_start.append(start)

    @property
    def all_points(self):
        # np.array of all points in global index order.
        return np.concatenate(self.cables)

    def idx(self, cable_idx):
        """
        Return global index of electrode, for given cable_index pair.
        :param cable_idx: (cable_idx, idx_on_cable)
        :return: global idx.
        """
        i_cable, i = cable_idx
        return self.cable_start[i_cable] + i

    def cable_idx(self, global_idx):
        """
        :param global_idx:
        :return: cable_idx = (i_cable, i_on_cable)
        """
        i_cable = bisect.bisect_right(self.cable_start, global_idx) - 1
        i = global_idx - self.cable_start[i_cable]
        return (i_cable, i)



class Survey:
    """
    Simplified interface to SimPeg survey.
    Measurements are defined as pair of:
    - electrodes spatial position
    - list of measurements as quadruplets (CA, CB, PA, PB),
      that is pair of current (transmitter Tx) electrodes CA, CB
      and pair of potential (recievers Rx) electrodes PA, PB.
    """

    def __init__(self, point_set):
        # PointSet instance
        self.points = point_set
        self.measurements = []
        self.obs_values = None
        self.obs_stds = None
        self._simpeg_survey = None
        self._values = None


    def clear(self):
        """
        Remove all measurements.
        """
        self.measurements = []

    def _append(self, m):
        """
        Append single measurement
        :param m: [ca, cb, pa, pb]
        """
        self._simpeg_survey = None
        self.measurements.append(tuple(m))

    def _for_all_cables(self, fn, cables):
        if cables is None:
            cables = self.points.cables
        else:
            cables = self.points.cables[cables]

        for cable in cables:
            fn(cable)

    def read_scheme(self, df, el_cols):
        """
        Read measurement points scheme.
        :param df: Pandas data frame.
        :param el_cols: Names of columns for electrodes [ ca, cb, pa, pb ].
        """
        el_mat = [ df[col] for col in el_cols ]
        for row  in np.array(el_mat).T:
            self._append(row)

    def read_data(self, df, data_cols):
        """
        Read data for measurements.
        :param df: Pandas data frame.
        :param el_cols: Names of columns for electrodes [ value, std ].
        """
        self._values = np.array(df[data_cols[0]])
        self._errors = np.array(df[data_cols[1]])


    def schlumberger_scheme(self, cables=None):
        """
        Create measurements using Schlumberger scheme on every
        one of given cables.

        Schlumberger scheme: CA - n - PA - 1 - PB - n - CB
        For every possible value of 'n; we move this scheme along the cable.

        :param cables: Indices of cables, None for all.
        """
        self._for_all_cables(self._schlumberger_scheme, cables)

    def _schlumberger_scheme(self, cable):
        n_points = len(cable)
        for i_pa in range(1, n_points - 1):
            i_pb = i_pa + 1
            min_ica = max(0, i_pa + i_pb - n_points + 1)
            for i_ca in range(min_ica, i_pa):
                i_cb = i_pb + (i_pa - i_ca)
                self._append([i_ca, i_cb, i_pa, i_pb])


    def full_per_cable(self, cables=None):
        """
        All positions for (CA, CB) pair. All (PA, PB) pairs with separation 1.
        scheme: CA - n  - CB   X  PA - 1 - PB
        :param cables: Indices of cables, None for all.
        """
        self._for_all_cables(self._full_per_cable, cables)

    def _full_per_cable(self, cable):
        n_points = len(cable)
        for i_ca in range(0, n_points - 4):
            for i_cb in range(i_ca + 3, n_points):
                for i_pa in range(i_ca + 1, i_cb - 1):
                    i_pb = i_pa + 1
                    self._append([i_ca, i_cb, i_pa, i_pb])


    def schlumberger_inv_scheme(self, cables=None):
        """
        Create measurements using "inversed Schlumberger scheme" on every
        one of given cables.

        scheme: PA - n - CA - 1 - CB - n - PB
        For every possible value of 'n; we move this scheme along the cable.

        :param cables: Indices of cables, None for all.
        """
        self._for_all_cables(self._schlumberger_inv_scheme, cables)

    def _schlumberger_inv_scheme(self, cable):
        n_points = len(cable)
        for i_pa in range(1, n_points-1):
            i_pb = i_pa + 1
            min_ica = max(0, i_pa +i_pb - n_points+1)
            for i_ca in range(min_ica, i_pa):
                i_cb = i_pb + (i_pa - i_ca)
                self._append([i_ca, i_cb, i_pa, i_pb])


    def marching_cc_pair(self, cables=None):
        """
        Create measurements using "inversed Schlumberger scheme" on every
        one of given cables.

        scheme: CA - 1 - CB - n - PA - 1 - PB
        For every possible value of 'n; we move this scheme along the cable.

        :param cables: Indices of cables, None for all.
        """
        self._for_all_cables(self._marching_cc_pair, cables)

    def _marching_cc_pair(self, cable):
        n_points = len(cable)
        for i_ca in range(n_points-3):
            i_cb = i_ca + 1
            for i_pa in range(i_cb+1, n_points-2):
                i_pb = i_pa + 1
                self._append([i_ca, i_cb, i_pa, i_pb])

    @property
    def simpeg_survey(self):
        """
        Create the measurement setup as an instance of DC Survey.
        :return: DC.Survey instance
        """
        if self._simpeg_survey is None:
            probe_points = self.points.all_points

            group_by_cc = {}
            for ca, cb, pa, pb in self.measurements:
                group_by_cc.setdefault((ca,cb), [])
                group_by_cc[(ca,cb)].append((pa,pb))

            src_list=[]
            sorted = [ (cc, pp) for cc, pp in group_by_cc.items()]
            sorted.sort()
            for cc, pp_list in sorted:
                rx_list = [DC.Rx.Dipole(probe_points[pp[0],:], probe_points[pp[1],:]) for pp in pp_list]
                src = DC.Src.Dipole(rx_list, probe_points[cc[0],:], probe_points[cc[1],:])
                src_list.append(src)
            self._simpeg_survey = DC.Survey(src_list)

        if self._values is not None:
            self._simpeg_survey.dobs = self._values
            self._simpeg_survey.std = self._errors

        return self._simpeg_survey


    def plot_measurements(self, ax, slice_axis=2, slice_value=0):
        """
        Plot survey scheme to ax Axis.

        """
        survey = self.simpeg_survey
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

    def print_summary(self):
        survey = self.simpeg_survey
        n_cc = len(survey.srcList)
        n_pp = 0
        n_locs = 0
        for cc in survey.srcList:
            n_pp += len(cc.rxList)
            for pp in cc.rxList:
                n_locs += len(pp.locs)
        print( "#cc : {}\n#pp : {} ({} per cc)\n#loc: {} ({} per pp)".format(n_cc, n_pp, n_pp/n_cc, n_locs, n_locs/n_pp))


    def median_apparent_conductivity(self):
        a_cond = []
        for U, ccpp in zip(self._values, self.measurements):
            ccpp = [self.points.all_points[idx, :] for idx in ccpp]
            capa = 1.0 / np.linalg.norm(ccpp[0] - ccpp[2])
            pacb = 1.0 / np.linalg.norm(ccpp[2] - ccpp[1])
            capb = 1.0 / np.linalg.norm(ccpp[0] - ccpp[3])
            pbcb = 1.0 / np.linalg.norm(ccpp[3] - ccpp[1])
            app_cond = capa - pacb - capb + pbcb
            a_cond.append(app_cond)
        return np.median(a_cond)

# class MeasurementScheme:
#     """
#     Defines position of measurement points (in 3d) and list of measurement quadruplets.
#     Optionally contains the measurement vector as well.
#     """
#
#     def __init__(self, points):
#
# TODO:
# - elementary shapes in fixed position, minimize number of parameters
# - implement shape transformation (affine)
# - implement _bbox functions, trivial for elementary shapes and transformed, aabb for union, ?? intersection
# - implement general plane projection and contour in terms of _inside and _bbox




def time_func(f):
    """
    Decorator to time a function (every call)
    :param f:
    :return:
    """
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        counter = getattr(self, 'counter', None)
        timer = - time.time()
        out = f(self, *args, **kwargs)
        timer += time.time()
        f_name = self.__class__.__name__ + '.' + f.__name__
        logging.debug("Time of {}: {}".format(f_name, timer))
        return out
    return wrapper