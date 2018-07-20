from SimPEG.EM.Static import DC
import numpy as np
import bisect

"""
TODO:
- check that produced SimPEG Survey have no performance drawback, compare tim if we generate Survey by native tools or 
  by our self.
- Optimal scheme for 1d cable, 2d forward model. Select measurements with maximal sensitivity to the changes in conductivity parameters,
  i.e.  derivative of   dP / (dC =  1)  by a parameter of a conductivity model.
  Algo:
  - for every conductivity parameter C
    - for every (Ca, Cb) compute derivative of potential field by C
        - find electrote with minimal sensitivity set to be Pa
        - find given number of electrodes with maximal sensitivity to be Pb
        - set quality of (Ca, Cb) pair to average of sensitivity of Pb - Pa, for selected Pb points
    - sort (Ca, Cb) pairs according to the quality metric
    - use first N pairs to perform inversion for syntetic test with C parameter set to 1 and other zero
    - increase N until we get different inversion, should by also close to best, including more measurements should
      decrease inversion quality   
"""
class PointSet:

    @staticmethod
    def points_line(start, end, n_points):
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

        self._append(points)
        return len(self.cables) - 1

    def _append_cable(self, points):
        start = self.cable_start[-1] + len(self.cables[-1])
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

    def clear(self):
        """
        Remove all measurements.
        """
        self.measurements = []

    def _append(self, m):
        self.measurements.append(tuple(m))

    def _for_all_cables(self, fn, cables):
        if cables is None:
            cables = self.points.cables
        else:
            cables = self.points.cables[cables]

        for cable in cables:
            fn(cable)


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


    def schlumberger_inv_scheme(self, cables):
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



    def get_simpeg_survey(self):
        """
        Create the measurement setup as an instance of DC Survey.
        :return: DC.Survey instance
        """
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
        survey = DC.Survey(src_list)
        return survey


# class MeasurementScheme:
#     """
#     Defines position of measurement points (in 3d) and list of measurement quadruplets.
#     Optionally contains the measurement vector as well.
#     """
#
#     def __init__(self, points):
#

class Shape:
    def inside(self, points):
        """
        :param points: List of points, array of points (N, 3)
        :return: Array of flags, True if corresponding point is inside the shape.
        """
        points = np.atleast_2d(np.array(points))
        result = self._inside(points)
        if len(result) == 1:
            return result[0]
        return result

class Sphere(Shape):
    """
    A sphere in 3d space.
    """
    def __init__(self, centre, radius):
        """
        :param centre: np.array, [x,y,z]
        :param radius: r >= 0
        """
        self.centre = np.array(centre)
        self.r = radius

    def _inside(self, points):
        return np.linalg.norm(points - self.centre[None, :], axis=1) < self.r

class Cylinder(Shape):
    """
    Arbitrary positioned cylinder.
    """
    def __init__(self, a_pt, b_pt, radius):
        """
        :param a_pt: Center of bottom base.
        :param b_pt: Center of top base.
        :param radius: Cylinder radius.
        """
        self.axis_line = [np.array(a_pt), np.array(b_pt)]
        self.r = radius

    def _inside(self, points):
        a_pt, b_pt = self.axis_line
        line_vec = b_pt - a_pt
        line_norm = np.linalg.norm(line_vec)
        line_point = np.dot(points - a_pt[None, :], line_vec) / (line_norm ** 2)
        projection = line_point[:, None] * line_vec[None, :] + a_pt[None, :]
        flag = np.linalg.norm(points - projection, axis=1) < self.r
        flag = np.logical_and(flag, 0 < line_point)
        flag = np.logical_and(flag, line_point < 1)
        return flag

class AABox(Shape):
    """
    Axes aligned box.
    """
    def __init__(self, a_pt, b_pt):
        min_pt = np.minimum(a_pt, b_pt)
        max_pt = np.maximum(a_pt, b_pt)
        self.corners = [min_pt, max_pt]

    def _inside(self, points):
        min_pt, max_pt = self.corners
        return np.logical_and(np.all(min_pt[None, :] < points, axis=1), np.all(points < max_pt[None, :], axis=1))



