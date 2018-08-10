import numpy as np
import copy

class Transform:
    """
    Affine transformation in the 3D space.
    """

    def __init__(self, matrix=None):
        """
        Initialize affine transform by a 4x4 transform matrix for homogeneous coordinates.
        :param matrix: (4,4) numpy matrix
        """
        if matrix is None:
            self.matrix = np.eye(4)[:3,:]
        else:
            matrix = np.array(matrix)
            assert matrix.shape == (3, 4)
            self.matrix = matrix

    def compose(self, other):
        matrix = np.dot(other.matrix[:, :3], self.matrix)
        matrix[:, -1] += other.matrix[:, -1]
        return Transform(matrix)

    def scale(self, scale_vec):
        """
        Scale in X,Y,Z directions.
        :param scale_vec: 3 array
        """
        matrix = np.dot(np.diag(scale_vec), self.matrix)
        return Transform(matrix)

    def shift(self, shift_vec):
        """
        :param shift_vec: 3 array
        """
        shift_vec = np.array(shift_vec)
        assert shift_vec.shape == (3,)
        matrix = self.matrix
        matrix[:, 3] += shift_vec
        return Transform(matrix)

    def rotate(self, axis, angle):
        """
        :param axis: 3 array, Vector of rotation axis.
        :param angle: Angle to rotate in radians.
        """
        unit_axis = np.array(axis, dtype=float)
        unit_axis /= np.linalg.norm(unit_axis)
        x, y, z = unit_axis
        a_cos = np.cos(angle)
        rot_mat = (1 - a_cos) * np.outer(unit_axis, unit_axis) + a_cos * np.eye(3)
        rot_mat += np.sin(angle) * np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        matrix = self.matrix
        matrix[:3, :] = np.dot(rot_mat, self.matrix[:3, :])
        return Transform(matrix)

    def invert(self):
        """
        Inversion transform.
        :return:
        """
        full_mat = np.r_[ self.matrix, [[0,0,0,1]] ]
        matrix = np.linalg.inv(full_mat)[:3, :]
        return Transform(matrix)

    def apply(self, points):
        """
        Apply the transformation to the array of points.
        :param points:
        :return:
        """
        N, dim = points.shape
        assert dim == 3
        h_points = np.ones( (N, 4) )
        h_points[:, :3] = points
        return np.dot(self.matrix[:3, :], h_points.T).T



class Shape:
    def __init__(self):
        self._transform = Transform()

    def inside(self, points):
        """
        :param points: List of points, array of points (N, 3)
        :return: Array of flags, True if corresponding point is inside the shape.
        """
        points = np.atleast_2d(np.array(points))
        result = self.implicit_function(points) > 0
        return result

    def implicit_function(self, points):
        """
        Implicit function (signed general distance function).
        :param points:  List of points, array of points (N, 3)
        :return: Array, implicit function evaluated at the points.
        """
        points = np.atleast_2d(np.array(points))
        # transform points to reference system of the shape
        to_ref_system = self._transform.invert()
        ref_points = to_ref_system.apply(points)
        result = self._implicit_fn(ref_points)
        if len(result) == 1:
            return result[0]
        return result


    def _implicit_fn(self, points):
        """
        Implicit definition of the shape.
        Should be implemented in particular shapes.
        :param points: (N, 3), array of points
        :return: array of point distances. Signed general boundary distance function.
        Possitive inside, negative outside, zero on the boundary.
        """
        raise Exception("Not implemented")

    def bbox(self):
        """
        Arbitrary bounding box of the shape.
        :return: Instance of Box shape.
        """
        extent_min, extent_max = self._bbox()
        diff = extent_max - extent_min
        box = Box().scale(diff).shift(extent_min)
        return box.transform(self._transform)

    def aabb(self):
        """
        Axes aligned bounding box.
        :return: (min_point, max_point)
        """
        box = self.bbox()
        verts = box.vertices()
        return ( np.min(verts, axis=0), np.max(verts, axis=0) )

    ##########################
    # Transformation wrappers:

    def transform(self, transform):
        cp_shape = copy.copy(self)
        cp_shape._transform = self._transform.compose(transform)
        return cp_shape

    def scale(self, scale_vec):
        """
        Scale in X,Y,Z directions.
        :param scale_vec: 3 array
        """
        return self.transform(Transform().scale(scale_vec))

    def shift(self, shift_vec):
        """
        :param shift_vec: 3 array
        """
        return self.transform(Transform().shift(shift_vec))

    def rotate(self, axis, angle):
        """
        :param axis: 3 array, Vector of rotation axis.
        :param angle: Angle to rotate in radians.
        """
        return self.transform(Transform().rotate(axis, angle))

    ##################################

    def contour(self, plane, n_points = 1000):
        """
        Return list of points on the contour of the plane - shape intersection.
        Algo:
        - find extent in the plane coordinates (assumes parametric plane).
        - use adaptive evaluation in the plane, to get contour points
        :param plane: (O, U, V) Plane given by three 3d points.
        Plane coordinate system:
            O - origin,
            OU - base vector for 'u' coordinate
            OV - base vector for 'v' coordinate
        :return: array (N, 3)
        """
        # transform the shape to the plane coordinates + W perpendicular direction
        O, U, V = plane
        u_base = U
        v_base = V
        w_base = np.cross(u_base, v_base)
        matrix = np.stack( (u_base, v_base, w_base, O), axis=1)
        trans = Transform(matrix).invert()   # transformation from UVW to XYZ

        plane_shape = self.transform(trans)
        extent_min, extent_max = plane_shape.aabb()
        return adaptive_squares(plane_shape, (extent_min, extent_max), 0, n_points)


def squares_split(squares):
    new_squares = []
    for p00, p10, p11, p01 in squares:
        pH0 = (p00 + p10) / 2
        p1H = (p10 + p11) / 2
        pH1 = (p11 + p01) / 2
        p0H = (p01 + p00) / 2
        pHH = (pH0 + pH1) / 2
        new_squares.append([p00, pH0, pHH, p0H])
        new_squares.append([pH0, p10, p1H, pHH])
        new_squares.append([pHH, p1H, p11, pH1])
        new_squares.append([p0H, pHH, pH1, p01])

    return new_squares


# from matplotlib import colors
#
#
# # set the colormap and centre the colorbar
# class MidpointNormalize(colors.Normalize):
# 	"""
# 	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
#
# 	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
# 	"""
# 	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
# 		self.midpoint = midpoint
# 		colors.Normalize.__init__(self, vmin, vmax, clip)
#
# 	def __call__(self, value, clip=None):
# 		# I'm ignoring masked values and all kinds of edge cases to make a
# 		# simple example...
# 		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
# 		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))



def adaptive_squares(shape, extent_xy, z, n_squares):
    """
    Intersection of the shape with the plane Z=z.
    :param shape:
    :param extent_xy:
    :param z:
    :param n_squares:
    :return:
    """

    # index pair of selected squares; start with all
    extent_min, extent_max = extent_xy
    p00 = np.array([extent_min[0], extent_min[1], z])
    p10 = np.array([extent_max[0], extent_min[1], z])
    p11 = np.array([extent_max[0], extent_max[1], z])
    p01 = np.array([extent_min[0], extent_max[1], z])
    squares = [[p00, p10, p11, p01]]

    # initial splitting
    for i in range(4):
        squares = squares_split(squares)

    # nodes_dist = np.empty_like(nodes[0,:], dtype=float)
    # nodes_sign = np.empty_like(nodes[0,:], dtype=bool)
    # nodes_eval_mask = np.ones_like(nodes[0,:], dtype=bool)

    n_bc_squares = 0
    while n_bc_squares < n_squares:
        # subdivide
        squares = squares_split(squares)
        squares = np.array(squares)
        # evaluate at nodes
        nodes_dist = shape.implicit_function(squares.reshape(-1, 3)).reshape(-1, 4)

        # import matplotlib.pyplot as plt
        # from matplotlib import colors
        #
        # fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        # # col = np.zeros_like(X, dtype=int)
        # # col[flags] = 1
        #
        # cmap = plt.get_cmap('coolwarm')
        # # #bounds = [0, 1]
        # norm = MidpointNormalize(midpoint=0.0, vmin=-1, vmax=1)
        # col = cmap(norm(nodes_dist.ravel()))
        # # #c_map = ['b', 'r']
        # # #col = c_map[col]
        # axes.scatter(squares.reshape(-1, 3)[:, 0], squares.reshape(-1, 3)[:, 1], c=col)

        nodes_sign = nodes_dist > 0
        # select BC squares
        bc_squares = np.sum(nodes_sign, axis=1)
        bc_squares = squares[np.isin(bc_squares, [1,2,3])]
        n_bc_squares = len(bc_squares)
        squares = bc_squares
        if len(squares) == 0:
            return []
    return np.average(np.array(squares), axis=1)[:, :2]








    #
    #
    # @staticmethod
    # def plane_plane_isec(a, b):
    #     """
    #     Compute intersection of two planes in 3D space.
    #
    #     :param a: [nx,ny,nz, d] - plane A in normal form
    #     :param b: [nx,ny,nz, d] - plane B in normal form
    #     :return: Line in parametric form: (A, T) where A is a point on the line, T is the direction vector.
    #     """
    #     T = np.cross(a[0:3], b[0:3])
    #     dim_of_third_plane = np.argmin(np.minimum(a, b))
    #     c = np.zeros(3)
    #     c[dim_of_third_plane] = 1
    #     mat = np.stack([a[0:3], b[0:3], c])
    #     rhs = np.array([a[3], b[3], 0])
    #     A = np.linalg.solve(mat, rhs)
    #     return (A, T)


class Sphere(Shape):
    """
    A sphere at origin.
    """

    def __init__(self, radius = 1.0):
        """
        :param radius: r > 0
        """
        super().__init__()
        assert radius > 0
        self.r = radius

    def _implicit_fn(self, points):
        return self.r - np.linalg.norm(points, axis=1)

    def _bbox(self):
        pt = self.r * np.ones(3)
        return (- pt, pt)


class Cylinder(Shape):
    """
    Z-axis cylinder with base at XY plane.
    """

    def __init__(self, radius = 1.0, height = 1.0):
        """
        :param height: h > 0
        :param radius: r > 0
        """
        super().__init__()
        assert radius > 0
        self.r = radius
        assert height > 0
        self.height = height

    def _implicit_fn(self, points):
        r, h = self.r, self.height
        z_dist = h / 2 - np.abs(points[:, 2] - h / 2)
        r_dist = self.r - np.linalg.norm(points[:, :2], axis=1)
        return z_dist * r_dist

    def _bbox(self):
        r, h = self.r, self.height
        return (np.array([-r, -r, 0]), np.array([+r, +r, h]))


class Box(Shape):


    """
    Unit cube.
    """
    def __init__(self, min_pt=np.zeros(3), max_pt=np.ones(3)):
        super().__init__()
        diff = np.array(max_pt) -np.array(min_pt)
        self._transform = self._transform.scale(diff).shift(np.array(min_pt))



    def _implicit_fn(self, points):
        point_dists = 1 / 2 - np.abs(points - 1 / 2)
        return np.prod(point_dists, axis=1)

    def _bbox(self):
        return (np.zeros(3), np.ones(3))

    def vertices(self):
        """
        :return: Array (8, 3). Vertices of the box.
        """
        verts = np.indices((2,2,2), dtype=float).T.reshape(-1,3)
        return self._transform.apply(verts)



class Union(Shape):
    def __init__(self, shapes):
        super().__init__()
        assert len(shapes) > 0
        self.shapes = shapes

    def _implicit_fn(self, points):
        dists = np.ones_like(points[:, 0], dtype=bool)
        for s in self.shapes:
            dists *= s.implicit_function(points)
        return dists

    def _bbox(self):
        """
        Union of AABB of sub shapes.
        :return:
        """
        N = len(self.shapes)
        min_pt = np.empty( (N, 3), dtype=float)
        max_pt = np.empty( (N, 3), dtype=float)
        for i, s in enumerate(self.shapes):
            min_pt[i], max_pt[i] = s.aabb()

        return ( np.min(min_pt, axis=0), np.max(max_pt, axis=0) )



