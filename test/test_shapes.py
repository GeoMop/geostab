
import pytest
import sys
#print("pp:", sys.path)
import numpy as np
import shapes as shp

import matplotlib.pyplot as plt


def test_transform():
    points = np.array([ [0,0,0], [1,0,0], [0,1,0], [0,0,1]])
    trans = shp.Transform()
    trans.apply(points)
    np.isclose(points, points)

    # shift
    trans  = trans.shift([1,2,3])
    trans.apply(points)
    ref_points = np.array([ [1,2,3], [2,2,3], [1,3,3], [1,2,4] ])
    np.isclose(ref_points, points)

    # scale
    trans = trans.scale([2, -1, 1])
    trans.apply(points)
    ref_points = np.array([ [2,-2,3], [4,-2,3], [2,-3,3], [2,-2,4] ])
    np.isclose(ref_points, points)

    # rotate Z-axis, pi/2
    trans = trans.rotate([0, 0, 2], np.pi/2)
    trans.apply(points)
    ref_points = np.array([ [2,2,3], [2,4,3], [3,2,3], [2,2,4] ])
    np.isclose(ref_points, points)

    # rotate X-axis, -pi/2
    trans = trans.rotate([3, 0, 0], -np.pi/2)
    trans.apply(points)
    ref_points = np.array([[2,-3 ,2], [2,-3 ,4], [3, -3,2], [2, -4, 2]])
    np.isclose(ref_points, points)



#
#
#
#
#
#
# def plot_contour(shape, ax, plane):
#     points = shape.contour(plane)
#     ax.plot(points[:, 0], points[:, 1], 'k--')
#

def plot_shape(shape, contour=False, center=[0,0,0]):
    """
    Plot the shape cut through center of bbox in all three axes.
    :param shape:
    :param contour:
    :return:
    """

    extent_min, extent_max = shape.aabb()
    fig, axes = plt.subplots(1, 3, figsize=(30, 5))

    for iw, ax in enumerate(axes):
        iu, iv = np.delete([0,1,2], iw)
        #diff = extent_max - extent_min
        O = np.zeros(3)
        O[iw] = center[iw]
        U = np.eye(3)[iu, :]
        V = np.eye(3)[iv, :]
        plane = (O, U, V)
        points = shape.contour(plane, 200)
        if len(points):
            ax.scatter(points[:, 0], points[:, 1])

        names = np.array(list('XYZ'))
        ax.set_xlabel(names[iu])
        ax.set_ylabel(names[iv])

    # x_grid = np.linspace(-5, 5, 40)
    # X, Y, Z = np.meshgrid(x_grid, x_grid, x    # col = np.zeros_like(X, dtype=int)
    # col[flags] = 1

    # cmap = colors.ListedColormap(['w', 'k'])
    # #bounds = [0, 1]
    # #norm = colors.BoundaryNorm([0.5], cmap.N)
    # #col = cmap(col)
    # #c_map = ['b', 'r']
    # #col = c_map[col]_grid)
    # X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
    # grid = np.stack([X, Y, Z], axis=1)
    # flags = shape.inside(grid)

    # axes_names = np.array(list('XYZ'))
    # for i_ax, sub_ax in enumerate(ax):
    #     i_xy = list(range(3))
    #     i_xy.remove(i_ax)
    #     names = axes_names[i_xy]
    #     X, Y = grid[:, i_xy].T
    #     sub_ax.scatter(X, Y, c=col, cmap =cmap, alpha=0.1)
    #     sub_ax.set_xlabel(names[0])
    #     sub_ax.set_ylabel(names[1])
    #     if contour:
    #         x_vec, y_vec = np.eye(3)[i_xy, :]
    #         plane = [[0,0,0], x_vec, y_vec]
    #         points = shape.contour(100, plane)
    #         sub_ax.plot(points[:, 0], points[:, 1], 'k--')
    return fig, axes







def test_sphere():
    sphere_center = [-0.5, 1, 0]
    s = shp.Sphere(1.5).shift(sphere_center)
    # plane Z=0, radius: 1.5
    # plane, Y=0, radius: sqrt(1.5**2 - 1) = 1.118
    # plane, X=0, redius: sqrt(1.5**2 - 0.25) = 1.41
    assert s.inside([0,0,0])
    assert not s.inside([1, 0, 0])
    assert not s.inside([0, -1, 0])
    assert not s.inside([0, 0, -1])
    assert not s.inside([0, 0, 1])



    # origin = [0.98, -0.4, 0]
    # fig, axes = plot_shape(s, center=origin)
    # diff = np.abs(np.array(origin) - np.array(sphere_center))
    # R = np.sqrt(1.5**2 - diff**2)
    # for i, r, ax in zip([0,1,2], R, axes):
    #     iu, iv = np.delete([0, 1, 2], i)
    #     circle_center = [ sphere_center[iu], sphere_center[iv]]
    #     circle = plt.Circle(circle_center, r, color='r')
    #     ax.add_artist(circle)
    #
    # plt.show()

def test_cylinder():
    s = shp.Cylinder(3, 6)  # [-0.5, 0, -3], [0.5, 0, 3],  3
    s = s.shift([0,0,-3]).rotate([0, 1, 0], 0.166)
    assert s.inside([0, 0, 0])
    assert s.inside([0, 0, 3])
    assert s.inside([0, 0, -3])
    assert s.inside([3, 0, 0])
    assert not s.inside([4, 0, 0])
    assert not s.inside([-3, 0, 2])

    #plot_shape(s, True)




def test_box():
    s = shp.Box([-2, 0, -3], [2, 1, 3])

    assert s.inside([0, 0.1, 0])
    assert s.inside([0, 0.1, 2.9])
    assert s.inside([0, 0.1, -2.9])
    assert s.inside([1.9, 0.1, 0])
    assert not s.inside([2.1, 0, 0])
    assert not s.inside([0, 0, 3.1])

    #plot_shape(s, True)
    #plt.show()