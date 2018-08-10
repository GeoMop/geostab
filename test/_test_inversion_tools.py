import pytest
import numpy as np
from inversion_tools import Sphere, Cylinder, AABox


def plot_contour(shape, ax, plane):
    points = shape.contour(plane)
    ax.plot(points[:, 0], points[:, 1], 'k--')


def plot_shape(shape, contour=False):
    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig, ax = plt.subplots(1, 3, figsize=(30, 5))
    x_grid = np.linspace(-5, 5, 40)
    X, Y, Z = np.meshgrid(x_grid, x_grid, x_grid)
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
    grid = np.stack([X, Y, Z], axis=1)
    flags = shape.inside(grid)
    col = np.zeros_like(X, dtype=int)
    col[flags] = 1

    cmap = colors.ListedColormap(['w', 'k'])
    #bounds = [0, 1]
    #norm = colors.BoundaryNorm([0.5], cmap.N)
    #col = cmap(col)
    #c_map = ['b', 'r']
    #col = c_map[col]
    axes_names = np.array(list('XYZ'))
    for i_ax, sub_ax in enumerate(ax):
        i_xy = list(range(3))
        i_xy.remove(i_ax)
        names = axes_names[i_xy]
        X, Y = grid[:, i_xy].T
        sub_ax.scatter(X, Y, c=col, cmap =cmap, alpha=0.1)
        sub_ax.set_xlabel(names[0])
        sub_ax.set_ylabel(names[1])
        if contour:
            x_vec, y_vec = np.eye(3)[i_xy, :]
            plane = [[0,0,0], x_vec, y_vec]
            points = shape.contour(100, plane)
            sub_ax.plot(points[:, 0], points[:, 1], 'k--')
    plt.show()






def test_sphere():
    s = Sphere([-0.5, 1, 0], 1.5)

    assert s.inside([0,0,0])
    assert not s.inside([1, 0, 0])
    assert not s.inside([0, -1, 0])
    assert not s.inside([0, 0, -1])
    assert not s.inside([0, 0, 1])

    #plot_shape(s)


def test_cylinder():
    s = Cylinder([-0.5, 0, -3], [0.5, 0, 3],  3)

    assert s.inside([0, 0, 0])
    assert s.inside([0, 0, 3])
    assert s.inside([0, 0, -3])
    assert s.inside([3, 0, 0])
    assert not s.inside([4, 0, 0])
    assert not s.inside([-3, 0, 2])

    plot_shape(s, True)




def test_aabox():
    s = AABox([-2, 0, -3], [2, 1, 3])

    assert s.inside([0, 0.1, 0])
    assert s.inside([0, 0.1, 2.9])
    assert s.inside([0, 0.1, -2.9])
    assert s.inside([1.9, 0.1, 0])
    assert not s.inside([2.1, 0, 0])
    assert not s.inside([0, 0, 3.1])

    #plot_shape(s)