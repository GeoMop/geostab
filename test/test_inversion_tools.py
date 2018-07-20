import pytest
import numpy as np
from inversion_tools import Sphere, Cylinder, AABox



def plot_shape(shape):
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
    ax[0].scatter(Y, Z, c=col, cmap =cmap, alpha=0.1)
    ax[0].set_xlabel('Y')
    ax[0].set_ylabel('Z')
    ax[1].scatter(X, Z, c=col, cmap =cmap, alpha=0.1)
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Z')
    ax[2].scatter(X, Y, c=col, cmap =cmap, alpha=0.1)
    ax[2].set_xlabel('X')
    ax[2].set_ylabel('Y')
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

    #plot_shape(s)


def test_aabox():
    s = AABox([-2, 0, -3], [2, 1, 3])

    assert s.inside([0, 0.1, 0])
    assert s.inside([0, 0.1, 2.9])
    assert s.inside([0, 0.1, -2.9])
    assert s.inside([1.9, 0.1, 0])
    assert not s.inside([2.1, 0, 0])
    assert not s.inside([0, 0, 3.1])

    #plot_shape(s)