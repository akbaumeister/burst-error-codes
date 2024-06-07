import math

import matplotlib.pyplot as plt
import numpy as np

# draw the A3 lattice
M = np.array([[1, 0, -1], [1/math.sqrt(3), -2/math.sqrt(3), 1/math.sqrt(3)]])


def plot_in_lattice(p):
    x = np.dot(M, p)[0]
    y = np.dot(M, p)[1]
    plt.plot(x, y, 'o', 'k')
    return 0


lattice_points = [[0,0,0],[-1, 0, 1],[1, 0, -1],[-1, 1, 0],[1, -1, 0],[0, -1, 1],[0, 1, -1]]
lattice_points += [(1, 1, -2), (1, -2, 1), (1, 1, -2), (1, -2, 1), (-2, 1, 1), (-2, 1, 1), (2, -1, -1), (2, -1, -1), (-1, 2, -1), (-1, -1, 2), (-1, 2, -1), (-1, -1, 2), (2, -2, 0), (2, 0, -2), (-2, 2, 0), (-2, 0, 2), (0, 2, -2), (0, -2, 2)]
lattice_points += [(1, 2, -3), (1, -3, 2), (2, 1, -3), (2, -3, 1), (-3, 1, 2), (-3, 2, 1), (3, -1, -2), (3, -2, -1), (-1, 3, -2), (-1, -2, 3), (-2, 3, -1), (-2, -1, 3), (3, -3, 0), (3, 0, -3), (-3, 3, 0), (-3, 0, 3), (0, 3, -3), (0, -3, 3)]


xp = []
yp = []
for p in lattice_points:
    plot_in_lattice(p)

plt.show()