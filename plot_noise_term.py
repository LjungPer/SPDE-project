####
# Code to plot one noise term sqrt(lambda_mn) * beta(t) * e_mn for m = n = 1. Also plots the beta(t) over the time
# interval to see that it corresponds well to the full term. Here lambda_mn = 1 / 2^(m+n-2) and e_mn = sin(n*pi*x) *
# sin(m*pi*y).
####

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from math import pi
from gridlod import util
from mpl_toolkits.mplot3d import Axes3D

# spatial parameters
fine = 64
fine_world = np.array([fine, fine])
np_fine = np.prod(fine_world + 1)
bc = np.array([[0, 0], [0, 0]])
N_list = [2, 4, 8, 16, 32]

def brownian(time_interval):
    b = [0]
    dt = time_interval[2] - time_interval[1]
    for t in time_interval:
        b.append(b[-1] + np.sqrt(dt) * np.random.normal())
    return np.array(b)

def noise_term(m, n, time_interval):
    x = np.linspace(0, 1, fine + 1)
    y = np.linspace(0, 1, fine + 1)
    X, Y = np.meshgrid(x, y)

    lambda_mn = 1 / 2 ** (m + n - 2)
    e_mn = 4 * np.sin(n * pi * X) * np.sin(m * pi * Y)
    space_mn = np.expand_dims((lambda_mn * e_mn).flatten(), axis=1)
    b = np.expand_dims(brownian(time_interval), axis=0)
    return np.dot(space_mn, b), b


# temporal parameters
T = 1
tau = 0.001
time_interval = np.arange(0 + tau, T + tau, tau)   # not including 0, ending at T

# generate one brownian path and corresponding term for Q-Wiener process W(t)
m = 1
n = 1
noise, b = noise_term(m, n, time_interval)

# plot brownian path
plt.plot(time_interval, b.squeeze()[1:])

# plot term on full grid for t = 0, 0.2, 0.4, 0.6
for i in range(4):
    fig = plt.figure('Noise')
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')

    xp = util.pCoordinates(fine_world)
    X = xp[0:, 1:].flatten()
    Y = xp[0:, :1].flatten()
    X = np.unique(X)
    Y = np.unique(Y)

    X, Y = np.meshgrid(X, Y)

    s = noise[:, i * 200 + 1]

    sol = s.reshape(fine_world + 1)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, sol, cmap=cm.coolwarm)
    ymin, ymax = ax.set_zlim()
    ax.set_zticks((ymin, ymax))
    ax.set_zlabel('$z$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.axis('on')


