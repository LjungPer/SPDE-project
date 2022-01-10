#####
# Implementation of a standard backward Euler-Galerkin Finite Element Method for the parabolic problem
#                                    d/dt u - div(A*grad(u)) = f
# in Omega x (0,T], with intiial value u(.,0) = 0 in Omega, and homogeneous Dirichlet boundary condition.
#
# Here A = 1 and f(t,x,y) = (1+2t*pi^2)*sin(pi*x)*sin(pi*y), with analytical solution u(t,x,y) = t*sin(pi*x)*sin(pi*y).
#
# Error plotted and computed in L2-norm at time T = 1.
#####

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, fem, linalg
from gridlod.world import World
from visualize import drawCoefficient
from math import pi


# spatial parameters
fine = 64
fine_world = np.array([fine, fine])
np_fine = np.prod(fine_world + 1)
bc = np.array([[0, 0], [0, 0]])
N_list = [2, 4, 8, 16, 32]

# temporal parameters
T = 1
tau = 0.01
time_interval = np.arange(0 + tau, T + tau, tau)   # not including 0, ending at T

# for coefficient plot
plot_coefficient = False

# construct simple coefficient
x = np.linspace(0, 1, fine)
y = np.linspace(0, 1, fine)
X, Y = np.meshgrid(x, y)
A = np.ones([fine, fine])
a_fine = A.flatten()
if plot_coefficient:
    plt.figure("OriginalCoefficient")
    drawCoefficient(fine_world, a_fine)
    plt.show()

# define source function
def f(t):
    x = np.linspace(0, 1, fine + 1)
    y = np.linspace(0, 1, fine + 1)
    X, Y = np.meshgrid(x, y)
    return ((1 + 2 * t * pi ** 2) * np.sin(pi * X) * np.sin(pi * Y)).flatten()

# define analytical solution
def u(t):
    x = np.linspace(0, 1, fine + 1)
    y = np.linspace(0, 1, fine + 1)
    X, Y = np.meshgrid(x, y)
    return (t * np.sin(pi * X) * np.sin(pi * Y)).flatten()

error = []
x = []
y = []
for N in N_list:
    print(N)
    coarse_world = np.array([N, N])
    coarse_el = fine_world // coarse_world
    world = World(coarse_world, coarse_el, bc)

    xp_coarse = util.pCoordinates(coarse_world).flatten()
    np_coarse = np.prod(coarse_world + 1)

    # create fine matrices
    S_fine = fem.assemblePatchMatrix(fine_world, world.ALocFine, a_fine)
    M_fine = fem.assemblePatchMatrix(fine_world, world.MLocFine)

    # construct coarse basis functions on fine grid
    basis = fem.assembleProlongationMatrix(coarse_world, coarse_el)

    # construct coarse matrices
    S_coarse = basis.T * S_fine * basis
    M_coarse = basis.T * M_fine * basis

    # find coarse free indices
    boundary_map = bc == 0
    fixed_coarse = util.boundarypIndexMap(coarse_world, boundaryMap=boundary_map)
    free_coarse = np.setdiff1d(np.arange(np_coarse), fixed_coarse)

    # construct free matrices
    S_coarse_free = S_coarse[free_coarse][:, free_coarse]
    M_coarse_free = M_coarse[free_coarse][:, free_coarse]

    U_coarse = np.zeros(np_coarse)
    for t in time_interval:
        L_free = (basis.T * M_fine * f(t))[free_coarse]

        lhs = M_coarse_free + tau * S_coarse_free
        rhs = tau * L_free + M_coarse_free * U_coarse[free_coarse]

        U_coarse[free_coarse] = linalg.linSolve(lhs, rhs)

    U_fine = basis * U_coarse
    error.append(np.sqrt(np.dot(u(T) - U_fine, u(T) - U_fine)))
    x.append(N)
    y.append(1. / N)


# plot errors
plt.figure('Error')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=18)
plt.loglog(N_list, error, '-s', basex=2, basey=2)
plt.grid(True, which="both")
plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel('$1/H$', fontsize=22)


