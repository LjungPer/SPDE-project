#####
# Implementation of a standard backward Euler-Galerkin Finite Element Method for the parabolic problem
#                                    d/dt u - div(A*grad(u)) = f
# in Omega x (0,T], with intiial value u(.,0) = 0 in Omega, and homogeneous Dirichlet boundary condition.
#
# Here A = ? and f(t,x,y) = 1, and a reference solution is computed on the fine grid.
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
fine = 128
fine_world = np.array([fine, fine])
np_fine = np.prod(fine_world + 1)
xp_fine = util.pCoordinates(fine_world).flatten()
bc = np.array([[0, 0], [0, 0]])
N_list = [2, 4, 8, 16, 32, 64]

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
n = 1
A = 102 + 100 * np.sin(n * 2 * pi * X) * np.sin(n * 2 * pi * Y)
a_fine = A.flatten()
if plot_coefficient:
    plt.figure("OriginalCoefficient")
    drawCoefficient(fine_world, a_fine)
    plt.show()

# define source function
def f(t):
    return np.ones(np_fine)

# compute reference solution
def u_ref(time_interval):

    # define fine world
    world = World(fine_world, fine_world // fine_world, bc)

    # create fine matrices
    S_fine = fem.assemblePatchMatrix(fine_world, world.ALocFine, a_fine)
    M_fine = fem.assemblePatchMatrix(fine_world, world.MLocFine)

    # find coarse free indices
    boundary_map = bc == 0
    fixed_free = util.boundarypIndexMap(fine_world, boundaryMap=boundary_map)
    free_fine = np.setdiff1d(np.arange(np_fine), fixed_free)

    # construct free matrices
    S_fine_free = S_fine[free_fine][:, free_fine]
    M_fine_free = M_fine[free_fine][:, free_fine]

    uref = np.zeros(np_fine)
    for t in time_interval:
        L_fine_free = (M_fine * f(t))[free_fine]

        lhs = M_fine_free + tau * S_fine_free
        rhs = tau * L_fine_free + M_fine_free * uref[free_fine]

        uref[free_fine] = linalg.linSolve(lhs, rhs)

    return uref

uref = u_ref(time_interval)
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
    error.append(np.sqrt(np.dot(uref - U_fine, uref - U_fine)))
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


