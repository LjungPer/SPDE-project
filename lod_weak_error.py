#####
# Multiscale parabolic spde example. Weak error.
#####

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, fem, linalg, interp, coef, lod, pglod
from gridlod.world import World, Patch
from visualize import drawCoefficient
from math import pi



# spatial parameters
fine = 64
fine_world = np.array([fine, fine])
np_fine = np.prod(fine_world + 1)
xp_fine = util.pCoordinates(fine_world).flatten()
bc = np.array([[0, 0], [0, 0]])
N_list = [2, 4, 8, 16, 32]

# temporal parameters
T = 1
tau = 0.01
num_time_steps = int(T / tau)

# for coefficient plot
plot_coefficient = False

# construct ms coefficient
n = 2
A = np.kron(np.random.rand(fine // n, fine // n) * 0.9 + 0.1, np.ones((n, n)))
a_fine = A.flatten()
if plot_coefficient:
    plt.figure("OriginalCoefficient")
    drawCoefficient(fine_world, a_fine)
    plt.show()

def brownian(num_time_steps):
    b = [0]
    for _ in range(num_time_steps):
        b.append(b[-1] + np.sqrt(tau) * np.random.normal())
    return np.array(b)

def noise_term(m, n, num_time_steps):
    x = np.linspace(0, 1, fine + 1)
    y = np.linspace(0, 1, fine + 1)
    X, Y = np.meshgrid(x, y)

    lambda_mn = 1 / 2 ** (m + n - 1)
    e_mn = 4 * np.sin(n * pi * X) * np.sin(m * pi * Y)
    space_mn = np.expand_dims((lambda_mn * e_mn).flatten(), axis=1)
    b = np.expand_dims(brownian(num_time_steps), axis=0)
    return np.dot(space_mn, b)

def full_noise(N, num_time_steps):
    fine_noise = 0
    for m in range(1, N + 1):
        for n in range(1, N + 1):
            #print(m, n)
            fine_noise += noise_term(m, n, num_time_steps)
    return fine_noise

# compute reference solution
def u_ref(num_time_steps, W):

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
    uref[free_fine] = 100
    for i in range(num_time_steps):
        L_fine_free = (M_fine * (W[:, i + 1] - W[:, i]))[free_fine]

        lhs = M_fine_free + tau * S_fine_free
        rhs = tau * L_fine_free + M_fine_free * uref[free_fine]

        uref[free_fine] = linalg.linSolve(lhs, rhs)

    return uref

# compute u_ref
M = 1000
uref = 0
for i in range(M):
    print('Reference solution   M = %d/%d' %(i + 1, M))
    W = full_noise(fine, num_time_steps)
    uref += u_ref(num_time_steps, W)
uref = uref / M

error = []
x = []
y = []
for N in N_list:
    x.append(N)
    y.append(1. / N ** 2)
    k = int(np.log2(N)) + 1

    coarse_world = np.array([N, N])
    coarse_el = fine_world // coarse_world
    world = World(coarse_world, coarse_el, bc)

    def computeKmsij(TInd):
        patch = Patch(world, k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, bc)
        aPatch = lambda: coef.localizeCoefficient(patch, a_fine)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        return patch, correctorsList, csi.Kmsij

    patchT, correctorsListT, KmsijT = zip(*map(computeKmsij, range(world.NtCoarse)))

    xp_coarse = util.pCoordinates(coarse_world).flatten()
    np_coarse = np.prod(coarse_world + 1)

    # create fine matrices
    S_fine = fem.assemblePatchMatrix(fine_world, world.ALocFine, a_fine)
    M_fine = fem.assemblePatchMatrix(fine_world, world.MLocFine)

    # construct coarse basis functions on fine grid
    basis = fem.assembleProlongationMatrix(coarse_world, coarse_el)
    basis_correctors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
    ms_basis = basis - basis_correctors

    # construct coarse ms matrices
    S_coarse = ms_basis.T * S_fine * ms_basis
    M_coarse = ms_basis.T * M_fine * ms_basis

    # find coarse free indices
    boundary_map = bc == 0
    fixed_coarse = util.boundarypIndexMap(coarse_world, boundaryMap=boundary_map)
    free_coarse = np.setdiff1d(np.arange(np_coarse), fixed_coarse)

    # construct free matrices
    S_coarse_free = S_coarse[free_coarse][:, free_coarse]
    M_coarse_free = M_coarse[free_coarse][:, free_coarse]

    m = 1000
    Em_U = 0
    for j in range(m):
        print('N = %d/%d   m = %d/%d' %(N, N_list[-1], j + 1, m))

        W = full_noise(fine, num_time_steps)

        U_coarse = np.zeros(np_coarse)
        U_coarse[free_coarse] = 100
        for i in range(num_time_steps):
            L_free = (ms_basis.T * M_fine * (W[:, i + 1] - W[:, i]))[free_coarse]

            lhs = M_coarse_free + tau * S_coarse_free
            rhs = tau * L_free + M_coarse_free * U_coarse[free_coarse]

            U_coarse[free_coarse] = linalg.linSolve(lhs, rhs)

        U_fine = ms_basis * U_coarse
        Em_U += U_fine
    Em_U = Em_U / m
    error.append(np.sqrt(np.dot(uref - Em_U, uref - Em_U)))


# plot errors
plt.figure('Error')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=18)
plt.loglog(N_list, error, '-s', basex=2, basey=2)
plt.loglog(x, [y / 2 ** 3 for y in y], '--', basex=2, basey=2)
plt.grid(True, which="both")
plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel('$1/H$', fontsize=22)
plt.show()

#error = [0.5100286016760015, 0.14729734362187077, 0.03680330880287353, 0.011767456523230928, 0.002507785495906016]