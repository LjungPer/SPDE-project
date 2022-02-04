#####
# Multiscale parabolic spde example. Weak error.
#####

import numpy as np
import matplotlib.pyplot as plt
import time

from gridlod import util, fem, linalg, interp, coef, lod, pglod
from gridlod.world import World, Patch

from methods import full_noise, load_reference_solution, store_number_of_simulations, load_solution, generate_coefficient

# spatial parameters
fine = 128
fine_world = np.array([fine, fine])
np_fine = np.prod(fine_world + 1)
xp_fine = util.pCoordinates(fine_world).flatten()
bc = np.array([[0, 0], [0, 0]])
N_list = [4, 8, 16, 32]

# for error plot
error = [0] * len(N_list)
y = [1 / N ** 2 for N in N_list]

# temporal parameters
T = 1
tau = T * 2 ** (-7)
num_time_steps = int(T / tau)

# generate coefficient
a_fine = generate_coefficient(fine)

# load pre-computed reference solution
uref, M = load_solution('refT1.txt', 'MrefT1.txt')
if M is not 0:
    uref = uref / M

for _ in range(1):
    for N in N_list:
        if N is not 8:
            continue
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
        sol_file = 'sol_N=' + str(N) + '.txt'
        sim_file = 'M_N=' + str(N) + '.txt'
        U_fine, M = load_solution(sol_file, sim_file)
        for j in range(1, m + 1):
            print('------------ N = %d ------------' % N)
            print('------------ %d/%d ------------' % (j, m))

            W = full_noise(fine, fine, num_time_steps, tau)

            U_coarse = np.zeros(np_coarse)
            U_coarse[free_coarse] = 1
            for i in range(num_time_steps):
                L_free = (ms_basis.T * M_fine * (W[:, i + 1] - W[:, i]))[free_coarse]

                lhs = M_coarse_free + tau * S_coarse_free
                rhs = tau * L_free + M_coarse_free * U_coarse[free_coarse]

                U_coarse[free_coarse] = linalg.linSolve(lhs, rhs)

            # add solution to total solution and store it
            U_fine += ms_basis * U_coarse
            np.savetxt(sol_file, U_fine, fmt='%.16f')
            store_number_of_simulations(sim_file, 1)

            print('Solution for N = %d updated.\nCurrent number of simulations: %d' % (N, M + j))

            # compute Monte-Carlo estimator and store the error
            Em_U = U_fine / (M + j)
            e = np.sqrt(np.dot(uref - Em_U, uref - Em_U))
            error[N_list.index(N)] = e
            print('Error: %.5f' % e)


# plot errors
plt.figure('Error')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=18)
plt.loglog(N_list, error, '-s', basex=2, basey=2)
plt.loglog(N_list, [y / 2 ** 3 for y in y], '--', basex=2, basey=2)
plt.grid(True, which="both")
plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel('$1/H$', fontsize=22)
plt.show()

# first errors (m = 10)
# error = [0.12391305633815408, 0.09048393591863425, 0.3422390801335623, 0.07194793616084011]

# next errors (m = 20)
# error = [0.10991776835390955, 0.0397043219459737, 0.11488576570419785, 0.044729276261100695]

# next errors (m = 100)
# error = [0.03588946187106183, 0.02335970097191432, 0.03181814233203731, 0.039553536445591744]

# m = 400
# error = [0.01783782029994698, 0.00752137755011207, 0.03609871708733352, 0.023350758088636663]

# N = 16, 32, m = 450
# error = [0, 0, 0.028791598054460903, 0.018641842919393675]

# N = 16, 32, m = 550
# error =

