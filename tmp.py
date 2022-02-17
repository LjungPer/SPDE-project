


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
T = 0.5
tau = T * 2 ** (-7)
num_time_steps = int(T / tau)

# generate coefficient
a_fine = generate_coefficient(fine)

# compute reference solution
def Eu_ref(num_time_steps):

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

    Eu_ref = np.zeros(np_fine)
    Eu_ref[free_fine] = 1
    for _ in range(num_time_steps):

        lhs = M_fine_free + tau * S_fine_free
        rhs = M_fine_free * Eu_ref[free_fine]

        Eu_ref[free_fine] = linalg.linSolve(lhs, rhs)

    return Eu_ref


u = Eu_ref(num_time_steps)
for _ in range(1000):
    for N in N_list:
        if N is 4 or N is 8:
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

        sol_file = 'T=05' + 'sol_N=' + str(N) + '.txt'
        sim_file = 'T=05' + 'M_N=' + str(N) + '.txt'
        U_fine, M = load_solution(sol_file, sim_file)
        if N is 16:
            m = 100
        else:
            m = 200
        for j in range(1, m):
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
            e = np.sqrt(np.dot(u - Em_U, u - Em_U))
            error[N_list.index(N)] = e
            print('Error: %.5f' % e)


# error by mesh size = [0.21, 0.04938, 0.01173, 0.00271]