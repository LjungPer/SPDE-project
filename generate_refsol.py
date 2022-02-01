#####
# Multiscale parabolic spde example. Weak error.
#####

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from gridlod import util, fem, linalg, interp, coef, lod, pglod
from gridlod.world import World, Patch
from visualize import drawCoefficient
from math import pi

# what files to store in
ref_file = "refT1.txt"
sim_file = "MrefT1.txt"

# set random seed
np.random.seed(0)

# spatial parameters
fine = 2 ** 7
fine_world = np.array([fine, fine])
np_fine = np.prod(fine_world + 1)
xp_fine = util.pCoordinates(fine_world).flatten()
bc = np.array([[0, 0], [0, 0]])

# temporal parameters
T = 1
tau = T * 2 ** (-7)
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

# reset seed
t = 1000 * time.time()                  # time in ms
np.random.seed(int(t) % 2 ** 32)        # seed must be between 0 and 2 ** 32 - 1

def full_noise(N, num_time_steps):

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

    fine_noise = 0
    for m in range(1, N + 1):
        for n in range(1, N + 1):
            if m + n - 1 > 55:      # skip terms on machine precision level
                continue
            else:
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
    uref[free_fine] = 1
    for i in range(num_time_steps):
        L_fine_free = (M_fine * (W[:, i + 1] - W[:, i]))[free_fine]

        lhs = M_fine_free + tau * S_fine_free
        rhs = tau * L_fine_free + M_fine_free * uref[free_fine]

        uref[free_fine] = linalg.linSolve(lhs, rhs)

    return uref


def store_number_of_simulations(simulations):
    def read_int(filename):
        f = open(filename, 'r')
        tmp = int(f.read())
        f.close()
        return tmp

    def write_int(filename, tmp):
        f = open(filename, 'w')
        f.write(str(tmp))
        f.close()

    # store number of MC-simulations made
    if os.path.isfile(sim_file):
        M = read_int(sim_file)
        M += simulations
        write_int(sim_file, M)
        print('Total number of simulations made: %d' % M)
    else:
        write_int(sim_file, simulations)


# load or initialize reference solution
if os.path.isfile(ref_file):
    uref = np.loadtxt(ref_file, dtype=float)
else:
    uref = 0

# add simulations to reference solution
sims_to_add = 10000
for i in range(sims_to_add):
    print('----------------- %d/%d -----------------' %(i + 1, sims_to_add))

    # generate noise
    start = time.time()
    W = full_noise(fine, num_time_steps)
    end = time.time()
    print('Noise generated (%.1f sec).' %(end - start))

    # compute reference solution
    start = time.time()
    uref += u_ref(num_time_steps, W)
    end = time.time()
    print('Solution generated (%.1f sec).' %(end - start))

    # store reference solution to txt
    start = time.time()
    np.savetxt(ref_file, uref, fmt='%.16f')
    store_number_of_simulations(1)
    end = time.time()
    print('Storage updated (%.1f sec)' %(end - start))






