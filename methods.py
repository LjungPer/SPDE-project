
import numpy as np
import matplotlib.pyplot as plt
import time

from gridlod import util, fem, linalg, interp, coef, lod, pglod
from gridlod.world import World, Patch
from math import pi
from visualize import drawCoefficient

def load_reference_solution():
    # read reference solution from file
    uref = np.loadtxt('reference.txt', dtype=float)

    # read number of simulations made from file
    f = open('Mref.txt', 'r')
    M = int(f.read())
    f.close()

    return uref / M


def compute_ms_basis(world, a_fine, k):

    def computeKmsij(TInd):
        patch = Patch(world, k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, world.boundaryConditions)
        aPatch = lambda: coef.localizeCoefficient(patch, a_fine)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        return patch, correctorsList, csi.Kmsij

    patchT, correctorsListT, KmsijT = zip(*map(computeKmsij, range(world.NtCoarse)))

    basis_correctors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
    return basis_correctors




def full_noise(fine, Nh, num_time_steps, tau):

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
    for m in range(1, Nh + 1):
        for n in range(1, Nh + 1):
            if m + n - 1 > 55:      # skip terms on machine precision level
                continue
            else:
                fine_noise += noise_term(m, n, num_time_steps)
    return fine_noise


def generate_coefficient(fine, plot_coefficient=False):
    # random seed for coefficient
    np.random.seed(0)

    # construct ms coefficient
    n = 2
    A = np.kron(np.random.rand(fine // n, fine // n) * 0.9 + 0.1, np.ones((n, n)))
    a_fine = A.flatten()
    if plot_coefficient:
        plt.figure("OriginalCoefficient")
        drawCoefficient(np.array([fine, fine]), a_fine)
        plt.show()

    # reset seed
    t = 1000 * time.time()  # time in ms
    np.random.seed(int(t) % 2 ** 32)  # seed must be between 0 and 2 ** 32 - 1

    return a_fine