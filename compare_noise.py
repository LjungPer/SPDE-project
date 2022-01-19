######
# Code to compare noise term W^N with a refined N, such as N = Nh, versus a noise term W^M with coarser M = Nc. The
# error between the two terms are measured in L2-norm at T = 1. Obviously, the decay of eigenvalues affects how fast
# the convergence is.
#######

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
xp_fine = util.pCoordinates(fine_world).flatten()
bc = np.array([[0, 0], [0, 0]])
N_list = [2, 4, 8, 16, 32]

# temporal parameters
T = 1
tau = 0.01
#time_interval = np.arange(0 + tau, T + tau, tau)   # not including 0, ending at T
num_time_steps = int(T / tau)

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

def brownian(num_time_steps):
    b = [0]
    for _ in range(num_time_steps):
        b.append(b[-1] + np.sqrt(tau) * np.random.normal())
    return np.array(b)

def noise_term(m, n, num_time_steps):
    x = np.linspace(0, 1, fine + 1)
    y = np.linspace(0, 1, fine + 1)
    X, Y = np.meshgrid(x, y)

    lambda_mn = 1 / 2 ** (m + n - 2)
    e_mn = 4 * np.sin(n * pi * X) * np.sin(m * pi * Y)
    space_mn = np.expand_dims((lambda_mn * e_mn).flatten(), axis=1)
    b = np.expand_dims(brownian(num_time_steps), axis=0)
    return np.dot(space_mn, b)

def full_noise(Nf, N_list, num_time_steps):
    fine_noise = 0
    coarse_noises = [0] * len(N_list)
    for m in range(1, Nf + 1):
        for n in range(1, Nf + 1):
            print(m, n)
            tmp = noise_term(m, n, num_time_steps)
            fine_noise += tmp
            for i in range(len(N_list)):
                if max(m, n) < N_list[i] + 1:
                    coarse_noises[i] += tmp
    return fine_noise, coarse_noises

# compute noise
Wf, Wc = full_noise(fine, N_list, num_time_steps)

error = []
for i in range(len(Wc)):
    error.append(np.sqrt(np.dot(Wf[:, -1] - Wc[i][:, -1], Wf[:, -1] - Wc[i][:, -1])))

# plot errors
plt.figure('Error')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=18)
plt.loglog(N_list, error, '-s', basex=2, basey=2)
plt.grid(True, which="both")
plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel('$1/H$', fontsize=22)
plt.show()