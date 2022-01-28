#####
# Multiscale parabolic spde example. Weak error.
#####

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, fem, linalg
from gridlod.world import World
from methods import compute_ms_basis, load_reference_solution, generate_coefficient, full_noise


# spatial parameters
fine = 2 ** 7
fine_world = np.array([fine, fine])
np_fine = np.prod(fine_world + 1)
xp_fine = util.pCoordinates(fine_world).flatten()
bc = np.array([[0, 0], [0, 0]])
N = 16

# temporal parameters
T = 0.1
tau = T * 2 ** (-7)
num_time_steps = int(T / tau)

# generate coefficient
a_fine = generate_coefficient(fine)

# load u_ref
uref = load_reference_solution()

k = int(np.log2(N)) + 1

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
basis_correctors = compute_ms_basis(world, a_fine, k)
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
error = []
U_fine = 0
for j in range(1, m + 1):
    print('------------ %d/%d ------------' %(j, m))

    W = full_noise(fine, num_time_steps, tau)

    U_coarse = np.zeros(np_coarse)
    U_coarse[free_coarse] = 1
    for i in range(num_time_steps):
        L_free = (ms_basis.T * M_fine * (W[:, i + 1] - W[:, i]))[free_coarse]

        lhs = M_coarse_free + tau * S_coarse_free
        rhs = tau * L_free + M_coarse_free * U_coarse[free_coarse]

        U_coarse[free_coarse] = linalg.linSolve(lhs, rhs)

    U_fine += ms_basis * U_coarse
    Em_U = U_fine / j
    error.append(np.sqrt(np.dot(uref - Em_U, uref - Em_U)))
    print('Error: %.4f' %error[j - 1])



# plot errors
plt.figure('Error')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=18)
plt.loglog(range(m), error, '-s', basex=2, basey=2)
plt.grid(True, which="both")
plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel('$1/H$', fontsize=22)
plt.show()
