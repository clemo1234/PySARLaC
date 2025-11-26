import numpy as np
from functools import reduce
from scipy.linalg import eigh, norm
import sys
import os


module_dir = os.path.abspath('../src')

# Add the directory to sys.path
sys.path.insert(0, module_dir)

import PySARLaC as sl


sx = np.array([[0., 1.], [1., 0.]])
sy = np.array([[0., -1j], [1j, 0.]])
sz = np.array([[1., 0.], [0., -1.]])
id2 = np.eye(2)

def kron_n(op_list):
    return reduce(np.kron, op_list)

def pauli_op_on_site(pauli, site, N):
    ops = [id2] * N
    ops[site] = pauli
    return kron_n(ops)

def build_tfim_hamiltonian(N, J, h, periodic=False):
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N-1):
        H += -J * (pauli_op_on_site(sz, i, N) @ pauli_op_on_site(sz, i+1, N))
    if periodic and N>1:
        H += -J * (pauli_op_on_site(sz, N-1, N) @ pauli_op_on_site(sz, 0, N))
    for i in range(N):
        H += -h * pauli_op_on_site(sx, i, N)
    return H

def expectation(state, operator):
    return np.vdot(state, operator @ state)

def matrix_element(bra, operator, ket):
    return np.vdot(bra, operator @ ket)

def correlator_matrix_from_eigstates(ops, evals, evecs, t):

    psi0 = evecs[:, 0]
    n_ops = len(ops)
    C = np.zeros((n_ops, n_ops), dtype=complex)
    n_states = evecs.shape[1]
    M = np.zeros((n_states, n_ops), dtype=complex)  # M[k,j] = <k| O_j |0>
    for k in range(n_states):
        psi_k = evecs[:, k]
        for j, Oj in enumerate(ops):
            M[k, j] = matrix_element(psi_k, Oj, psi0)
    
    weights = np.exp(-evals * t)
    for i in range(n_ops):
        for j in range(n_ops):
            C[i, j] = np.sum(weights * np.conjugate(M[:, i]) * M[:, j])
    return C

def build_trial_state_from_vector(v, ops, psi0):

    vec = np.zeros_like(psi0, dtype=complex)
    for i, vi in enumerate(v):
        vec += vi * (ops[i] @ psi0)
    nrm = norm(vec)
    if nrm == 0:
        return vec, 0.0
    return vec / nrm, nrm

N = 4        # number of spins
J = 1.0
h = 2.5      
periodic = False

t0 = 0
t  = 2

H = build_tfim_hamiltonian(N, J, h, periodic=periodic)

evals, evecs = eigh(H)

dim = 2**N

O_g  = sum(pauli_op_on_site(sx, i, N) for i in range(N))
O_e1 = pauli_op_on_site(sz, 0, N)
O_e2 = pauli_op_on_site(sz, 1 % N, N)
ops = [O_g, O_e1, O_e2]
labels = ["O_g = Σ Sx", "O_e1 = Sz(0)", "O_e2 = Sz(1)"]

# Build correlator matrices at t0 and t
Ct0 = correlator_matrix_from_eigstates(ops, evals, evecs, t0)
Ct  = correlator_matrix_from_eigstates(ops, evals, evecs, t)


corrs = []

for t_val in range(t0, t):
    corrs.append(correlator_matrix_from_eigstates(ops, evals, evecs, t_val))
print(corrs)