"""Transition matrix for spring packets example.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import kron as skron
from scipy.special import comb


def transition_matrix(N, m, p):
    """Markov chain transition matrix for chained spring packets.

    Parameters
    ----------
    N: int
        Number of spring packets
    m: int
        Number of springs per packet.
    p: float
        Failure probability of each single spring.

    Returns
    -------
    T: ((m+1)**(N-1), (m+1)**(N-1)) csr_matrix
        Sparse matrix of transition probabilities.
    """
    # generate spring packet transition matrix
    T = np.eye(m+1)
    for i in range(m+1):
        for j in range(i):
            T[i,j] = comb(i, i-j) * p**(i-j)
            T[i,i] -= T[i,j]
    T = csr_matrix(T)

    # assemble scenario matrix:
    # entry [i,j]: number of springs in packet j of parameter realization i
    n_realizations = (m+1)**N
    o_to_m = np.arange(m+1)
    e = np.ones((1, m+1), dtype=int)
    n_springs = np.array([o_to_m])
    for i in range(N-1):
        n_springs = np.vstack((np.kron(n_springs, e),
            np.kron(np.repeat(e, (m+1)**i), o_to_m)))

    # assemble global transition matrix
    A = T
    for _ in range(N-1):
        A = skron(T, A)
    A = csr_matrix(A) # BSR (kron) -> CSR
    return A, n_springs

