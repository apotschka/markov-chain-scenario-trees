"""Markov chain scenario trees.

Computes optimally pruned Markov chain scenario trees.

Notes
-----
Uses the polynomial-time algorithm published in [1]_.

References
----------
.. [1] C. Leidereiter, D. Kouzoupis, M. Diehl, A. Potschka, "Fast optimal
   pruning for Markov chain scenario tree NMPC", 2019.
"""

from itertools import product
from warnings import warn
import numpy as np
from scipy import sparse, linalg
import scipy.sparse as sps
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy.matlib
from time import perf_counter


def branchwise(tree):
    """Return sets of nodes and edges from a dict-based tree.
    
    Parameters
    ----------
    tree: dict
        Representation of a scenario tree with as a dictionary with
        scenario tuple keys and probability values.

    Returns
    -------
    nodes: set
        Set of `nodes` of the tree represented by scenario tuples.
    edges: set
        Set of `edges` of type (`nodes`[i], `nodes`[j]) of the tree.
    """
    edges = {(v[0:i], v[0:i+1]) for v in tree for i in range(1, len(v))}
    nodes = {v for e in edges for v in e}
    return nodes, edges


def prefix_subtree(prefix, node_probs):
    """Recursively enumerate subtrees preferring larger probabilities.

    Recursive generator to enumerate subtrees starting with prefix sorted
    according to node probabilities node_probs. Used for plotting scenario
    trees.

    Parameters
    ----------
    prefix: tuple
        Prefix that characterizes the subtree.
    node_probs: dict
        Contains for each partial scenario the accumulated probabilities of
        the corresponding subtree.

    Yields
    ------
    tuple
        A scenario of the subtree.
    """
    ext = sorted([(-p, n) for n, p in node_probs.items() if n[:-1] == prefix])
    if ext:
        for _, nodes in ext:
            yield from prefix_subtree(nodes, node_probs)
    else:
        yield prefix


def plot_scenario_tree(tree):
    """Plot scenario tree.

    Parameters
    ----------
    tree: dict
        Representation of a scenario tree with as a dictionary with
        scenario tuple keys and probability values.

    See Also
    --------
    scenario_tree_from_markov_chain
    """
    # obtain scenario tree depth
    n_levels = len(next(iter(tree)))
    # create nodes and edges
    nodes, edges = branchwise(tree)
    G = nx.Graph()
    G.add_edges_from(edges)
    # compute layout
    node_probs = {node: 0. for node in nodes}
    for s, p in tree.items():
        for l in range(len(s)):
            node_probs[s[:l+1]] += p
    pos = {}
    upper = [None,] * n_levels
    lower = [0,] * n_levels
    for s in prefix_subtree((), node_probs):
        for l in range(n_levels):
            if s[:l+1] not in pos:
                if upper[l] is None:
                    pos[s[:l+1]] = (l, 0)
                    upper[l] = 0
                elif pos[s[:l]][1] == 0:
                    if upper[l] <= -lower[l]:
                        upper[l] += 1
                        pos[s[:l+1]] = (l, upper[l])
                    else:
                        lower[l] -= 1
                        pos[s[:l+1]] = (l, lower[l])
                elif pos[s[:l]][1] > 0:
                    upper[l] += 1
                    pos[s[:l+1]] = (l, upper[l])
                else:
                    lower[l] -= 1
                    pos[s[:l+1]] = (l, lower[l])
    # draw scenario tree
    nx.draw_networkx_edges(G, pos, edges, width=1)
    nx.draw_networkx_nodes(G, pos, node_size=1, node_color="k")
    plt.axis("off")


def scenario_tree_from_markov_chain(A, depth, start_state=0,
        max_scenarios=1000, prob_coverage=0.99, min_scenario_prob=1e-8,
        interactive_visualization=False, verbose=False, timing=False):
    """Generate scenario tree from a Markov chain.

    A polynomial-time algorithm that enumerates the scenarios of a Markov
    chain scenario tree of given depth with non-increasing scenario
    probabilities.  The enumeration stops if a maximum number of scenarios
    has been found, or if the probability coverage of the tree reaches a
    certain threshold, or if only single scenarios of very small
    probability are left to be added.

    Parameters
    ----------
    A: (N,N) ndarray
        Markov chain transition matrix.
    depth: int
        The depth of the scenario tree.
    start_state: int
        The inital state of the Markov chain.
    max_scenarios: int
        The maximum number of returned scenarios.
    prob_coverage: float
        Probability coverage threshold.
    min_scenario_prob: float
        Minimum single scenario probability to consider. All scenarios with
        smaller probability are discarded immediately. Choosing this
        parameter wisely can greatly improve the runtime of the algorithm.
    interactive_visualization: bool
        Repeatedly call `print_scenario_tree` when a new scenario was
        found.
    verbose: bool
        Print information when a new scenario was found.
    timing: bool
        Measure performance.

    Returns
    -------
    tree: dict
        Representation of a scenario tree with as a dictionary with
        scenario tuple keys and probability values.
    cum_probs: (len(tree),) ndarray
        Cumulative probability (coverage) of the scenarios.
    max_cut_scenario_prob: float
        Maximum of all discarded single scenario probabilities.
    cum_timings (len(tree),) ndarray
        Only if `timing`: Cumulative runtime for each computed scenario.

    Notes
    -----
    Uses the polynomial-time algorithm published in [1]_.

    References
    ----------
    .. [1] C. Leidereiter, D. Kouzoupis, M. Diehl, A. Potschka, "Fast
       optimal pruning for Markov chain scenario tree NMPC", 2019.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> A = csr_matrix([[0.875, 0.125], [0.125, 0.875]])
    >>> scenario_tree_from_markov_chain(A, depth=3)
    ({(0, 0, 0, 0): 0.669921875,
      (0, 0, 0, 1): 0.095703125,
      (0, 0, 1, 1): 0.095703125,
      (0, 1, 1, 1): 0.095703125,
      (0, 0, 1, 0): 0.013671875,
      (0, 1, 0, 0): 0.013671875,
      (0, 1, 1, 0): 0.013671875},
     array([0.66992188, 0.765625  , 0.86132812, 0.95703125, 0.97070312,
            0.984375  , 0.99804688]),
     0.0)
    """
    # initialize timing
    if timing:
        start = perf_counter()
        cum_timings = np.zeros(max_scenarios)

    # Safely remove small entries from T to speed up tree generation
    T = sps.coo_matrix(A)
    keep = np.absolute(T.data) >= min_scenario_prob
    T = sps.coo_matrix((T.data[keep], (T.row[keep], T.col[keep])), A.shape)
    T = T.tocsr()

    # represent the trees as a dictionary of tuples
    tree = {}
    cum_prob = 0
    cum_probs = np.zeros(max_scenarios)

    # initialize priority queue R of reachable vertices
    # (with negative probabilities, because heapq pops the minimimal entry)
    # Each entry of the queue contains the following fields:
    #  0: negative scenario probability
    #  1: negative length of scenario
    #  2: scenario encoded as tuple
    R = [(-1., 0, (start_state,))]
    #for i in range(T.shape[0]):
    #    heapq.heappush(R, (-invariant_distribution[i], (-1,i)))

    # extend by one vertex at a time
    n_scenarios = 0
    max_cut_scenario_prob = 0.
    while R:
        # get vertex from R with maximum probability
        neg_prob, neg_len, v = heapq.heappop(R)
        prob = -neg_prob
        if len(v) < depth + 1: # check if maximal depth is reached
            # extend set of reachable vertices
            irn = v[-1]
            begin, end = T.indptr[irn], T.indptr[irn+1]
            for i, T_val in zip(T.indices[begin:end], T.data[begin:end]):
                p = prob * T_val
                if p >= min_scenario_prob:
                    heapq.heappush(R, (-p, neg_len-1, v + (i,)))
                else:
                    max_cut_scenario_prob = max(p, max_cut_scenario_prob)
        else:
            tree[v] = prob # add to set of vertices
            if interactive_visualization:
                plot_scenario_tree(tree)
                plt.show()
            # accept and update cumulative probability
            cum_prob += prob
            cum_probs[n_scenarios] = cum_prob
            if timing:
                cum_timings[n_scenarios] = perf_counter() - start
            n_scenarios += 1
            if verbose:
                fmt = "Coverage: {:.3f}% with {} scenarios"
                print(fmt.format(cum_prob, n_scenarios))
            if cum_prob >= prob_coverage or n_scenarios >= max_scenarios:
                break

    if timing:
        return (tree, cum_probs[:len(tree)], max_cut_scenario_prob,
                cum_timings[:len(tree)])
    else:
        return tree, cum_probs[:len(tree)], max_cut_scenario_prob

