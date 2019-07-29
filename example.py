"""
Plot optimally pruned Markov chain trees for spring packets example.
"""

import numpy as np
import spring_packets
from mcst import scenario_tree_from_markov_chain, plot_scenario_tree
import matplotlib.pyplot as plt

tree_depth = 5
N = 4 # number of spring packets
m = 3 # number of springs per packet
p = 0.035 # fault probability for each spring

tree_opts = {"prob_coverage": 1.0, "min_scenario_prob": 1e-8}

A, _ = spring_packets.transition_matrix(N, m, p)

S = 2000
n_scen = np.arange(1, S+1)
tree, coverage, _, runtimes = scenario_tree_from_markov_chain(A, tree_depth,
        start_state=A.shape[0]-1, max_scenarios=S, verbose=False, timing=True,
        **tree_opts)

n_scenarios = 50

tree_pruned, cum_probs, _ = scenario_tree_from_markov_chain(A, tree_depth,
        start_state=A.shape[0]-1, max_scenarios=n_scenarios)
coverage_nominal = cum_probs[0]
coverage_pruned = cum_probs[-1]

print(f"Coverage: {coverage_nominal} (nominal)")
print(f"Coverage: {coverage_pruned} (pruned)")
print(f"{n_scenarios} scenarios")

plt.figure(1).clear()
plot_scenario_tree(tree_pruned)

plt.figure(2).clear()
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, num=2)
axes[0].plot(n_scen, runtimes, "k-", label="Runtime")
axes[0].set_ylabel("Runtime [s]")
axes[1].plot(n_scen, 100 * coverage, "k-", label="Coverage")
axes[1].set_ylabel("Coverage [%]")
axes[1].set_xlabel("Number of scenarios")

