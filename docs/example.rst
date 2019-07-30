Spring packets example
----------------------

We take an example of a fault tolerant system, where a few masses are coupled
via packets of springs. Each spring has the same failure probability independent
of its age. This leads to a Markov chain. Its transition matrix can be computed
with the module :ref:`label-spring-packets`::

    >>> import spring_packets
    >>> from mcst import scenario_tree_from_markov_chain, plot_scenario_tree
    >>> T, _ = spring_packets.transition_matrix(4, 3, 0.035)
    >>> print(A.shape, A.nnz)
    (256, 256) 40960
    >>> print("Scenarios in complete tree:", 256**5)
    Scenarios in complete tree: 1099511627776
    >>> tree_opts = {"depth": 5, "start_state": 255, "max_scenarios": 50}
    >>> tree, cum_probs, _ = scenario_tree_from_markov_chain(A, **tree_opts)
    >>> fmt = "Coverage of optimally pruned trees: {:.2f}% (1 scenario), {:.2f}% (50 scenarios)"
    >>> print(fmt.format(100*cum_probs[0], 100*cum_probs[49]))
    Coverage of optimally pruned trees: 10.01% (1 scenario), 40.99% (50 scenarios)
    >>> plot_scenario_tree(tree)

.. image:: ../figures/optimally-pruned-tree.svg
   :alt: An optimally pruned scenario tree

