Markov chain scenario trees
---------------------------

The scenario trees considered here are based on a Markov chain stochastic model
of an uncertain system parameter with finitely many possible states numbered
from 0 to *N*. The Markov chain is represented by its *N*-by-*N* transition
matrix *T*, whose entry (*i*, *j*) contains the probability of changing from
state *i* to state *j*. The transition matrix must be stochastic, i.e., its
elements must be nonnegative and each row must sum up to 1.

A scenario of depth *M* is a sequence of states of length *M* + 1. By
multiplying the corresponding transition probabilities from each state to the
next, we can assign a probability to the scenario. The entirety of all such
scenarios constitutes an enormous tree, which is exponential (*N* to the power
of *M*). This module computes optimal depth *M* subtrees that cover the
maximally attainable probability under restrictions on their number of contained
scenarios.

