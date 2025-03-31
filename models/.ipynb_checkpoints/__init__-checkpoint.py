"""
Model components for probabilistic trajectory inference.
"""

from .trajectory import TrajectoryGraph, initialize_trajectory
from .posterior import TreeVariationalPosterior
from .belief import BeliefPropagator
from .emission import pack_emission_params, emission_nll
from .loss import compute_elbo, compute_elbo_batch, compute_kl_beta, continuity_penalty