"""
Training utilities for Variational Inference for Continuous-State HMMs.
Includes the main training runner and structure learning components.
"""

import torch

# Expose the main classes
from .loop import VIRunner
from .structure_learner import TrajectoryStructureLearner

# Note: Removed initialize_training_components function.
#       Initialization is now handled within VIRunner.
