"""
General-purpose utility functions for math, initialization, logging, etc.
"""
import numpy as np
# Expose math helpers
from .math import cubic_bezier
from .inference import initialize_beta_from_cell_assignment, find_path_index_for_edge, batch_indices

