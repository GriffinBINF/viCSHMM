import numpy as np
import torch

def batch_indices(N, batch_size):
    """Yield random minibatch indices from total N cells."""
    perm = torch.randperm(N)
    for i in range(0, N, batch_size):
        yield perm[i:i + batch_size]


def find_path_index_for_edge(edge, posterior):
    for i, path_id in enumerate(posterior.path_ids):
        edge_list = posterior.paths[path_id]
        if edge in edge_list:
            return i
    return None


def initialize_beta_from_cell_assignment(posterior, cell_assignment, sharpness=0.01):
    alpha = posterior.alpha.data
    beta = posterior.beta.data

    for i, (cell, row) in enumerate(cell_assignment.iterrows()):
        edge = row['edge']
        if edge is None:
            continue
        t = float(np.clip(row['latent_time'], 1e-3, 1 - 1e-3))

        p_idx = find_path_index_for_edge(edge, posterior)
        if p_idx is None:
            continue

        alpha_val = t * (1 - sharpness) / sharpness + 1
        beta_val = (1 - t) * (1 - sharpness) / sharpness + 1

        alpha[i, p_idx] = alpha_val
        beta[i, p_idx] = beta_val
