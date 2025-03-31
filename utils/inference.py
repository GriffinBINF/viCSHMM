import numpy as np
import torch
import pandas as pd

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

def make_beta_assignment_df(posterior, cell_assignment, sharpness=0.01):
    """
    Convert hard cell assignments to a DataFrame of beta distribution parameters
    usable for probabilistic trajectory visualization.

    This:
    1. Initializes alpha/beta distributions on the posterior object using
       `initialize_beta_from_cell_assignment(...)`
    2. Extracts the relevant (cell, edge) assignments and matches them to
       posterior path indices.

    Args:
        posterior: Posterior object with `alpha`, `beta`, and `initialize_...` method.
        cell_assignment (pd.DataFrame): Must contain 'edge' per cell.
        sharpness (float): Sharpness of the initialized beta (e.g. 0.01 = very diffuse).

    Returns:
        pd.DataFrame: Indexed by cell ID, with columns 'edge', 'alpha', 'beta'.
    """
    from utils.inference import initialize_beta_from_cell_assignment, find_path_index_for_edge

    initialize_beta_from_cell_assignment(posterior, cell_assignment, sharpness=sharpness)

    cell_ids = list(cell_assignment.index)
    edges = list(cell_assignment["edge"])
    alpha_vals = posterior.alpha.data.cpu().numpy()
    beta_vals = posterior.beta.data.cpu().numpy()

    records = []
    for i, edge in enumerate(edges):
        if edge is None:
            continue
        p_idx = find_path_index_for_edge(edge, posterior)
        if p_idx is None:
            continue
        records.append({
            "cell_id": cell_ids[i],
            "edge": edge,
            "alpha": alpha_vals[i, p_idx],
            "beta": beta_vals[i, p_idx],
        })

    return pd.DataFrame(records).set_index("cell_id")
