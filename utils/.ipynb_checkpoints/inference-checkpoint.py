import numpy as np
import torch
import pandas as pd

def batch_indices(N, batch_size):
    """Yield random minibatch indices from total N cells."""
    perm = torch.randperm(N)
    for i in range(0, N, batch_size):
        yield perm[i:i + batch_size]

def canonical_edge(edge):
    """Convert edge tuple to a standard string-based representation."""
    u, v = edge
    return (str(u).strip(), str(v).strip())

def find_path_index_for_edge(edge, posterior):
    edge = canonical_edge(edge)
    for i, path_id in enumerate(posterior.path_ids):
        edge_list = [canonical_edge(e) for e in posterior.paths[path_id]]
        if edge in edge_list:
            return i
    return None


def initialize_beta_from_cell_assignment(posterior, cell_assignment, edge_to_index, sharpness=0.01):
    """
    Initializes posterior Beta distributions from initial cell_assignment (edge, t).
    """
    for i, row in enumerate(cell_assignment.itertuples()):
        edge = row.edge
        t = float(np.clip(row.latent_time, 1e-3, 1 - 1e-3))
        if edge not in edge_to_index:
            continue
        edge_idx = edge_to_index[edge]
        posterior.alpha.data[i, edge_idx] = (1.0 - t) / sharpness
        posterior.beta.data[i, edge_idx] = t / sharpness


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
