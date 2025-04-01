import numpy as np
import torch
import math


def pack_emission_params(traj, device='cpu'):
    name_to_index = traj.node_to_index
    index_to_name = {v: k for k, v in name_to_index.items()}
    edge_list = traj.edge_list
    edge_to_index = {edge: i for i, edge in enumerate(edge_list)}
    index_to_edge = {i: edge for edge, i in edge_to_index.items()}

    G = next(iter(traj.node_emission.values())).shape[0]  # genes
    E = len(edge_list)

    g_node = torch.zeros(len(name_to_index), G, device=device)

    for node, vec in traj.node_emission.items():
        g_node[name_to_index[node]] = torch.as_tensor(vec, dtype=torch.float32, device=device)

    g_a = torch.zeros(E, G, device=device)
    g_b = torch.zeros(E, G, device=device)
    K = torch.ones(E, G, device=device)
    sigma2 = torch.ones(E, G, device=device)
    pi = torch.zeros(E, G, device=device)

    for (u, v), i in edge_to_index.items():
        params = traj.emission_params.get((u, v), None)
        if params is None:
            raise KeyError(f"Missing emission params for edge: ({u}, {v})")
        u_idx = name_to_index[u]
        v_idx = name_to_index[v]
        g_a[i] = g_node[u_idx]
        g_b[i] = g_node[v_idx]
        K[i] = torch.as_tensor(params['K'], dtype=torch.float32, device=device)
        sigma2[i] = torch.as_tensor(params['r2'], dtype=torch.float32, device=device)
        pi[i] = torch.as_tensor(params['pi'], dtype=torch.float32, device=device)

    return edge_to_index, g_a, g_b, K, sigma2, pi, g_node



def emission_nll(X, P_idx, T, g, K, sigma2, index_to_edge, node_to_index, pi=None):
    """
    Args:
        X: [B, G] observed gene expression
        P_idx: [B] edge indices
        T: [B] latent time values in [0,1]
        g: [n_nodes, G] node-level expression parameters
        K, sigma2, pi: [n_edges, G] emission parameters per edge
        index_to_edge: dict[int → (u_name, v_name)]
        node_to_index: dict[u_name → int]
    """
    device = X.device
    P_idx = P_idx.long()

    # Convert index_to_edge into vectorized u/v lookup
    uv_pairs = [index_to_edge[i.item()] for i in P_idx]
    u_idx = torch.tensor([node_to_index[u] for u, _ in uv_pairs], device=device)
    v_idx = torch.tensor([node_to_index[v] for _, v in uv_pairs], device=device)

    g_a = g[u_idx]
    g_b = g[v_idx]

    mu = g_b + (g_a - g_b) * torch.exp(-K[P_idx] * T.unsqueeze(-1))
    var = sigma2[P_idx]
    log_var = torch.log(var + 1e-6)

    log_normal = -0.5 * ((X - mu)**2 / (var + 1e-6) + log_var + math.log(2 * math.pi))

    if pi is not None:
        pi_vals = pi[P_idx].clamp(min=1e-6, max=1 - 1e-6)
        log_zero = torch.log(pi_vals)
        log_nonzero = torch.log1p(-pi_vals) + log_normal
        log_prob = torch.where(X == 0, log_zero, log_nonzero)
    else:
        log_prob = log_normal

    return -log_prob.sum(dim=1).mean()


