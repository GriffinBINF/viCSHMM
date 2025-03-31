import torch
import numpy as np


def pack_emission_params(traj, device='cpu'):
    edge_to_index = {}
    index_to_edge = {}
    name_to_index = traj.node_to_index
    index_to_name = {v: k for k, v in name_to_index.items()}

    edge_list = []
    for i, (u_idx, v_idx) in enumerate(traj.edge_list):
        u_name = index_to_name[u_idx]
        v_name = index_to_name[v_idx]
        edge = (u_name, v_name)
        edge_to_index[edge] = i
        index_to_edge[i] = edge
        edge_list.append(edge)

    G = next(iter(traj.emission_params.values()))['g_a'].shape[0]
    E = len(edge_list)

    g_a = torch.zeros(E, G)
    g_b = torch.zeros(E, G)
    K = torch.ones(E, G)
    sigma2 = torch.ones(E, G)
    pi = torch.zeros(E, G)

    for (u_name, v_name), i in edge_to_index.items():
        params = traj.emission_params.get((u_name, v_name), None)
        if params is None:
            raise KeyError(f"Missing emission params for edge: ({u_name}, {v_name})")
        g_a[i] = torch.tensor(params['g_a'])
        g_b[i] = torch.tensor(params['g_b'])
        K[i] = torch.tensor(params['K'])
        sigma2[i] = torch.tensor(params['r2'])
        pi[i] = torch.tensor(params['pi'])

    return edge_to_index, g_a.to(device), g_b.to(device), K.to(device), sigma2.to(device), pi.to(device)


def emission_nll(X, P_idx, T, g_a, g_b, K, sigma2, pi=None):
    mu = g_b[P_idx] + (g_a[P_idx] - g_b[P_idx]) * torch.exp(-K[P_idx] * T.unsqueeze(-1))
    var = sigma2[P_idx]
    log_var = torch.log(var + 1e-6)

    log_normal = -0.5 * ((X - mu) ** 2 / (var + 1e-6) + log_var + np.log(2 * np.pi))

    if pi is not None:
        pi = pi[P_idx].clamp(min=1e-6, max=1 - 1e-6)
        log_zero = torch.log(pi)
        log_nonzero = torch.log(1 - pi) + log_normal
        log_prob = torch.where(X == 0, log_zero, log_nonzero)
    else:
        log_prob = log_normal

    return -log_prob.sum(dim=1).mean()
