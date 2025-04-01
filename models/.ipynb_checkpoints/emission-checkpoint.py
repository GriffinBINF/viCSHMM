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

    G = next(iter(traj.node_emission.values())).shape[0]  # genes
    E = len(edge_list)

    # Node-based g_v vector
    g_node = torch.zeros(len(traj.node_to_index), G)
    missing_nodes = [
        node for node in traj.node_to_index
        if node not in traj.node_emission
    ]
    if missing_nodes:
        print("Missing emission vectors for nodes:", missing_nodes)
    assert not missing_nodes, f"Missing emission vectors for nodes: {missing_nodes}"

    
    for node, vec in traj.node_emission.items():
        g_node[traj.node_to_index[node]] = torch.tensor(vec)

    missing_nodes = [
        node for node in traj.node_to_index
        if node not in traj.node_emission
    ]
    assert not missing_nodes, f"Missing emission vectors for nodes: {missing_nodes}"

    g_a = torch.zeros(E, G)
    g_b = torch.zeros(E, G)
    K = torch.ones(E, G)
    sigma2 = torch.ones(E, G)
    pi = torch.zeros(E, G)

    for (u_name, v_name), i in edge_to_index.items():
        params = traj.emission_params.get((u_name, v_name), None)
        if params is None:
            raise KeyError(f"Missing emission params for edge: ({u_name}, {v_name})")
        u_idx = traj.node_to_index[u_name]
        v_idx = traj.node_to_index[v_name]
        g_a[i] = g_node[u_idx]
        g_b[i] = g_node[v_idx]
        K[i] = torch.tensor(params['K'])
        sigma2[i] = torch.tensor(params['r2'])
        pi[i] = torch.tensor(params['pi'])

    return edge_to_index, g_a.to(device), g_b.to(device), K.to(device), sigma2.to(device), pi.to(device), g_node.to(device)


def emission_nll(X, P_idx, T, g, K, sigma2, index_to_edge, node_to_index, pi=None):
    # Move everything to CPU for safe debugging
    X = X.cpu()
    P_idx = P_idx.cpu()
    T = T.cpu()
    g = g.cpu()
    K = K.cpu()
    sigma2 = sigma2.cpu()
    if pi is not None:
        pi = pi.cpu()
    device = torch.device("cpu")

    """
    Args:
        X: [B, G] observed gene expression
        P_idx: [B] edge indices
        T: [B] latent time values in [0,1]
        g: [n_nodes, G] node-level expression parameters
        K, sigma2, pi: [n_edges, G] emission parameters per edge
        index_to_edge: dict[int → (u_idx, v_idx)]
    """
    device = X.device

    try:
        p_values = [int(i) for i in P_idx.cpu()]
        # print("P_idx shape:", P_idx.shape)
        # print("P_idx values:", p_values)
        # print("Max valid index:", max(index_to_edge.keys()))
    
        invalid = [i for i in p_values if i < 0 or i >= len(index_to_edge)]
        if invalid:
            raise ValueError(f"Invalid indices in P_idx: {invalid}")
    except Exception as e:
        print("Exception while inspecting P_idx:", e)
        raise


    u_indices = torch.tensor([node_to_index[index_to_edge[i][0]] for i in p_values], device=device)
    v_indices = torch.tensor([node_to_index[index_to_edge[i][1]] for i in p_values], device=device)



    g_a = g[u_indices]
    g_b = g[v_indices]

    mu = g_b + (g_a - g_b) * torch.exp(-K[P_idx] * T.unsqueeze(-1))
    var = sigma2[P_idx]
    log_var = torch.log(var + 1e-6)

    log_normal = -0.5 * ((X - mu)**2 / (var + 1e-6) + log_var + np.log(2 * np.pi))

    if pi is not None:
        pi_vals = pi[P_idx].clamp(min=1e-6, max=1 - 1e-6)
        log_zero = torch.log(pi_vals)
        log_nonzero = torch.log(1 - pi_vals) + log_normal
        log_prob = torch.where(X == 0, log_zero, log_nonzero)
    else:
        log_prob = log_normal

    return -log_prob.sum(dim=1).mean()

