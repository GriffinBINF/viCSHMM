import torch
import math


def pack_emission_params(traj, device='cpu'):
    name_to_index = traj.node_to_index
    edge_list = list(traj.G_traj.edges())
    edge_name_to_index = {edge: i for i, edge in enumerate(edge_list)}
    index_to_edge_name = {i: edge for edge, i in edge_name_to_index.items()}
    edge_list_indices = [(name_to_index[u], name_to_index[v]) for u, v in edge_list]
    edge_tuple_to_index = {v: k for k, v in enumerate(edge_list_indices)}

    G = next(iter(traj.node_emission.values())).shape[0]
    E = len(edge_list)
    N_nodes = len(name_to_index)

    g_node_init = torch.zeros(N_nodes, G, device=device)
    for node_name, vec in traj.node_emission.items():
        if node_name in name_to_index:
            g_node_init[name_to_index[node_name]] = torch.as_tensor(vec, dtype=torch.float32, device=device)

    K_init = torch.ones(E, G, device=device)
    sigma2_init = torch.ones(E, G, device=device)
    pi_init = torch.zeros(E, G, device=device)

    for edge_name, i in edge_name_to_index.items():
        params = traj.emission_params.get(edge_name)
        if params is None:
            continue
        if 'K' in params:
            K_init[i] = torch.as_tensor(params['K'], dtype=torch.float32, device=device)
        if 'r2' in params:
            sigma2_init[i] = torch.as_tensor(params['r2'], dtype=torch.float32, device=device).clamp(min=1e-6)
        if 'pi' in params:
            pi_init[i] = torch.as_tensor(params['pi'], dtype=torch.float32, device=device)

    return (edge_tuple_to_index, index_to_edge_name, name_to_index,
            g_node_init, K_init, sigma2_init, pi_init)


import math
import torch

def emission_nll(X, P_idx, T, g, K, log_sigma2, index_to_edge, node_to_index, pi=None):
    device = X.device
    P_idx = P_idx.long()

    # Use index-based edge tuples directly: (u_idx, v_idx)
    uv_pairs = [index_to_edge[i.item()] for i in P_idx]
    u_idx = torch.tensor([u for u, _ in uv_pairs], device=device)
    v_idx = torch.tensor([v for _, v in uv_pairs], device=device)

    g_a = g[u_idx]
    g_b = g[v_idx]

    sigma2 = log_sigma2.exp()  # 🔥 fix: sigma2 was in log-space
    var = sigma2[P_idx]
    mu = g_b + (g_a - g_b) * torch.exp(-K[P_idx] * T.unsqueeze(-1))
    log_var = log_sigma2[P_idx]

    log_normal = -0.5 * ((X - mu) ** 2 / (var + 1e-6) + log_var + math.log(2 * math.pi))

    if pi is not None:
        pi_vals = pi[P_idx].clamp(min=1e-6, max=1 - 1e-6)
        log_zero = torch.log(pi_vals)
        log_nonzero = torch.log1p(-pi_vals) + log_normal
        log_prob = torch.where(X == 0, log_zero, log_nonzero)
    else:
        log_prob = log_normal

    return -log_prob.sum(dim=1).mean()




def update_emission_means_variances(
    X, posterior, K, traj_graph, edge_tuple_to_index, epsilon=1e-8
):
    device = X.device
    N, G = X.shape
    E = posterior.n_edges
    N_nodes = len(traj_graph.node_to_index)

    with torch.no_grad():
        q_e = torch.softmax(posterior.edge_logits.detach(), dim=1)
        alpha = posterior.alpha.detach().clamp(min=epsilon)
        beta = posterior.beta.detach().clamp(min=epsilon)
        t_exp = alpha / (alpha + beta)

    edge_list = [(traj_graph.node_to_index[u], traj_graph.node_to_index[v])
                 if not isinstance(u := e[0], int) else e
                 for e in posterior.edge_list]

    u_idx = torch.tensor([u for u, _ in edge_list], device=device)
    v_idx = torch.tensor([v for _, v in edge_list], device=device)

    source_weights = q_e * (1 - t_exp)
    target_weights = q_e * t_exp

    weighted_X_source = source_weights.T @ X
    weighted_X_target = target_weights.T @ X

    g_new = torch.zeros(N_nodes, G, device=device)
    total_weights = torch.zeros(N_nodes, device=device)

    g_new.scatter_add_(0, u_idx.unsqueeze(1).expand(-1, G), weighted_X_source)
    g_new.scatter_add_(0, v_idx.unsqueeze(1).expand(-1, G), weighted_X_target)
    total_weights.scatter_add_(0, u_idx, source_weights.sum(0))
    total_weights.scatter_add_(0, v_idx, target_weights.sum(0))

    g_new = g_new / (total_weights.unsqueeze(1) + epsilon)
    g_new[total_weights < epsilon, :] = X.mean(0)

    g_a = g_new[u_idx]
    g_b = g_new[v_idx]

    mu = g_b.unsqueeze(0) + (g_a.unsqueeze(0) - g_b.unsqueeze(0)) * torch.exp(-K.unsqueeze(0) * t_exp.unsqueeze(2))
    sq_diff = (X.unsqueeze(1) - mu) ** 2
    weighted_sq_diff = q_e.unsqueeze(2) * sq_diff
    sum_weighted_sq_diff = weighted_sq_diff.sum(0)
    sum_weights = q_e.sum(0)

    sigma2_new = sum_weighted_sq_diff / (sum_weights.unsqueeze(1) + epsilon)
    sigma2_new = sigma2_new.clamp(min=epsilon)

    return g_new.detach(), sigma2_new.detach()
