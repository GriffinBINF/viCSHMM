import torch
import math
from utils.constants import EPSILON, EPSILON_LOG

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

def emission_nll(X, P_idx, T, g, K, sigma2_param, index_to_edge, node_to_index, pi=None):
    """
    Calculates the negative log-likelihood of observed data X given latent variables.
    Corrected version to handle variance input and improve numerical stability.

    Args:
        X (Tensor): Gene expression data [batch_size, G].
        P_idx (Tensor): Sampled edge indices [batch_size].
        T (Tensor): Sampled latent time [batch_size].
        g (Tensor): Node emission means [N_nodes, G].
        K (Tensor): Edge-specific decay rates [E, G].
        sigma2_param (Tensor): Edge-specific variances [E, G]. *Receives variance.*
        index_to_edge (dict): Maps edge index -> (u_idx, v_idx) tuple.
        node_to_index (dict): Maps node name -> node index.
        pi (Tensor, optional): Dropout probabilities [E, G]. Defaults to None.

    Returns:
        Tensor: Mean negative log-likelihood over the batch (scalar).
    """
    device = X.device
    P_idx = P_idx.long() # Ensure indices are long

    # Resolve edge indices to node indices
    uv_pairs = [index_to_edge[i.item()] for i in P_idx]
    u_idx = torch.tensor([u for u, _ in uv_pairs], device=device, dtype=torch.long)
    v_idx = torch.tensor([v for _, v in uv_pairs], device=device, dtype=torch.long)

    # Get corresponding node means
    g_a = g[u_idx]
    g_b = g[v_idx]

    # --- Robust Variance Handling ---
    var = sigma2_param[P_idx].clamp(min=EPSILON)
    log_var = var.clamp(min=EPSILON_LOG).log()
    # --- End Robust Variance Handling ---

    mu = g_b + (g_a - g_b) * torch.exp(-K[P_idx] * T.unsqueeze(-1))
    sq_diff = (X - mu) ** 2
    log_normal = -0.5 * (sq_diff / var + log_var + math.log(2 * math.pi))

    if not torch.all(torch.isfinite(log_normal)):
        # print("--- DEBUG: Non-finite values in log_normal ---") # Keep if needed
        log_normal = torch.nan_to_num(log_normal, nan=-1e8, posinf=-1e8, neginf=-1e8)


    # Handle dropout component if pi is provided
    if pi is not None:
        pi_edge = pi[P_idx]

        # --- More Robust Pi Clamping & Log Calculation ---
        # Clamp slightly MORE strictly away from 1.0 to help float32
        # Using 2*EPSILON_LOG, but can adjust if needed
        pi_vals = pi_edge.clamp(min=EPSILON_LOG, max=1.0 - 2*EPSILON_LOG)

        # Calculate log(pi) and log(1-pi)
        log_zero = pi_vals.clamp(min=EPSILON_LOG).log() # Clamp again just before log
        # Calculate log(1-pi). Clamp the argument (1-pi) away from zero before log.
        log_one_minus_pi = (1.0 - pi_vals).clamp(min=EPSILON_LOG).log()
        # --- End More Robust Pi Handling ---

        # --- Debug Checks for Logs ---
        if not torch.all(torch.isfinite(log_zero)):
             print(f"--- DEBUG PI (Corrected): Non-finite log_zero! pi_vals min: {pi_vals.min():.2e}")
             log_zero = torch.nan_to_num(log_zero, nan=-1e8, posinf=-1e8, neginf=-1e8)
        if not torch.all(torch.isfinite(log_one_minus_pi)):
             print(f"--- DEBUG PI (Corrected): Non-finite log_one_minus_pi! (1-pi_vals) min: {(1.0 - pi_vals).min():.2e}")
             log_one_minus_pi = torch.nan_to_num(log_one_minus_pi, nan=-1e8, posinf=-1e8, neginf=-1e8)
        # --- End Debug Checks ---

        # Log probability for non-zero observations: log(1-pi) + log N(X | mu, var)
        log_nonzero = log_one_minus_pi + log_normal

        # No need to check log_nonzero separately if components are finite
        # Add check + nan_to_num just in case addition causes issues (unlikely)
        if not torch.all(torch.isfinite(log_nonzero)):
             print(f"--- DEBUG PI (Corrected): Non-finite log_nonzero after addition!")
             log_nonzero = torch.nan_to_num(log_nonzero, nan=-1e8, posinf=-1e8, neginf=-1e8)

        # Select probability based on whether X is zero
        is_zero = X == 0.0
        log_prob = torch.where(is_zero, log_zero, log_nonzero)
    else:
        # No dropout, just use the Gaussian log probability
        log_prob = log_normal


    if not torch.all(torch.isfinite(log_prob)):
        # print("--- DEBUG: Non-finite values detected in final log_prob! Replacing. ---") # Keep if needed
        log_prob = torch.nan_to_num(log_prob, nan=-1e8, posinf=-1e8, neginf=-1e8)

    nll_per_cell = -log_prob.sum(dim=1)

    if not torch.all(torch.isfinite(nll_per_cell)):
        # print(f"--- DEBUG: Non-finite NLL per cell! Replacing. ---") # Keep if needed
        nll_per_cell = torch.nan_to_num(nll_per_cell, nan=1e8, posinf=1e8, neginf=1e8)

    return nll_per_cell.mean()



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

    return g_new, sigma2_new
