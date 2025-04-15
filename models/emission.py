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

def emission_nll(X, P_idx, T, g, K, sigma2_param, index_to_edge, node_to_index, pi=None, eps=EPSILON, log_eps=EPSILON_LOG):
    """
    Calculates the negative log-likelihood of observed data X given latent variables.
    Corrected version to handle variance input and improve numerical stability.
    *** Returns NLL per cell in the batch. ***

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
        eps (float): Small epsilon for clamping variance.
        log_eps (float): Small epsilon for clamping before log.

    Returns:
        Tensor: Negative log-likelihood *per cell* [batch_size].
    """
    device = X.device
    batch_size = X.shape[0]
    if P_idx.shape[0] != batch_size or T.shape[0] != batch_size:
         raise ValueError(f"Batch size mismatch: X={X.shape[0]}, P_idx={P_idx.shape[0]}, T={T.shape[0]}")

    P_idx = P_idx.long() # Ensure indices are long

    # Resolve edge indices to node indices
    # Handle potential device mismatches if index_to_edge values aren't indices directly
    # Assume index_to_edge maps edge_idx (int) -> node_indices (int, int)
    try:
        uv_pairs = [index_to_edge[i.item()] for i in P_idx]
        u_idx = torch.tensor([u for u, _ in uv_pairs], device=device, dtype=torch.long)
        v_idx = torch.tensor([v for _, v in uv_pairs], device=device, dtype=torch.long)
    except KeyError as e:
        print(f"Error resolving edge index: {e}. Max P_idx: {P_idx.max()}, Min P_idx: {P_idx.min()}")
        print(f"Available keys in index_to_edge: {list(index_to_edge.keys())[:10]}...") # Show some keys
        raise e
    except IndexError as e:
        print(f"Error accessing index_to_edge with P_idx: {e}")
        print(f"P_idx values: {P_idx}")
        raise e


    # Get corresponding node means, edge params
    try:
        g_a = g[u_idx] # [batch_size, G]
        g_b = g[v_idx] # [batch_size, G]
        K_e = K[P_idx] # [batch_size, G]
        sigma2_e = sigma2_param[P_idx] # [batch_size, G]
        if pi is not None:
            pi_e = pi[P_idx] # [batch_size, G]
        else:
            pi_e = None
    except IndexError as e:
        print(f"IndexError accessing g, K, sigma2, or pi.")
        print(f"u_idx max: {u_idx.max()}, min: {u_idx.min()}, shape: {u_idx.shape}, g shape: {g.shape}")
        print(f"v_idx max: {v_idx.max()}, min: {v_idx.min()}, shape: {v_idx.shape}, g shape: {g.shape}")
        print(f"P_idx max: {P_idx.max()}, min: {P_idx.min()}, shape: {P_idx.shape}")
        print(f"K shape: {K.shape}, sigma2 shape: {sigma2_param.shape}")
        if pi is not None: print(f"pi shape: {pi.shape}")
        raise e


    # --- Robust Variance Handling ---
    var = sigma2_e.clamp(min=eps)
    log_var = var.clamp(min=log_eps).log() # Clamp again before log
    # --- End Robust Variance Handling ---

    # Calculate mean expression (mu)
    # Ensure T is correctly shaped: [batch_size] -> [batch_size, 1]
    T_unsqueezed = T.unsqueeze(-1)
    mu = g_b + (g_a - g_b) * torch.exp(-K_e * T_unsqueezed) # [batch_size, G]

    sq_diff = (X - mu) ** 2
    log_normal = -0.5 * (sq_diff / var + log_var + math.log(2 * math.pi))

    # Check for non-finite values early
    if not torch.all(torch.isfinite(log_normal)):
        # print(f"DEBUG emission_nll: Non-finite values in log_normal detected.")
        # print(f"  var min/max: {var.min():.2e}/{var.max():.2e}")
        # print(f"  log_var min/max: {log_var.min():.2e}/{log_var.max():.2e}")
        # print(f"  sq_diff min/max: {sq_diff.min():.2e}/{sq_diff.max():.2e}")
        log_normal = torch.nan_to_num(log_normal, nan=-1e8, posinf=-1e8, neginf=-1e8) # Replace with large neg number


    # Handle dropout component if pi is provided
    if pi_e is not None:
        # Clamp pi strictly away from 0 and 1 before log calculation
        # Use 2*log_eps for 1.0 clamping as float32 can struggle near 1.0
        pi_vals = pi_e.clamp(min=log_eps, max=1.0 - (2*log_eps))

        log_zero = pi_vals.log() # log(pi)
        log_one_minus_pi = (1.0 - pi_vals).log() # log(1-pi)

        # Check for non-finite log pi values
        if not torch.all(torch.isfinite(log_zero)):
            # print(f"DEBUG emission_nll: Non-finite log_zero! Min pi_vals: {pi_vals.min():.2e}")
            log_zero = torch.nan_to_num(log_zero, nan=-1e8, posinf=-1e8, neginf=-1e8)
        if not torch.all(torch.isfinite(log_one_minus_pi)):
             # print(f"DEBUG emission_nll: Non-finite log_one_minus_pi! Max pi_vals: {pi_vals.max():.2e}")
             log_one_minus_pi = torch.nan_to_num(log_one_minus_pi, nan=-1e8, posinf=-1e8, neginf=-1e8)

        # Log probability for non-zero observations: log(1-pi) + log N(X | mu, var)
        log_nonzero = log_one_minus_pi + log_normal

        # Check combined term
        if not torch.all(torch.isfinite(log_nonzero)):
            # print(f"DEBUG emission_nll: Non-finite log_nonzero after addition!")
            log_nonzero = torch.nan_to_num(log_nonzero, nan=-1e8, posinf=-1e8, neginf=-1e8)

        # Select probability based on whether X is zero (handle numerical precision)
        is_zero = torch.abs(X) < 1e-9 # Use tolerance instead of exact zero comparison
        log_prob = torch.where(is_zero, log_zero, log_nonzero)
    else:
        # No dropout, just use the Gaussian log probability
        log_prob = log_normal

    # Final check before summing
    if not torch.all(torch.isfinite(log_prob)):
        # print("DEBUG emission_nll: Non-finite values detected in final log_prob! Replacing.")
        log_prob = torch.nan_to_num(log_prob, nan=-1e8, posinf=-1e8, neginf=-1e8)

    # Sum log probabilities over genes for each cell
    log_prob_per_cell = log_prob.sum(dim=1) # [batch_size]

    # Calculate NLL per cell
    nll_per_cell = -log_prob_per_cell

    # Final check on NLL values per cell
    if not torch.all(torch.isfinite(nll_per_cell)):
        # print(f"DEBUG emission_nll: Non-finite NLL per cell found! Replacing.")
        # print(f"Min/Max nll_per_cell before replace: {nll_per_cell.min():.2e}/{nll_per_cell.max():.2e}")
        nll_per_cell = torch.nan_to_num(nll_per_cell, nan=1e8, posinf=1e8, neginf=-1e8) # Replace with large positive NLL

    # *** RETURN NLL PER CELL ***
    return nll_per_cell # Shape [batch_size]



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
