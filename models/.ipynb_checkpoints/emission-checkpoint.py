import numpy as np
import torch
import math


def pack_emission_params(traj, device='cpu'):
    """
    Packs emission parameters from the TrajectoryGraph into tensors.

    Returns:
        tuple: edge_to_index (dict), g_node (Tensor[N_nodes, G]),
               K (Tensor[N_edges, G]), sigma2 (Tensor[N_edges, G]),
               pi (Tensor[N_edges, G] or None)
    """
    name_to_index = traj.node_to_index
    index_to_name = {v: k for k, v in name_to_index.items()} # Needed? maybe not directly returned
    edge_list = list(traj.G_traj.edges()) # Use G_traj directly
    edge_to_index = {(u, v): i for i, (u, v) in enumerate(edge_list)}
    index_to_edge = {i: edge for edge, i in edge_to_index.items()} # Keep for emission_nll

    if not traj.node_emission:
        raise ValueError("TrajectoryGraph.node_emission is empty. Initialize emission parameters first.")
    G = next(iter(traj.node_emission.values())).shape[0]  # genes
    N_nodes = len(name_to_index)
    E = len(edge_list)

    g_node = torch.zeros(N_nodes, G, device=device)
    for node, vec in traj.node_emission.items():
        if node in name_to_index:
            g_node[name_to_index[node]] = torch.as_tensor(vec, dtype=torch.float32, device=device)
        # else: # Handle case where a node might be in node_emission but not the final graph? Unlikely.
        #     print(f"Warning: Node '{node}' found in node_emission but not in node_to_index.")

    # Initialize K, sigma2, pi based on edge_list
    K = torch.ones(E, G, device=device)
    sigma2 = torch.ones(E, G, device=device)
    # Determine if pi should be used based on the first edge's params
    first_edge = edge_list[0] if E > 0 else None
    use_pi = first_edge and traj.emission_params.get(first_edge) and 'pi' in traj.emission_params[first_edge]
    pi = torch.zeros(E, G, device=device) if use_pi else None

    for edge, i in edge_to_index.items():
        params = traj.emission_params.get(edge)
        if params is None:
            # Fallback: Use defaults (already set), but maybe warn?
            print(f"Warning: Missing emission params for edge: {edge}. Using default K=1, sigma2=1.")
            # Optionally, try to estimate from node params g_node[u], g_node[v]? More complex.
        else:
            try:
                K[i] = torch.as_tensor(params['K'], dtype=torch.float32, device=device)
                sigma2[i] = torch.as_tensor(params['r2'], dtype=torch.float32, device=device)
                if use_pi and 'pi' in params:
                     pi[i] = torch.as_tensor(params['pi'], dtype=torch.float32, device=device)
            except KeyError as e:
                raise KeyError(f"Missing key {e} in emission_params for edge {edge}") from e
            except Exception as e:
                raise RuntimeError(f"Error processing params for edge {edge}: {e}") from e

    # Return the core parameters needed for the model
    # g_a, g_b are derived inside emission_nll using g_node
    return edge_to_index, g_node, K, sigma2, pi


def emission_nll(X, P_idx, T, g, K, sigma2, index_to_edge, node_to_index, pi=None):
    """
    Calculates the negative log-likelihood of observations X given assignments.

    Args:
        X: [B, G] observed gene expression
        P_idx: [B] edge indices (long tensor)
        T: [B] latent time values in [0,1]
        g: [n_nodes, G] node-level expression parameters
        K: [n_edges, G] rate parameters per edge
        sigma2: [n_edges, G] variance parameters per edge
        index_to_edge: dict[int -> (u_name, v_name)] mapping edge index to node names
        node_to_index: dict[node_name -> int] mapping node names to indices in g
        pi: Optional[Tensor[n_edges, G]] dropout probabilities per edge

    Returns:
        Tensor: Scalar average negative log-likelihood per cell.
    """
    device = X.device
    if P_idx.dtype != torch.long:
         P_idx = P_idx.long() # Ensure long type

    B = X.shape[0]
    G_dim = X.shape[1]
    n_nodes = g.shape[0]
    n_edges = K.shape[0]

    # --- Input Validation ---
    if P_idx.shape[0] != B or T.shape[0] != B:
        raise ValueError("Shape mismatch: P_idx and T must have Batch size B.")
    if K.shape[0] != n_edges or sigma2.shape[0] != n_edges or (pi is not None and pi.shape[0] != n_edges):
        raise ValueError("Shape mismatch: K, sigma2, pi must have n_edges dimension.")
    if g.shape[1] != G_dim or K.shape[1] != G_dim or sigma2.shape[1] != G_dim or (pi is not None and pi.shape[1] != G_dim):
        raise ValueError("Shape mismatch: g, K, sigma2, pi must have G_dim dimension.")
    if torch.any(P_idx < 0) or torch.any(P_idx >= n_edges):
         raise ValueError(f"Edge index out of bounds [0, {n_edges-1}]. Found: min={P_idx.min()}, max={P_idx.max()}")


    # Convert P_idx to u_idx, v_idx (vectorized)
    u_idx_list, v_idx_list = [], []
    valid_mask = torch.ones(B, dtype=torch.bool, device=device)
    for i in range(B):
        edge_i = P_idx[i].item()
        try:
            u_name, v_name = index_to_edge[edge_i]
            u_idx = node_to_index[u_name]
            v_idx = node_to_index[v_name]
            # Check node index bounds
            if not (0 <= u_idx < n_nodes and 0 <= v_idx < n_nodes):
                print(f"Warning: Invalid node index resolved for edge index {edge_i}: u={u_idx}, v={v_idx}. Max node index is {n_nodes-1}. Masking cell {i}.")
                valid_mask[i] = False
                u_idx_list.append(0) # Append dummy index
                v_idx_list.append(0)
            else:
                u_idx_list.append(u_idx)
                v_idx_list.append(v_idx)
        except KeyError:
            # This shouldn't happen if P_idx is valid, but good safeguard
            print(f"Warning: Edge index {edge_i} not found in index_to_edge. Masking cell {i}.")
            valid_mask[i] = False
            u_idx_list.append(0)
            v_idx_list.append(0)

    if not torch.all(valid_mask):
        print(f"Warning: {torch.sum(~valid_mask)} cells masked due to invalid index resolution in emission_nll.")
        if torch.sum(valid_mask) == 0:
            print("Error: All cells masked in emission_nll. Returning NaN.")
            return torch.tensor(float('nan'), device=device)
        # Filter inputs based on mask
        X = X[valid_mask]
        P_idx = P_idx[valid_mask]
        T = T[valid_mask]
        u_idx_list = [idx for i, idx in enumerate(u_idx_list) if valid_mask[i]]
        v_idx_list = [idx for i, idx in enumerate(v_idx_list) if valid_mask[i]]
        B = X.shape[0] # Update batch size


    u_idx_tensor = torch.tensor(u_idx_list, device=device, dtype=torch.long)
    v_idx_tensor = torch.tensor(v_idx_list, device=device, dtype=torch.long)

    # Gather parameters for the batch
    g_a = g[u_idx_tensor]  # [B, G]
    g_b = g[v_idx_tensor]  # [B, G]
    K_batch = K[P_idx]     # [B, G]
    var_batch = sigma2[P_idx] # [B, G]

    # Clamp variance for numerical stability
    var_safe = var_batch.clamp(min=1e-6)
    log_var_safe = torch.log(var_safe)

    # Calculate expected expression mu
    T_col = T.unsqueeze(-1) # [B, 1]
    mu = g_b + (g_a - g_b) * torch.exp(-K_batch * T_col) # [B, G]

    # Calculate log probability under Normal distribution
    log_normal = -0.5 * ((X - mu)**2 / var_safe + log_var_safe + math.log(2 * math.pi))

    # Handle dropout (Zero-Inflated Negative Binomial inspired logic)
    if pi is not None:
        pi_batch = pi[P_idx].clamp(min=1e-6, max=1 - 1e-6) # [B, G]
        log_pi = torch.log(pi_batch)
        log_one_minus_pi = torch.log1p(-pi_batch)

        # Log prob for zero counts: log(pi + (1-pi)*Normal(0|mu, var))
        # Log prob for non-zero counts: log((1-pi)*Normal(x|mu, var)) -> log(1-pi) + log_normal
        # Use logsumexp trick for zero counts? More robust.
        # log P(x=0) = log( pi + (1-pi) * P_N(0|mu,var) )
        #           = log( pi_batch + (1-pi_batch) * exp(log_normal_at_zero) )
        # Where log_normal_at_zero is log_normal evaluated with X=0.
        # For simplicity, use the common ZINB approach:
        # if x == 0: log(pi + (1-pi) * P_model(0)) -> approximated as log(pi) if P_model(0) is small
        # if x > 0: log(1-pi) + log P_model(x)

        # Use torch.where: if X is 0, use log(pi), otherwise use log(1-pi) + log_normal
        # This is an approximation assuming Normal(0|mu, var) is small relative to pi.
        # A more accurate version would involve logsumexp, but let's stick to the simpler one first.
        log_prob = torch.where(X == 0, log_pi, log_one_minus_pi + log_normal)

    else:
        # No dropout model
        log_prob = log_normal

    # Sum over genes, average over batch
    # Handle potential NaNs from log_prob before summing/averaging
    if torch.isnan(log_prob).any():
        # print("Warning: NaN detected in log_prob within emission_nll. Replacing NaNs with large negative value.")
        log_prob = torch.nan_to_num(log_prob, nan=-1e9) # Replace NaN with large negative number

    nll_per_cell = -log_prob.sum(dim=1) # Sum NLL over genes for each cell
    mean_nll = nll_per_cell.mean()     # Average NLL over cells in the batch

    return mean_nll