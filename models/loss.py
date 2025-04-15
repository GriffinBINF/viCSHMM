import torch
import torch.distributions as D
import math
import networkx as nx # Needed for compute_proposal_distribution logic if not already imported

# Assuming these are correctly defined/imported from elsewhere in the models package
from .emission import emission_nll
from .posterior import TreeVariationalPosterior # For type hinting
from .belief import BeliefPropagator # For type hinting
# Assuming TrajectoryGraph might be needed for node_to_index etc.
from .trajectory import TrajectoryGraph # For type hinting
from .proposal import compute_proposal_distribution, compute_kl_beta

# Assuming constants are available
from utils.constants import EPSILON

def continuity_penalty(edge_idx, t, traj, index_to_edge_tuple, node_to_index):
    """
    Penalizes temporal inconsistencies at branch transitions.

    Args:
        edge_idx (Tensor): [N] LongTensor of sampled edge indices.
        t (Tensor): [N] Continuous latent time samples.
        traj (TrajectoryGraph): Object holding the node/edge graph.
        index_to_edge_tuple (dict): Maps edge_idx → (u_idx, v_idx)
        node_to_index (dict): Maps node name → node index

    Returns:
        Tensor: Mean penalty (scalar).
    """
    device = t.device
    penalties = []

    for i in range(edge_idx.shape[0]):
        u_idx, v_idx = index_to_edge_tuple[edge_idx[i].item()]

        # Check if v_idx has any children
        has_child = any((v_idx, x) in traj.edge_list for x in node_to_index.values())
        if has_child:
            penalties.append((1 - t[i]) ** 2)

        # Check if u_idx has any parents
        has_parent = any((x, u_idx) in traj.edge_list for x in node_to_index.values())
        if has_parent:
            penalties.append(t[i] ** 2)

    return torch.stack(penalties).mean() if penalties else t.new_zeros(1).squeeze()

def resolve_edge_nodes(edge_idx_tensor, index_to_edge_tuple_map, device=None):
    """Resolves edge indices to source (u) and target (v) node indices."""
    device = device or edge_idx_tensor.device
    # Ensure edge_idx_tensor is on CPU for iteration if it's large, or handle on device
    edge_indices_list = edge_idx_tensor.tolist()
    u_idx = [index_to_edge_tuple_map[i][0] for i in edge_indices_list]
    v_idx = [index_to_edge_tuple_map[i][1] for i in edge_indices_list]
    return torch.tensor(u_idx, device=device, dtype=torch.long), \
           torch.tensor(v_idx, device=device, dtype=torch.long)

def compute_elbo(
    # Core data and model components
    X, cell_indices, traj, posterior, edge_tuple_to_index,
    g, K, sigma2, # Generative params (g, sigma2 assumed non-gradient)
    pi=None, # Optional dropout param (assumed learnable if not None)
    # VI and IS components
    belief_propagator=None,
    n_samples=10, # Number of IS samples
    # ELBO weights (some may be less relevant now)
    kl_weight=1.0,
    kl_p_weight=1.0,
    # IS proposal hyperparameters (passed via kwargs or fixed)
    proposal_beta_const: float = 10.0,
    proposal_target_temp: float = 1.0,
    prop_edge_probs_all=None,
    prop_alpha_all=None,
    prop_beta_all=None,
    # Optional weights for other terms (e.g., branch entropy)
    branch_entropy_weight=0.0, # Default to 0 unless explicitly used
    # -- Unused args from original signature (kept for compatibility) --
    t_cont_weight=0.0,
    transition_weight=0.0,
    l1_weight=0.0,
    tau=1.0, # Gumbel-softmax tau no longer used for sampling
    # -------------------------------------------------------------
    eps=EPSILON # Small constant for stability
):
    """
    Computes the ELBO estimate using Importance Sampling (IS).

    Calls compute_proposal_distribution internally (inefficiently)
    and performs IS estimate of E_q[log p(X|z)] - KL(q||p).
    """
    device = X.device
    batch_size = X.shape[0] # Note: X is X_batch here
    n_edges = posterior.n_edges

    # --- (Re)compute Proposal Distribution Parameters for ALL cells ---
    # Inefficient: Ideally computed once per epoch outside this function.
    # Computing it here to satisfy the "don't change training loop" constraint.
    if prop_edge_probs_all is None or prop_alpha_all is None or prop_beta_all is None:
        raise ValueError("Proposal parameters must be precomputed and passed to compute_elbo")
    assert prop_edge_probs_all.shape[0] == posterior.n_cells, "Mismatched proposal shape"


    # --- Get Parameters for the Current Batch ---
    # Proposal parameters for the batch
    prop_edge_probs_batch = prop_edge_probs_all[cell_indices]
    prop_alpha_batch = prop_alpha_all[cell_indices] # [batch, E]
    prop_beta_batch = prop_beta_all[cell_indices]  # [batch, E]

    # Posterior (q) parameters for the batch
    edge_logits_batch = posterior.edge_logits[cell_indices]
    q_edge_probs_batch = torch.softmax(edge_logits_batch, dim=1)
    alpha_q_batch = posterior.alpha[cell_indices] # [batch, E]
    beta_q_batch = posterior.beta[cell_indices]  # [batch, E]

    # --- Importance Sampling ---
    total_weighted_log_p_x_given_z = torch.zeros(batch_size, device=device)
    all_log_weights_list = []

    # Prepare proposal distributions for sampling
    prop_edge_dist = D.Categorical(probs=prop_edge_probs_batch)

    for s in range(n_samples):
        # 1. Sample edge from proposal p_prop(edge)
        edge_idx_prop = prop_edge_dist.sample() # [batch]

        # 2. Sample time from proposal p_prop(t | edge_prop)
        alpha_p_gathered = prop_alpha_batch[torch.arange(batch_size), edge_idx_prop]
        beta_p_gathered = prop_beta_batch[torch.arange(batch_size), edge_idx_prop]
        prop_time_dist = D.Beta(alpha_p_gathered.clamp(min=eps), beta_p_gathered.clamp(min=eps))
        t_prop = prop_time_dist.rsample().clamp(eps, 1.0 - eps) # [batch]

        # 3. Calculate log p(X | z_prop) = -NLL
        nll_sample = emission_nll(
            X, edge_idx_prop, t_prop, g, K, sigma2, # Pass X_batch, generative params
            index_to_edge=posterior.index_to_edge,
            node_to_index=traj.node_to_index,
            pi=pi # Pass learnable pi if used
        ) # Shape [batch_size], NLL value for each cell
        log_p_x_given_z = -nll_sample # Shape [batch_size]

        # 4. Calculate log q(z_prop)
        log_q_edge = torch.log_softmax(edge_logits_batch, dim=1)[torch.arange(batch_size), edge_idx_prop]
        alpha_q_gathered = alpha_q_batch[torch.arange(batch_size), edge_idx_prop]
        beta_q_gathered = beta_q_batch[torch.arange(batch_size), edge_idx_prop]
        q_time_dist = D.Beta(alpha_q_gathered.clamp(min=eps), beta_q_gathered.clamp(min=eps))
        log_q_time = q_time_dist.log_prob(t_prop)
        log_q_z = log_q_edge + log_q_time

        # 5. Calculate log p_proposal(z_prop)
        log_prop_edge = prop_edge_dist.log_prob(edge_idx_prop)
        log_prop_time = prop_time_dist.log_prob(t_prop)
        log_prop_z = log_prop_edge + log_prop_time

        # 6. Calculate Log Importance Weight
        log_weight = log_q_z - log_prop_z
        all_log_weights_list.append(log_weight.detach()) # Store detached for diagnostics

        # print(f"--- Sample {s} Debug ---")
        # print(f"  log_q_z (mean): {log_q_z.mean():.4f}, std: {log_q_z.std():.4f}")
        # print(f"  log_prop_z (mean): {log_prop_z.mean():.4f}, std: {log_prop_z.std():.4f}")
        # print(f"  log_weight (mean): {log_weight.mean():.4f}, std: {log_weight.std():.4f}")
        # print(f"  exp(log_weight) (mean): {torch.exp(log_weight.detach()).mean():.4e}, std: {torch.exp(log_weight.detach()).std():.4e}")
        # print(f"  log_p_x_given_z (mean): {log_p_x_given_z.mean():.4f}, std: {log_p_x_given_z.std():.4f}")
        # print(f"  Contribution (mean): {(torch.exp(log_weight.detach()) * log_p_x_given_z).mean():.4e}")
        # print(f"----------------------")

        # 7. Accumulate Weighted log p(X|z)
        # Using detached weights for stable loss estimate
        total_weighted_log_p_x_given_z += torch.exp(log_weight.detach()) * log_p_x_given_z

    # --- Average over samples and batch ---
    # E_prop[w * log p(X|z)] estimate for the batch
    mean_weighted_log_p_x_given_z = (total_weighted_log_p_x_given_z / n_samples).mean()

    # --- Calculate KL divergences for q (using q parameters directly) ---
    # KL(q(edge) || uniform)
    log_uniform = -math.log(n_edges)
    kl_p = (q_edge_probs_batch * (q_edge_probs_batch.clamp(min=eps).log() - log_uniform)).sum(dim=1).mean()

    # KL(q(t|edge) || Beta(1,1)) - Averaged over edges, weighted by q(edge)
    kl_t_per_edge = compute_kl_beta(alpha_q_batch, beta_q_batch, 1.0, 1.0, eps=eps) # [batch, E]
    kl_t = (q_edge_probs_batch.detach() * kl_t_per_edge).sum(dim=1).mean() # Weight KL(t|e) by q(e)

    # --- (Optional) Branching Entropy Term ---
    branch_entropy_term = 0.0
    if branch_entropy_weight > 0:
         branch_probs = posterior.compute_branch_probs() # [E]
         # Prevent log(0) - clamp probs slightly away from 0
         branch_probs_clamped = branch_probs.clamp(min=eps)
         entropy = - (branch_probs * torch.log(branch_probs_clamped)).sum()
         branch_entropy_term = - branch_entropy_weight * entropy # Add negative entropy to loss

    # --- Final Loss (Negative ELBO approximation) ---
    loss = -mean_weighted_log_p_x_given_z \
           + kl_p_weight * kl_p \
           + kl_weight * kl_t \
           + branch_entropy_term

    # --- Metrics for Logging ---
    all_log_weights_tensor = torch.stack(all_log_weights_list)
    metrics = {
        "loss": loss.item(), # Total loss
        "nll_weighted": (-mean_weighted_log_p_x_given_z).item(), # Weighted NLL part
        "kl_t": kl_t.item(),
        "kl_p": kl_p.item(),
        "log_weight_mean": all_log_weights_tensor.mean().item(),
        "log_weight_std": all_log_weights_tensor.std().item(),
        "branch_entropy": -branch_entropy_term.item() / branch_entropy_weight if branch_entropy_weight > 0 else 0.0
        # Add back other metrics if needed, but t_cont, emission_cont, transition are removed
    }

    # Note on gradients: The gradient calculation relies on autograd through log_q_z
    # (in log_weight) and the KL terms. It approximates the true IS gradient.

    return loss, metrics


compute_elbo_batch = compute_elbo
