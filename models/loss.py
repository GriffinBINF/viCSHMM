import torch
import math
from models.emission import emission_nll
from utils.constants import EPSILON, EPSILON_LOG # Use constants

def compute_kl_beta(alpha, beta):
    """KL(Beta(α, β) || Uniform(0,1)) for each (α, β) pair"""
    # Add eps inside operations for stability
    alpha_c = alpha.clamp(min=EPSILON)
    beta_c = beta.clamp(min=EPSILON)
    digamma_alpha = torch.digamma(alpha_c)
    digamma_beta = torch.digamma(beta_c)
    digamma_alpha_beta = torch.digamma(alpha_c + beta_c)
    lgamma_alpha = torch.lgamma(alpha_c)
    lgamma_beta = torch.lgamma(beta_c)
    lgamma_alpha_beta = torch.lgamma(alpha_c + beta_c)

    kl = (
        (alpha_c - 1) * (digamma_alpha - digamma_alpha_beta) +
        (beta_c - 1) * (digamma_beta - digamma_alpha_beta) +
        lgamma_alpha_beta - lgamma_alpha - lgamma_beta
    )
    # Clamp KL to be non-negative
    return kl.clamp(min=0.0)


def continuity_penalty(edge_idx, t, traj, edge_to_index, node_to_index):
    """ Penalizes t near 0 for child edges and t near 1 for parent edges. """
    device = t.device
    index_to_edge = {v: k for k, v in edge_to_index.items()}
    G_traj = traj.G_traj # Use the networkx graph

    penalties = []
    for i in range(edge_idx.shape[0]):
        edge_idx_item = edge_idx[i].item()
        try:
            u_name, v_name = index_to_edge[edge_idx_item]
        except KeyError:
            # print(f"Warning: Edge index {edge_idx_item} not in index_to_edge for continuity penalty.")
            continue # Skip if edge index is invalid

        # Check if nodes exist in the graph
        if u_name not in G_traj or v_name not in G_traj:
             # print(f"Warning: Node {u_name} or {v_name} not in graph for continuity penalty.")
             continue

        # Penalize t ≈ 0 if u_name has predecessors (is a child)
        # Check predecessors in the actual graph G_traj
        if G_traj.in_degree(u_name) > 0:
            # Check if *any* predecessor edge exists in edge_to_index
            has_parent_in_model = any((pred, u_name) in edge_to_index for pred in G_traj.predecessors(u_name))
            if has_parent_in_model:
                penalties.append(t[i] ** 2)

        # Penalize t ≈ 1 if v_name has successors (is a parent)
        # Check successors in the actual graph G_traj
        if G_traj.out_degree(v_name) > 0:
             # Check if *any* successor edge exists in edge_to_index
            has_child_in_model = any((v_name, succ) in edge_to_index for succ in G_traj.successors(v_name))
            if has_child_in_model:
                penalties.append((1 - t[i]) ** 2)

    if not penalties:
        return torch.tensor(0.0, device=device)

    # Return mean, handle potential NaN/inf from t? t should be in [0,1] from Beta.rsample
    return torch.stack(penalties).mean()


def resolve_edge_nodes(edge_idx, index_to_edge, node_to_index, device=None):
    """
    Given edge indices, return corresponding source and target node *indices*.
    Converts from edge_idx -> (u_name, v_name) -> u_idx, v_idx.
    Handles potential errors gracefully.
    """
    device = device or edge_idx.device
    u_names, v_names = [], []
    valid_mask = []
    for i in range(edge_idx.shape[0]):
        idx_item = edge_idx[i].item()
        edge = index_to_edge.get(idx_item)
        if edge:
            u_name, v_name = edge
            if u_name in node_to_index and v_name in node_to_index:
                u_names.append(u_name)
                v_names.append(v_name)
                valid_mask.append(True)
            else:
                valid_mask.append(False)
        else:
            valid_mask.append(False)

    if not any(valid_mask): # All failed
        # Return empty tensors or tensors of zeros? Let's return empty.
        # print("Warning: resolve_edge_nodes failed for all input indices.")
        return (torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor(valid_mask, dtype=torch.bool, device=device))


    u_indices = torch.tensor([node_to_index[u] for u, m in zip(u_names, valid_mask) if m], dtype=torch.long, device=device)
    v_indices = torch.tensor([node_to_index[v] for v, m in zip(v_names, valid_mask) if m], dtype=torch.long, device=device)
    return u_indices, v_indices, torch.tensor(valid_mask, dtype=torch.bool, device=device)


# Renamed compute_elbo to compute_elbo_batch for clarity
def compute_elbo_batch(
    X, cell_indices, traj, posterior, edge_to_index,
    g, K, sigma2, belief_propagator=None, n_samples=1,
    kl_weight=1.0, kl_p_weight=1.0, t_cont_weight=1.0, pi=None,
    branch_entropy_weight=1.0, # Removed transition_weight, l1_weight (handled in VIRunner)
    temperature=1.0, # Added temperature
    struct_prior_weight=0.0 # Added structural prior weight
):
    """
    Computes the Evidence Lower Bound (ELBO) for a batch of cells.

    Args:
        X (Tensor): Expression data for the batch [batch_size, n_genes].
        cell_indices (Tensor): Original indices of the cells in the batch [batch_size].
        traj (TrajectoryGraph): The trajectory graph object.
        posterior (TreeVariationalPosterior): The posterior model instance.
        edge_to_index (dict): Mapping from edge tuple (u_name, v_name) to integer index.
        g (Tensor): Node expression parameters [n_nodes, n_genes].
        K (Tensor): Edge rate parameters [n_edges, n_genes].
        sigma2 (Tensor): Edge variance parameters [n_edges, n_genes].
        belief_propagator (BeliefPropagator, optional): For diffusing edge probabilities. Defaults to None.
        n_samples (int, optional): Number of Monte Carlo samples for latent vars. Defaults to 1.
        kl_weight (float, optional): Weight for KL divergence of time. Defaults to 1.0.
        kl_p_weight (float, optional): Weight for KL divergence of edge assignment. Defaults to 1.0.
        t_cont_weight (float, optional): Weight for time continuity penalty. Defaults to 1.0.
        pi (Tensor, optional): Dropout probabilities [n_edges, n_genes]. Defaults to None.
        branch_entropy_weight (float, optional): Weight for branch probability entropy regularization. Defaults to 1.0.
        temperature (float, optional): Temperature for softmax scaling of edge logits. Defaults to 1.0.
        struct_prior_weight (float, optional): Weight for structural KL divergence against prior A matrix. Defaults to 0.0.


    Returns:
        tuple: (elbo_value (scalar Tensor), metrics_dict (dict))
    """
    device = X.device
    batch_size = X.shape[0]
    if batch_size == 0: # Handle empty batch
        return torch.tensor(0.0, device=device), {}

    all_nll, all_kl_t, all_t_cont = [], [], []

    # 1. Calculate Edge Probabilities (q_eff)
    # Use original cell indices to get logits from the full posterior object
    raw_logits = posterior.edge_logits[cell_indices]
    tempered_logits = raw_logits / temperature
    q_e_raw = torch.softmax(tempered_logits, dim=1)

    # Apply optional belief propagation using structural info from posterior
    q_eff = q_e_raw # Default if no propagator
    if belief_propagator:
        # Belief propagator uses posterior internal A matrix derived from branch logits
        # alpha=0.5 means 50% original, 50% diffused; steps=2 applies diffusion twice
        q_eff = belief_propagator.diffuse(q_e_raw, alpha=0.5, steps=2)

    # Ensure probabilities sum to 1 and handle potential NaNs
    q_eff = q_eff / q_eff.sum(dim=1, keepdim=True).clamp(min=EPSILON) # Normalize
    q_eff = torch.nan_to_num(q_eff, nan=0.0) # Replace NaNs with 0 probability
    q_eff = q_eff.clamp(min=EPSILON_LOG) # Clamp slightly above zero for sampling/log

    index_to_edge = posterior.index_to_edge # Use mapping from posterior
    node_index = posterior.traj.node_to_index

    # 2. Sample Edges and Times, Calculate NLL and KL(t)
    sampled_t_all = [] # Store t for continuity penalty calculation later
    sampled_edge_idx_all = [] # Store edge_idx for continuity penalty

    for s in range(n_samples):
        # Sample edge index P_idx ~ Cat(q_eff)
        try:
            edge_idx_s = torch.multinomial(q_eff, num_samples=1, replacement=True).squeeze(1) # [batch_size]
        except RuntimeError as e:
             # This can happen if q_eff sums to zero or has NaNs/Infs despite checks
             print(f"Error during multinomial sampling (sample {s+1}/{n_samples}): {e}")
             print(f"q_eff sample sum: {q_eff.sum(dim=1).min()} - {q_eff.sum(dim=1).max()}")
             print(f"q_eff has NaNs: {torch.isnan(q_eff).any()}, Infs: {torch.isinf(q_eff).any()}")
             # Fallback: Assign to most likely edge? Or skip sample? Let's skip.
             continue # Skip this sample if sampling fails

        # Gather corresponding alpha, beta for the sampled edges
        # Use gather to select params based on sampled edge_idx_s for each cell
        alpha_s = posterior.alpha[cell_indices].gather(1, edge_idx_s.unsqueeze(1)).squeeze(1).clamp(min=EPSILON)
        beta_s = posterior.beta[cell_indices].gather(1, edge_idx_s.unsqueeze(1)).squeeze(1).clamp(min=EPSILON)

        # Sample time t ~ Beta(alpha_s, beta_s) using reparameterization trick
        try:
             # Use rsample for gradient flow
             t_s = torch.distributions.Beta(alpha_s, beta_s).rsample().clamp(min=EPSILON, max=1.0-EPSILON)
        except ValueError as e:
             print(f"Error during Beta sampling (sample {s+1}/{n_samples}): {e}")
             print(f"Alpha sample: min={alpha_s.min()}, max={alpha_s.max()}")
             print(f"Beta sample: min={beta_s.min()}, max={beta_s.max()}")
             # Fallback: sample from uniform? Or skip sample? Let's skip.
             continue # Skip this sample if sampling fails

        # Calculate NLL: -log p(X | P_idx=edge_idx_s, T=t_s)
        # Note: emission_nll averages over the batch internally
        nll_s = emission_nll(X, edge_idx_s, t_s, g, K, sigma2, index_to_edge, node_index, pi=pi)

        # Calculate KL(t): KL( Beta(alpha_s, beta_s) || Uniform(0,1) )
        # compute_kl_beta operates element-wise, then we average over the batch
        kl_t_s = compute_kl_beta(alpha_s, beta_s).mean()

        # Calculate time continuity penalty for this sample
        t_cont_s = continuity_penalty(edge_idx_s, t_s, traj, edge_to_index, node_index)

        # Accumulate results for this sample
        all_nll.append(nll_s)
        all_kl_t.append(kl_t_s)
        all_t_cont.append(t_cont_s)

        # Store sampled values (only need one set for KL(p) and continuity penalty averaging)
        if s == 0:
            sampled_t_all = t_s.detach() # Detach as penalty shouldn't drive posterior t directly
            sampled_edge_idx_all = edge_idx_s.detach() # Detach


    # Average results over samples (if any samples were successful)
    if not all_nll: # Check if any samples were processed
        print("Warning: No valid samples were processed in compute_elbo_batch. Returning ELBO=0.")
        # Need to return valid tensors even if zero
        return torch.tensor(0.0, device=device), {
             "nll": 0.0, "kl_t": 0.0, "kl_p": 0.0, "t_cont": 0.0,
             "branch_entropy": 0.0, "struct_kl": 0.0,
             "q_e_raw": torch.zeros_like(q_e_raw), "q_eff": torch.zeros_like(q_eff)
        }

    mean_nll = torch.stack(all_nll).mean()
    mean_kl_t = torch.stack(all_kl_t).mean()
    mean_t_cont = torch.stack(all_t_cont).mean() # Average the continuity penalty across samples

    # 3. Calculate KL(p): KL divergence for edge assignment probability q_eff
    # KL( Cat(q_eff) || Cat(p_prior) )
    # Here, p_prior is implicitly uniform over edges if not specified,
    # or could be derived from the structural prior A_prior.
    # The original code uses: sum_e q_eff[e] * log(q_eff[e] / p_prior[e])
    # Often simplified to: sum_e q_eff[e] * log(q_eff[e]) - sum_e q_eff[e] * log(p_prior[e])
    # If p_prior is uniform (1/E), log(p_prior) = -log(E), so KL = sum(q*log(q)) + log(E)
    # The implementation in the original code seems to calculate KL against the raw tempered logits:
    # (q_eff * (log q_eff - log q_e_raw)).sum().mean() -> This is KL(q_eff || q_e_raw)

    # Let's calculate KL( q_eff || uniform ) = Entropy(uniform) - CrossEntropy(uniform, q_eff)
    # Entropy(uniform) = log(n_edges)
    # CrossEntropy = - sum_e (1/n_edges) * log(q_eff[e])
    # KL = log(n_edges) + (1/n_edges) * sum_e log(q_eff[e])  -- This doesn't seem right.

    # Let's calculate KL( q_eff || q_e_raw ), average over batch
    log_q_eff = torch.log(q_eff.clamp(min=EPSILON_LOG))
    log_q_e_raw = torch.log(q_e_raw.clamp(min=EPSILON_LOG))
    # KL divergence term per cell: sum_edges [ q_eff * (log_q_eff - log_q_e_raw) ]
    kl_p = (q_eff * (log_q_eff - log_q_e_raw)).sum(dim=1).mean() # Average KL over the batch

    # 4. Calculate Branch Entropy Regularization
    # Compute branch probabilities B from posterior's branch_logits
    B = posterior.compute_branch_probs() # Tensor[n_edges] - probabilities for outgoing edges at splits
    # Calculate entropy: - sum_{e where B[e]>0} B[e] * log(B[e])
    # Need to mask B for non-zero values to avoid log(0)
    B_nz = B[B > EPSILON_LOG] # Select non-zero probabilities
    branch_entropy = - (B_nz * torch.log(B_nz)).sum() if B_nz.numel() > 0 else torch.tensor(0.0, device=device)

    # 5. Calculate Structural KL Divergence (Optional)
    # KL( A_eff || A_prior ) where A_eff is derived from q_eff / belief propagation?
    # Or more simply: KL divergence between the inferred transition matrix A (from posterior branch probs)
    # and a prior transition matrix A_prior (e.g., from PAGA).
    # Let's use KL( q_e_raw || A_prior_row ) for each cell, averaged? This seems complex.

    # Simpler approach: Penalize divergence between posterior's A and a fixed prior A.
    struct_kl = torch.tensor(0.0, device=device)
    if struct_prior_weight > 0:
         # We need A_prior. Let's assume it's stored or computed in the posterior or traj object.
         # For now, let's *compute* the posterior's A matrix and assume a simple prior (e.g., uniform at splits).
         A_posterior, _ = posterior.compute_transition_probs() # [E, E] derived from learned branch_logits

         # Define a simple prior A_prior: e.g., uniform transitions at splits, 1.0 otherwise?
         # This needs a proper definition based on the desired structural bias.
         # Placeholder: Use the initial transition probabilities from the graph as prior
         A_prior = torch.zeros_like(A_posterior)
         for i in range(posterior.n_edges):
              u, v = posterior.index_to_edge[i]
              # Get initial probability from traj graph if available
              prob = traj.transition_probabilities.get((u,v), 0.0) # Using original names
              # Need to map this back to the posterior's edge indices... complex.

         # Alternative: Calculate KL divergence between q_e_raw and the prior transition probabilities
         # For each cell, find its most likely edge (u, v). Calculate KL between q_e_raw[cell, next_edges]
         # and A_prior[edge_uv_idx, next_edges]. This is also complex.

         # Simplest meaningful penalty: KL divergence between the *global* branch probabilities B
         # inferred by the posterior and some prior branch probabilities B_prior.
         # Let B_prior be uniform at splits (1/num_children).
         B_prior = torch.zeros_like(B)
         for parent_node, child_edges in posterior.children_map.items():
              if len(child_edges) > 0:
                   child_idxs = torch.tensor([posterior.edge_to_index[edge] for edge in child_edges], device=device)
                   prior_prob = 1.0 / len(child_edges)
                   B_prior[child_idxs] = prior_prob

         # Calculate KL( B || B_prior ) where B > 0 and B_prior > 0
         mask = (B > EPSILON_LOG) & (B_prior > EPSILON_LOG)
         if torch.any(mask):
              struct_kl = (B[mask] * (torch.log(B[mask]) - torch.log(B_prior[mask]))).sum()
         else:
              struct_kl = torch.tensor(0.0, device=device)


    # 6. Combine terms for ELBO
    elbo = (-mean_nll                            # Reconstruction term
            - kl_weight * mean_kl_t              # KL divergence for time t
            - kl_p_weight * kl_p                 # KL divergence for edge assignment p
            - t_cont_weight * mean_t_cont        # Time continuity penalty
            + branch_entropy_weight * branch_entropy # Regularization: Encourage diverse branches
            - struct_prior_weight * struct_kl    # Structural prior regularization
            )

    # Prepare metrics dictionary
    metrics = {
        "nll": mean_nll.item(),
        "kl_t": mean_kl_t.item(),
        "kl_p": kl_p.item(),
        "t_cont": mean_t_cont.item(),
        "branch_entropy": branch_entropy.item(),
        "struct_kl": struct_kl.item(),
        # Include q_eff and q_e_raw for potential logging/analysis (detached)
        "q_e_raw": q_e_raw.detach(),
        "q_eff": q_eff.detach(),
        # Include sampled t and edge_idx from the first sample for visualization/debugging
        "t_sampled": sampled_t_all,
        "edge_idx_sampled": sampled_edge_idx_all
    }

    return elbo, metrics

# Keep compute_elbo alias if needed elsewhere, but prefer compute_elbo_batch
compute_elbo = compute_elbo_batch