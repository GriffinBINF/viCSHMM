import torch
from .posterior import TreeVariationalPosterior
from .belief import BeliefPropagator
from utils.constants import EPSILON

def compute_kl_beta(alpha_q, beta_q, alpha_p=1.0, beta_p=1.0, eps=EPSILON):
    """
    Computes KL( Beta(alpha_q, beta_q) || Beta(alpha_p, beta_p) ).
    Handles tensor inputs and clamps for stability.
    """
    # Ensure target parameters are tensors with matching device/dtype
    if not isinstance(alpha_p, torch.Tensor):
        alpha_p = torch.tensor(alpha_p, device=alpha_q.device, dtype=alpha_q.dtype)
    if not isinstance(beta_p, torch.Tensor):
        beta_p = torch.tensor(beta_p, device=beta_q.device, dtype=beta_q.dtype)

    # Clamp q parameters for stability before log/digamma
    alpha_q = alpha_q.clamp(min=eps)
    beta_q = beta_q.clamp(min=eps)

    # Log gamma terms
    lgamma_q_sum = torch.lgamma(alpha_q + beta_q)
    lgamma_q_alpha = torch.lgamma(alpha_q)
    lgamma_q_beta = torch.lgamma(beta_q)
    lgamma_p_sum = torch.lgamma(alpha_p + beta_p)
    lgamma_p_alpha = torch.lgamma(alpha_p)
    lgamma_p_beta = torch.lgamma(beta_p)

    # Digamma terms
    digamma_q_alpha = torch.digamma(alpha_q)
    digamma_q_beta = torch.digamma(beta_q)
    digamma_q_sum = torch.digamma(alpha_q + beta_q)

    # KL divergence formula components
    term1 = lgamma_q_sum - lgamma_q_alpha - lgamma_q_beta
    term2 = -lgamma_p_sum + lgamma_p_alpha + lgamma_p_beta
    term3 = (alpha_q - alpha_p) * (digamma_q_alpha - digamma_q_sum)
    term4 = (beta_q - beta_p) * (digamma_q_beta - digamma_q_sum)

    kl_div = term1 + term2 + term3 + term4

    # Handle potential NaNs/Infs arising from extreme parameter values after clamping
    kl_div = torch.nan_to_num(kl_div, nan=0.0, posinf=1e6, neginf=-1e6) # Use 0 for NaN, large value for Inf
    return kl_div.clamp(min=0) # KL should be non-negative

def compute_proposal_distribution(
    posterior: TreeVariationalPosterior,
    belief_propagator: BeliefPropagator,
    proposal_beta_const: float = 10.0,
    proposal_target_temp: float = 1.0,
    proposal_edge_temp: float = 1.0,
    diffusion_alpha: float = 0.5,
    diffusion_steps: int = 2,
    eps: float = EPSILON
):
    """
    Computes the parameters for the importance sampling proposal distribution
    p_proposal(edge, t) = p_prop(edge) * p_prop(t | edge).

    Returns:
        tuple: Contains:
            - prop_edge_probs (Tensor): [N_cells, N_edges]
            - prop_alpha (Tensor): [N_cells, N_edges]
            - prop_beta (Tensor): [N_cells, N_edges]
    """
    device = posterior.device
    n_cells = posterior.n_cells
    n_edges = posterior.n_edges

    with torch.no_grad():
        # --- 1. Proposal edge probabilities ---
        edge_logits = posterior.edge_logits.detach() / proposal_edge_temp
        q_edge_probs = torch.nn.functional.softmax(edge_logits, dim=1)
        prop_edge_probs = belief_propagator.diffuse(q_edge_probs, alpha=diffusion_alpha, steps=diffusion_steps)
        prop_edge_probs = prop_edge_probs / prop_edge_probs.sum(dim=1, keepdim=True).clamp(min=eps)

        # --- 2. Extract target edge per cell ---
        e_max_idx = q_edge_probs.argmax(dim=1)  # [N]

        alpha_q = posterior.alpha.detach()  # [N, E]
        beta_q = posterior.beta.detach()    # [N, E]

        # --- 3. Edge index to (u, v) map ---
        edge_index_to_uv = torch.tensor(posterior.edge_list, device=device)  # [E, 2]
        edge_u = edge_index_to_uv[:, 0]  # [E]
        edge_v = edge_index_to_uv[:, 1]  # [E]

        e_max_uv = edge_index_to_uv[e_max_idx]  # [N, 2]
        u_max = e_max_uv[:, 0].unsqueeze(1)  # [N, 1]
        v_max = e_max_uv[:, 1].unsqueeze(1)  # [N, 1]

        # --- 4. Create proposal time distributions ---
        prop_alpha = torch.ones((n_cells, n_edges), device=device)
        prop_beta = torch.ones((n_cells, n_edges), device=device)

        edge_indices = torch.arange(n_edges, device=device).unsqueeze(0)  # [1, E]
        e_max_idx_exp = e_max_idx.unsqueeze(1)  # [N, 1]

        is_target_edge = edge_indices == e_max_idx_exp  # [N, E]
        is_child_edge = edge_u.unsqueeze(0) == v_max    # [N, E]
        is_parent_edge = edge_v.unsqueeze(0) == u_max   # [N, E]

        # Target: broadened q(t)
        target_alpha = (alpha_q[torch.arange(n_cells), e_max_idx] / proposal_target_temp).clamp(min=eps)
        target_beta = (beta_q[torch.arange(n_cells), e_max_idx] / proposal_target_temp).clamp(min=eps)

        prop_alpha = torch.where(is_target_edge, target_alpha.unsqueeze(1), prop_alpha)
        prop_beta = torch.where(is_target_edge, target_beta.unsqueeze(1), prop_beta)

        # Children: Beta(1, B)
        prop_alpha = torch.where(is_child_edge, torch.tensor(1.0, device=device), prop_alpha)
        prop_beta = torch.where(is_child_edge, torch.tensor(proposal_beta_const, device=device), prop_beta)

        # Parents: Beta(B, 1)
        prop_alpha = torch.where(is_parent_edge, torch.tensor(proposal_beta_const, device=device), prop_alpha)
        prop_beta = torch.where(is_parent_edge, torch.tensor(1.0, device=device), prop_beta)

        prop_alpha = prop_alpha.clamp(min=eps)
        prop_beta = prop_beta.clamp(min=eps)

    return prop_edge_probs, prop_alpha, prop_beta