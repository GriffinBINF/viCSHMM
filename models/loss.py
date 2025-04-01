import torch
import numpy as np
from models.emission import emission_nll

def compute_kl_beta(alpha, beta, eps=1e-8):
    """KL(Beta(α, β) || Uniform(0,1)) for each (α, β) pair"""
    return (
        (alpha - 1) * (torch.digamma(alpha + eps) - torch.digamma(alpha + beta + eps)) +
        (beta - 1) * (torch.digamma(beta + eps) - torch.digamma(alpha + beta + eps)) +
        torch.lgamma(alpha + beta + eps) -
        torch.lgamma(alpha + eps) -
        torch.lgamma(beta + eps)
    )


def continuity_penalty(edge_idx, t, traj, edge_to_index, node_to_index):
    """
    Penalizes time discontinuity across edges in the trajectory tree.

    If a sampled edge (u, v) is a child of another edge (x, u), then
    we assume t=1 on (x, u) should match t=0 on (u, v), and penalize deviations.

    So, if a cell is on (u, v) at t = ε, we penalize (ε)^2.
    """
    penalties = []
    index_to_edge = {v: k for k, v in edge_to_index.items()}

    for i in range(edge_idx.shape[0]):
        edge = index_to_edge[edge_idx[i].item()]
        parent_node = edge[0]

        # Find all parent edges ending at this node
        parent_edges = [
            (u, v) for (u, v) in traj.G_traj.edges()
            if v == parent_node and (u, v) in edge_to_index
        ]

        if not parent_edges:
            continue  # Root edge — no penalty

        # Assume time was 1.0 on parent, now t[i] on child → penalize (1 - t)^2
        penalties.append((t[i]) ** 2)

    return torch.stack(penalties).mean() if penalties else torch.tensor(0.0, device=t.device)

def transition_log_prob(edge_idx, traj, edge_to_index):
    """
    Get log transition probability for each sampled edge (u,v)
    as log(A(s_{u,t}, s_{v,t'})) = log(branch_prob / Z_u)
    """
    log_probs = []
    index_to_edge = {v: k for k, v in edge_to_index.items()}

    for idx in edge_idx:
        u, v = index_to_edge[int(idx.item())]
        branch_prob = traj.branch_probabilities.get((u, v), 1e-8)
        Z_u = traj.normalizing_constants.get(u, 1.0)
        A = branch_prob / Z_u
        log_probs.append(np.log(A + 1e-8))
    return torch.tensor(log_probs, device=edge_idx.device)

def resolve_edge_nodes(edge_idx, index_to_edge, node_to_index, device=None):
    """
    Given edge indices, return corresponding source and target node indices.

    Args:
        edge_idx: Tensor of edge indices [B]
        index_to_edge: dict mapping edge index → (u_name, v_name)
        node_to_index: dict mapping node name → node index
        device: torch.device (optional)

    Returns:
        u_indices: Tensor of source node indices [B]
        v_indices: Tensor of target node indices [B]
    """
    u_names = [index_to_edge[int(e)][0] for e in edge_idx]
    v_names = [index_to_edge[int(e)][1] for e in edge_idx]
    u_indices = torch.tensor([node_to_index[u] for u in u_names], device=device or edge_idx.device)
    v_indices = torch.tensor([node_to_index[v] for v in v_names], device=device or edge_idx.device)
    return u_indices, v_indices



def compute_elbo(
    X, cell_indices, traj, posterior, edge_to_index,
    g, K, sigma2, belief_propagator=None, n_samples=1,
    kl_weight=1.0, kl_p_weight=1.0, t_cont_weight=1.0, pi=None,
    transition_weight=1.0, l1_weight=0.0
):
    B = X.shape[0]
    device = X.device
    all_nll, all_kl_t, all_trans_logp = [], [], []

    # Softmax over edge logits to get variational q(edge)
    q_e = torch.softmax(posterior.edge_logits[cell_indices], dim=1)
    q_eff = belief_propagator.diffuse(q_e, alpha=0.5, steps=2) if belief_propagator else q_e

    index_to_edge = {v: k for k, v in edge_to_index.items()}
    node_index = traj.node_to_index

    # === Retrieve updated transition probabilities from variational model ===
    # This provides: A_probs[(u_idx, v_idx)] = B_uv / Z_u
    A_probs, Z_map = posterior.compute_transition_probs()

    for _ in range(n_samples):
        # Sample edge and time from variational posterior
        assert torch.all(torch.isfinite(q_eff)), "Non-finite values in q_eff!"
        assert torch.all(q_eff.sum(dim=1) > 0), "Some rows of q_eff sum to 0!"
        edge_idx = torch.multinomial(q_eff, num_samples=1, replacement=True).squeeze(1)
        assert edge_idx.min() >= 0, "Negative edge_idx!"
        assert edge_idx.max() < posterior.n_edges, f"Invalid edge_idx! Got max {edge_idx.max().item()} for {posterior.n_edges} edges"
    
        for idx in edge_idx:
            assert int(idx.item()) in index_to_edge, f"edge_idx {int(idx.item())} not in index_to_edge"
    
        # --- Emission parameter shape checks ---
        assert K.shape[0] == posterior.n_edges, f"K.shape[0] = {K.shape[0]}, expected {posterior.n_edges}"
        assert sigma2.shape[0] == posterior.n_edges, f"sigma2.shape[0] = {sigma2.shape[0]}, expected {posterior.n_edges}"
        assert g.shape[0] == len(traj.node_to_index), f"g.shape[0] = {g.shape[0]}, expected {len(traj.node_to_index)}"
    
        alpha = posterior.alpha[cell_indices, edge_idx].clamp(min=1e-6)
        beta = posterior.beta[cell_indices, edge_idx].clamp(min=1e-6)
        t = torch.distributions.Beta(alpha, beta).rsample()

        # Resolve g_a, g_b for each edge from node expression
        u_idx, v_idx = resolve_edge_nodes(edge_idx, index_to_edge, node_index)
        g_a = g[u_idx]
        g_b = g[v_idx]

    

        # Emission negative log-likelihood
        nll = emission_nll(
            X, edge_idx, t, g, K, sigma2,
            index_to_edge=index_to_edge,
            node_to_index=traj.node_to_index,
            pi=pi if pi is not None else None
        )


        # KL divergence of Beta distribution
        kl_t = compute_kl_beta(alpha, beta).mean()

        # Transition log-probability: log A(edge)
        trans_logp = []
        for idx in edge_idx:
            u, v = index_to_edge[int(idx.item())]
            u_idx = traj.node_to_index[u]
            v_idx = traj.node_to_index[v]
            A_val = A_probs.get((u_idx, v_idx), 1e-8)
            trans_logp.append(torch.log(torch.tensor(A_val + 1e-8, device=device)))
        trans_logp = torch.stack(trans_logp).mean()

        all_nll.append(nll)
        all_kl_t.append(kl_t)
        all_trans_logp.append(trans_logp)

    # === Aggregate and compute final terms ===
    mean_nll = torch.stack(all_nll).mean()
    mean_kl_t = torch.stack(all_kl_t).mean()
    mean_trans = torch.stack(all_trans_logp).mean()

    # KL(p) between variational edge assignment and prior
    edge_probs = torch.softmax(posterior.edge_logits[cell_indices], dim=1)
    log_prior = posterior.edge_logits.detach().log_softmax(dim=1)[cell_indices]
    kl_p = (edge_probs * (edge_probs.clamp(min=1e-8).log() - log_prior)).sum(dim=1).mean()

    # Temporal continuity penalty for t at branch points
    t_cont = continuity_penalty(edge_idx, t, traj, edge_to_index, traj.node_to_index)
    
    # === L1 penalty on expression change across edges ===
    l1 = 0.0
    for u_idx, v_idx in traj.edge_list:
        l1 += torch.nn.functional.l1_loss(g[u_idx], g[v_idx], reduction='sum')
    l1_penalty = l1 / len(traj.edge_list)

    # === Final ELBO objective ===
    # Note: Transition log-probs are maximized, so we add them (not subtract)
    elbo = -mean_nll - kl_weight * mean_kl_t - kl_p_weight * kl_p \
           - t_cont_weight * t_cont + transition_weight * mean_trans \
           - l1_weight * l1_penalty

    return -elbo, {
        "nll": mean_nll.item(),
        "kl_t": mean_kl_t.item(),
        "kl_p": kl_p.item(),
        "t_cont": t_cont.item(),
        "transition": mean_trans.item(),
        "l1": l1_penalty.item(),
        "t": t.detach(),
        "q_eff": q_eff.detach(),
    }
compute_elbo_batch = compute_elbo





