import torch
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
    device = t.device
    index_to_edge = {v: k for k, v in edge_to_index.items()}

    penalties = []

    for i in range(edge_idx.shape[0]):
        u_name, v_name = index_to_edge[edge_idx[i].item()]

        # Penalize t ≈ 0 on child edge (if it has a parent)
        has_parent = any((x, u_name) in edge_to_index for x in traj.node_to_index)
        if has_parent:
            penalties.append(t[i] ** 2)

        # Penalize t ≈ 1 on parent edge (if it has children)
        has_child = any((v_name, x) in edge_to_index for x in traj.node_to_index)
        if has_child:
            penalties.append((1 - t[i]) ** 2)

    return torch.stack(penalties).mean() if penalties else t.new_zeros(1).squeeze()

def transition_log_prob(edge_idx, traj, edge_to_index):
    """
    Compute log transition probability log(A(edge)) for each sampled edge (u,v).
    A(edge) = branch_prob / Z_u
    """
    device = edge_idx.device
    n_edges = len(edge_to_index)
    A_tensor = edge_idx.new_zeros(n_edges)

    index_to_edge = {v: k for k, v in edge_to_index.items()}
    for i in range(n_edges):
        u, v = index_to_edge[i]
        B = traj.branch_probabilities.get((u, v), 1e-8)
        Z = traj.normalizing_constants.get(u, 1.0)
        A_tensor[i] = B / (Z + 1e-8)

    A_vals = A_tensor[edge_idx]
    return torch.log(A_vals.clamp(min=1e-8))

def resolve_edge_nodes(edge_idx, index_to_edge, node_to_index, device=None):
    """
    Given edge indices, return corresponding source and target node *indices*.
    Converts from edge_idx -> (u_name, v_name) -> u_idx, v_idx.
    """
    device = device or edge_idx.device
    u_names = [index_to_edge[i.item()][0] for i in edge_idx]
    v_names = [index_to_edge[i.item()][1] for i in edge_idx]

    u_idx = torch.tensor([node_to_index[u] for u in u_names], device=device)
    v_idx = torch.tensor([node_to_index[v] for v in v_names], device=device)
    return u_idx, v_idx



def compute_elbo(
    X, cell_indices, traj, posterior, edge_to_index,
    g, K, sigma2, belief_propagator=None, n_samples=1,
    kl_weight=1.0, kl_p_weight=1.0, t_cont_weight=1.0, pi=None,
    transition_weight=1.0, l1_weight=0.0, branch_entropy_weight=1.0
):
    device = X.device
    all_nll, all_kl_t, all_trans_logp = [], [], []

    q_e = torch.softmax(posterior.edge_logits[cell_indices], dim=1)

    # Use structure to bias posterior
    with torch.no_grad():
        A_prior, _ = posterior.compute_transition_probs()
    
    q_e = q_e @ A_prior  # Encourage edge selection via transition structure

    q_eff = belief_propagator.diffuse(q_e, alpha=0.5, steps=2) if belief_propagator else q_e

    index_to_edge = {v: k for k, v in edge_to_index.items()}
    node_index = traj.node_to_index
    A_probs, _ = posterior.compute_transition_probs()

    for _ in range(n_samples):
        edge_idx = torch.multinomial(q_eff, num_samples=1, replacement=True).squeeze(1)

        alpha = posterior.alpha[cell_indices, edge_idx].clamp(min=1e-6)
        beta = posterior.beta[cell_indices, edge_idx].clamp(min=1e-6)
        t = torch.distributions.Beta(alpha, beta).rsample()

        u_idx, v_idx = resolve_edge_nodes(edge_idx, index_to_edge, node_index)
        g_a = g[u_idx]
        g_b = g[v_idx]

        nll = emission_nll(
            X, edge_idx, t, g, K, sigma2,
            index_to_edge=index_to_edge,
            node_to_index=node_index,
            pi=pi
        )

        kl_t = compute_kl_beta(alpha, beta).mean()

        u_idx_clamped = u_idx.clamp(max=A_probs.shape[0] - 1)
        v_idx_clamped = v_idx.clamp(max=A_probs.shape[1] - 1)
        A_vals = A_probs[u_idx_clamped, v_idx_clamped].clamp(min=1e-8)
        trans_logp = torch.log(A_vals).mean()

        all_nll.append(nll)
        all_kl_t.append(kl_t)
        all_trans_logp.append(trans_logp)

    mean_nll = torch.stack(all_nll).mean()
    mean_kl_t = torch.stack(all_kl_t).mean()
    mean_trans = torch.stack(all_trans_logp).mean()

    edge_probs = torch.softmax(posterior.edge_logits, dim=1)[cell_indices]
    log_prior = torch.log_softmax(posterior.edge_logits, dim=1)[cell_indices]
    kl_p = (edge_probs * (edge_probs.clamp(min=1e-8).log() - log_prior)).sum(dim=1).mean()

    t_cont = continuity_penalty(edge_idx, t, traj, edge_to_index, node_index)

    node_index = traj.node_to_index
    l1 = sum(
        torch.nn.functional.l1_loss(g[node_index[u]], g[node_index[v]], reduction='sum')
        for u, v in traj.edge_list
    )

    l1_penalty = l1 / len(traj.edge_list)

    # Emission continuity penalty
    continuity_mse = []
    for (u_name, v_name), i in edge_to_index.items():
        v_idx = node_index[v_name]
        children = [(v_name, w) for w in node_index if (v_name, w) in edge_to_index]
        for (v, w) in children:
            w_idx = node_index[w]
            continuity_mse.append(torch.nn.functional.mse_loss(g[v_idx], g[w_idx], reduction='mean'))

    emission_cont = torch.stack(continuity_mse).mean() if continuity_mse else g.new_zeros(1).squeeze()

    # Branch entropy penalty
    B = posterior.compute_branch_probs()
    entropy = - (B * B.clamp(min=1e-8).log()).sum()

    elbo = -mean_nll \
           - kl_weight * mean_kl_t \
           - kl_p_weight * kl_p \
           - t_cont_weight * t_cont \
           + transition_weight * mean_trans \
           - l1_weight * l1_penalty \
           - emission_cont \
           + branch_entropy_weight * entropy

    return -elbo, {
        "nll": mean_nll.detach().item(),
        "kl_t": mean_kl_t.detach().item(),
        "kl_p": kl_p.detach().item(),
        "t_cont": t_cont.detach().item(),
        "transition": mean_trans.detach().item(),
        "l1": l1_penalty.detach().item(),
        "emission_cont": emission_cont.detach().item(),
        "entropy": entropy.detach().item(),
        "t": t.detach(),
        "q_eff": q_eff.detach(),
    }

compute_elbo_batch = compute_elbo
