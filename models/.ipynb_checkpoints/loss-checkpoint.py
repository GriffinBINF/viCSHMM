import torch
from models.emission import emission_nll


def compute_kl_beta(alpha, beta, eps=1e-8):
    return (
        (alpha - 1) * (torch.digamma(alpha + eps) - torch.digamma(alpha + beta + eps)) +
        (beta - 1) * (torch.digamma(beta + eps) - torch.digamma(alpha + beta + eps)) +
        torch.lgamma(alpha + beta + eps) -
        torch.lgamma(alpha + eps) -
        torch.lgamma(beta + eps)
    )


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
    device = device or edge_idx_tensor.device
    u_idx = [index_to_edge_tuple_map[i.item()][0] for i in edge_idx_tensor]
    v_idx = [index_to_edge_tuple_map[i.item()][1] for i in edge_idx_tensor]
    return torch.tensor(u_idx, device=device), torch.tensor(v_idx, device=device)


def compute_elbo(
    X, cell_indices, traj, posterior, edge_tuple_to_index,
    g, K, sigma2,
    belief_propagator=None, n_samples=1,
    kl_weight=1.0, kl_p_weight=1.0, t_cont_weight=1.0, pi=None,
    transition_weight=1.0, l1_weight=0.0, branch_entropy_weight=1.0,
    tau=1.0  
):
    device = X.device
    all_nll, all_kl_t, all_trans_logp = [], [], []

    q_e = torch.softmax(posterior.edge_logits[cell_indices], dim=1)
    q_eff = belief_propagator.diffuse(q_e, alpha=0.5, steps=2) if belief_propagator else q_e

    index_to_edge_tuple = {v: k for k, v in edge_tuple_to_index.items()}
    index_to_edge_name = posterior.index_to_edge
    node_to_index = traj.node_to_index
    A_probs, _ = posterior.compute_transition_probs()

    for _ in range(n_samples):
        edge_logits_batch = posterior.edge_logits[cell_indices]
        edge_one_hot = torch.nn.functional.gumbel_softmax(edge_logits_batch, tau=tau, hard=True)
        edge_idx = edge_one_hot.argmax(dim=1)
        alpha = posterior.alpha[cell_indices, edge_idx].clamp(min=1e-6)
        beta = posterior.beta[cell_indices, edge_idx].clamp(min=1e-6)
        t = torch.distributions.Beta(alpha, beta).rsample()

        nll = emission_nll(
            X, edge_idx, t, g, K, sigma2,
            index_to_edge=index_to_edge_name,
            node_to_index=node_to_index,
            pi=pi
        )

        kl_t = compute_kl_beta(alpha, beta).mean()
        u_idx, v_idx = resolve_edge_nodes(edge_idx, index_to_edge_tuple)
        trans_logp = torch.log(posterior.compute_branch_probs()[edge_idx].clamp(min=1e-8)).mean()

        all_nll.append(nll)
        all_kl_t.append(kl_t)
        all_trans_logp.append(trans_logp)

    mean_nll = torch.stack(all_nll).mean()
    mean_kl_t = torch.stack(all_kl_t).mean()
    mean_trans = torch.stack(all_trans_logp).mean()

    edge_probs = q_eff
    n_edges = posterior.n_edges
    log_uniform = torch.log(torch.tensor(1.0 / n_edges, device=device))
    kl_p = (edge_probs * (edge_probs.clamp(min=1e-8).log() - log_uniform)).sum(dim=1).mean()

    t_cont = continuity_penalty(edge_idx, t, traj, index_to_edge_tuple, node_to_index)

    continuity_mse = []
    for parent_idx, (u_idx, v_idx) in index_to_edge_tuple.items():
        children = [ci for ci, (cu, _) in index_to_edge_tuple.items() if cu == v_idx]
        for child_idx in children:
            w_idx = index_to_edge_tuple[child_idx][1]
            continuity_mse.append(torch.nn.functional.mse_loss(g[v_idx], g[w_idx], reduction='mean'))

    emission_cont = torch.stack(continuity_mse).mean() if continuity_mse else g.new_zeros(1).squeeze()

    entropy = - (posterior.compute_branch_probs() * posterior.compute_branch_probs().clamp(min=1e-8).log()).sum()

    loss = mean_nll \
           + kl_weight * mean_kl_t \
           + kl_p_weight * kl_p \
           - transition_weight * mean_trans \
           + t_cont_weight * t_cont \
           + l1_weight * torch.tensor(0.0, device=device) \
           + emission_cont \
           - branch_entropy_weight * entropy

    return loss, {
        "nll": mean_nll.item(),
        "kl_t": mean_kl_t.item(),
        "kl_p": kl_p.item(),
        "t_cont": t_cont.item(),
        "transition": mean_trans.item(),
        "l1": 0.0,
        "emission_cont": emission_cont.item(),
        "entropy": entropy.item(),
        "t": t.detach(),
        "q_eff": q_eff.detach(),
    }


compute_elbo_batch = compute_elbo
