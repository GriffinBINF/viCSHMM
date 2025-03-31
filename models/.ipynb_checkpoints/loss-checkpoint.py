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


def continuity_penalty(path_idx, t, traj, posterior):
    penalties = []
    for i in range(len(path_idx)):
        p = posterior.index_to_path[path_idx[i].item()]
        edge_list = posterior.paths[p]
        if len(edge_list) < 2:
            continue
        parent_node = edge_list[-2][0]
        child_node = edge_list[-1][1]
        parent_level = traj.levels.get(parent_node.split('_')[-1], 0)
        child_level = traj.levels.get(child_node.split('_')[-1], 0)
        if child_level > parent_level:
            penalties.append((1.0 - t[i]) ** 2)
        else:
            penalties.append(t[i] ** 2)
    return torch.stack(penalties).mean() if penalties else torch.tensor(0.0, device=t.device)


def compute_elbo(
    X, traj, posterior, edge_to_index,
    g_a, g_b, K, sigma2, pi,
    belief_propagator=None, n_samples=1,
    kl_weight=1.0, kl_p_weight=1.0, t_cont_weight=1.0
):
    N, G = X.shape
    device = X.device
    q_p = torch.softmax(posterior.path_logits, dim=1)  # [N, P]

    q_eff = belief_propagator.diffuse(q_p, alpha=0.5, steps=2) if belief_propagator else q_p
    with torch.no_grad():
        path_priors = posterior.compute_path_probs().clamp(min=1e-6)
    kl_p = (q_p * (q_p.clamp(min=1e-6).log() - path_priors.log())).sum(dim=1).mean()

    all_nll, all_kl_t, all_t_cont = [], [], []
    for p in range(q_eff.shape[1]):
        alpha = posterior.alpha[:, p].clamp(min=1e-6)
        beta = posterior.beta[:, p].clamp(min=1e-6)
        t = torch.distributions.Beta(alpha, beta).rsample()
        path_idx = torch.full((N,), p, device=device, dtype=torch.long)
        nll = emission_nll(X, path_idx, t, g_a, g_b, K, sigma2, pi=pi)
        kl_t = compute_kl_beta(alpha, beta)
        t_penalty = continuity_penalty(path_idx, t, traj, posterior)

        all_nll.append(q_eff[:, p] * nll)
        all_kl_t.append(q_eff[:, p] * kl_t)
        all_t_cont.append(t_penalty)

    mean_nll = torch.stack(all_nll, dim=0).sum(dim=0).mean()
    mean_kl_t = torch.stack(all_kl_t, dim=0).sum(dim=0).mean()
    mean_t_cont = torch.stack(all_t_cont).mean()

    elbo = -mean_nll - kl_weight * mean_kl_t - kl_p_weight * kl_p - t_cont_weight * mean_t_cont
    return -elbo, {
        "nll": mean_nll.item(),
        "kl_t": mean_kl_t.item(),
        "kl_p": kl_p.item(),
        "t_cont": mean_t_cont.item(),
        "q_eff": q_eff.detach()
    }


def compute_elbo_batch(
    X_batch, cell_indices, traj, posterior, edge_to_index,
    g_a, g_b, K, sigma2, belief_propagator=None, n_samples=1,
    kl_weight=1.0, kl_p_weight=1.0, t_cont_weight=1.0, pi=None
):
    B = X_batch.shape[0]
    device = X_batch.device
    all_nll, all_kl_t = [], []

    q_p = torch.softmax(posterior.path_logits[cell_indices], dim=1)
    q_eff = belief_propagator.diffuse(q_p, alpha=0.5, steps=2) if belief_propagator else q_p

    for _ in range(n_samples):
        path_idx = torch.multinomial(q_eff, num_samples=1, replacement=True).squeeze(1)
        alpha = posterior.alpha[cell_indices, path_idx].clamp(min=1e-6)
        beta = posterior.beta[cell_indices, path_idx].clamp(min=1e-6)
        t = torch.distributions.Beta(alpha, beta).rsample()

        nll = emission_nll(X_batch, path_idx, t, g_a, g_b, K, sigma2, pi=pi)
        kl_t = compute_kl_beta(alpha, beta).mean()

        all_nll.append(nll)
        all_kl_t.append(kl_t)

    mean_nll = torch.stack(all_nll).mean()
    mean_kl_t = torch.stack(all_kl_t).mean()
    t_cont = continuity_penalty(path_idx, t, traj, posterior)

    path_probs = torch.softmax(posterior.path_logits[cell_indices], dim=1)
    log_path_prior = torch.log(posterior.compute_path_probs().clamp(min=1e-8))
    kl_p = (path_probs * (path_probs.clamp(min=1e-8).log() - log_path_prior)).sum(dim=1).mean()

    elbo = -mean_nll - kl_weight * mean_kl_t - kl_p_weight * kl_p - t_cont_weight * t_cont
    return -elbo, {
        "nll": mean_nll.item(),
        "kl_t": mean_kl_t.item(),
        "kl_p": kl_p.item(),
        "t_cont": t_cont.item(),
        "t": t.detach(),
        "q_eff": q_eff.detach(),
    }
