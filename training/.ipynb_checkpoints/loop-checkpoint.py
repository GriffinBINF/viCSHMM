import torch
import time
from models.loss import compute_elbo, compute_elbo_batch
from utils.inference import batch_indices


def train_model(
    X, traj_graph, posterior, belief_propagator,
    g_a, g_b, K, log_sigma2, pi, edge_to_index,
    use_lagging=False,
    minibatch=False,
    batch_size=512,
    freeze_epochs=10,
    num_epochs=100,
    inference_steps=5,
    generative_steps=1,
    n_samples=3,
    kl_weight=1.0,
    kl_p_weight=1.0,
    t_cont_weight=1.0,
    lr=1e-2
):
    if use_lagging:
        optimizer_inf = torch.optim.Adam(
            [posterior.alpha, posterior.beta, posterior.path_logits], lr=lr)
        optimizer_gen = torch.optim.Adam([g_a, g_b, K, log_sigma2], lr=lr)
    else:
        params = [posterior.alpha, posterior.beta, posterior.path_logits, g_a, g_b, K, log_sigma2]
        optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in range(num_epochs):
        start = time.time()
        if use_lagging:
            # === Inference step ===
            for _ in range(inference_steps):
                optimizer_inf.zero_grad()
                loss = 0.0
                for batch in _get_batches(X, batch_size) if minibatch else [torch.arange(X.shape[0])]:
                    X_batch = X[batch]
                    elbo, _ = compute_elbo_batch(
                        X_batch, batch, traj_graph, posterior, edge_to_index,
                        g_a, g_b, K, log_sigma2.exp(),
                        belief_propagator=belief_propagator,
                        pi=pi,
                        n_samples=n_samples,
                        kl_weight=kl_weight,
                        kl_p_weight=kl_p_weight,
                        t_cont_weight=t_cont_weight
                    )
                    elbo.backward()
                    loss += elbo.item()
                optimizer_inf.step()

            # === Generative step ===
            for _ in range(generative_steps):
                optimizer_gen.zero_grad()
                loss = 0.0
                for batch in _get_batches(X, batch_size) if minibatch else [torch.arange(X.shape[0])]:
                    X_batch = X[batch]
                    elbo, _ = compute_elbo_batch(
                        X_batch, batch, traj_graph, posterior, edge_to_index,
                        g_a, g_b, K, log_sigma2.exp(),
                        belief_propagator=belief_propagator,
                        pi=pi,
                        n_samples=n_samples,
                        kl_weight=kl_weight,
                        kl_p_weight=kl_p_weight,
                        t_cont_weight=t_cont_weight
                    )
                    elbo.backward()
                    loss += elbo.item()
                optimizer_gen.step()
        else:
            # Standard training (joint inference/generative)
            optimizer.zero_grad()
            elbo, metrics = compute_elbo(
                X, traj_graph, posterior, edge_to_index,
                g_a, g_b, K, log_sigma2.exp(), pi=pi,
                belief_propagator=belief_propagator,
                n_samples=n_samples,
                kl_weight=kl_weight,
                kl_p_weight=kl_p_weight,
                t_cont_weight=t_cont_weight
            )
            elbo.backward()

            # Freeze emissions during curriculum phase
            if epoch < freeze_epochs:
                for p in [g_a, g_b, K, log_sigma2]:
                    p.requires_grad_(False)
            else:
                for p in [g_a, g_b, K, log_sigma2]:
                    p.requires_grad_(True)

            optimizer.step()
            _log_stats(epoch, elbo.item(), metrics, time.time() - start)


def _get_batches(X, batch_size):
    for batch in batch_indices(X.shape[0], batch_size=batch_size):
        yield batch


def _log_stats(epoch, loss, metrics, elapsed):
    print(f"[Epoch {epoch}] Loss: {loss:.3e}")
    print(f"  NLL:      {metrics['nll']:.3e}")
    print(f"  KL(t):    {metrics['kl_t']:.3f}")
    print(f"  KL(p):    {metrics['kl_p']:.3f}")
    print(f"  t_cont:   {metrics['t_cont']:.3f}")
    entropy = -(metrics["q_eff"] * metrics["q_eff"].clamp(min=1e-6).log()).sum(dim=1).mean()
    print(f"  Entropy:  {entropy:.3f}")
    print(f"  Time:     {elapsed:.2f}s")
