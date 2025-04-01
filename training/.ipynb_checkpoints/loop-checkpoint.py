import torch
import time
from models.loss import compute_elbo, compute_elbo_batch
from utils.inference import batch_indices


def train_model(
    X, traj_graph, posterior, belief_propagator,
    g, K, log_sigma2, pi, edge_to_index,
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
    transition_weight=1.0,
    l1_weight=0.0,
    lr=1e-2
):
    node_params = [g, K, log_sigma2]

    if use_lagging:
        optimizer_inf = torch.optim.Adam(
            [posterior.alpha, posterior.beta, posterior.edge_logits, posterior.branch_logits],
            lr=lr
        )
        optimizer_gen = torch.optim.Adam(node_params, lr=lr)
    else:
        optimizer = torch.optim.Adam(
            [posterior.alpha, posterior.beta, posterior.edge_logits, posterior.branch_logits] + node_params,
            lr=lr
        )

    emissions_frozen = False

    for epoch in range(num_epochs):
        start_time = time.time()

        if use_lagging:
            for _ in range(inference_steps):
                optimizer_inf.zero_grad()
                total_loss = 0.0

                for batch in _get_batches(X, batch_size) if minibatch else [torch.arange(X.shape[0], device=X.device)]:
                    X_batch = X[batch]
                    elbo, _ = compute_elbo_batch(
                        X_batch, batch, traj_graph, posterior, edge_to_index,
                        g, K, log_sigma2.exp(), pi=pi,
                        belief_propagator=belief_propagator,
                        n_samples=n_samples,
                        kl_weight=kl_weight,
                        kl_p_weight=kl_p_weight,
                        t_cont_weight=t_cont_weight,
                        transition_weight=transition_weight,
                        l1_weight=l1_weight
                    )
                    elbo.backward()
                    total_loss += elbo.detach()

                optimizer_inf.step()

            for _ in range(generative_steps):
                optimizer_gen.zero_grad()
                total_loss = 0.0

                for batch in _get_batches(X, batch_size) if minibatch else [torch.arange(X.shape[0], device=X.device)]:
                    X_batch = X[batch]
                    elbo, _ = compute_elbo_batch(
                        X_batch, batch, traj_graph, posterior, edge_to_index,
                        g, K, log_sigma2.exp(), pi=pi,
                        belief_propagator=belief_propagator,
                        n_samples=n_samples,
                        kl_weight=kl_weight,
                        kl_p_weight=kl_p_weight,
                        t_cont_weight=t_cont_weight,
                        transition_weight=transition_weight,
                        l1_weight=l1_weight
                    )
                    elbo.backward()
                    total_loss += elbo.detach()

                optimizer_gen.step()

        else:
            optimizer.zero_grad()

            cell_indices = torch.arange(X.shape[0], device=X.device)
            elbo, metrics = compute_elbo(
                X, cell_indices, traj_graph, posterior, edge_to_index,
                g, K, log_sigma2.exp(), pi=pi,
                belief_propagator=belief_propagator,
                n_samples=n_samples,
                kl_weight=kl_weight,
                kl_p_weight=kl_p_weight,
                t_cont_weight=t_cont_weight,
                transition_weight=transition_weight,
                l1_weight=l1_weight
            )
            elbo.backward()

            if epoch == 0 and not emissions_frozen:
                for p in node_params:
                    p.requires_grad_(False)
                emissions_frozen = True

            elif epoch == freeze_epochs and emissions_frozen:
                for p in node_params:
                    p.requires_grad_(True)
                emissions_frozen = False

            optimizer.step()
            _log_stats(epoch, elbo.detach().item(), metrics, time.time() - start_time)


def _get_batches(X, batch_size):
    for batch in batch_indices(X.shape[0], batch_size=batch_size):
        yield batch


def _log_stats(epoch, loss, metrics, elapsed):
    q_eff = metrics["q_eff"].detach()
    entropy = -(q_eff * q_eff.clamp(min=1e-6).log()).sum(dim=1).mean().item()

    print(f"[Epoch {epoch}] Loss: {loss:.3e}")
    print(f"  NLL:        {metrics['nll']:.3e}")
    print(f"  KL(t):      {metrics['kl_t']:.3f}")
    print(f"  KL(p):      {metrics['kl_p']:.3f}")
    print(f"  t_cont:     {metrics['t_cont']:.3f}")
    print(f"  Transition: {metrics.get('transition', 0.0):.3f}")
    print(f"  L1 Δg:      {metrics.get('l1', 0.0):.3f}")
    print(f"  Entropy:    {entropy:.3f}")
    print(f"  Time:       {elapsed:.2f}s")
