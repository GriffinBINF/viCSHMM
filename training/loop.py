import torch
import time
from models.loss import compute_elbo, compute_elbo_batch
from models.emission import update_emission_means_variances
from utils.inference import batch_indices


def train_model(
    X, traj_graph, posterior, belief_propagator,
    g_init, K_init, sigma2_init, pi_init,
    edge_tuple_to_index,
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
    branch_entropy_weight=1.0,
    lr=1e-2,
    tau=1.0
):
    device = X.device

    # Learnable params
    K = torch.nn.Parameter(K_init.clone().to(device))
    pi = torch.nn.Parameter(pi_init.clone().to(device)) if pi_init is not None else None

    posterior_params = list(posterior.parameters())
    emission_params = [K]
    if pi is not None:
        emission_params.append(pi)

    g = g_init.clone().to(device)
    sigma2 = sigma2_init.clone().to(device).clamp(min=1e-6)

    if use_lagging:
        optimizer_inf = torch.optim.Adam(posterior_params, lr=lr)
        optimizer_gen = torch.optim.Adam(emission_params, lr=lr)
    else:
        optimizer = torch.optim.Adam(posterior_params + emission_params, lr=lr)

    emissions_frozen = False

    print(f"Starting training with dynamic g/σ² and learnable K{', π' if pi is not None else ''}.")

    for epoch in range(num_epochs):
        start_time = time.time()

        if not use_lagging:
            if freeze_epochs > 0 and epoch < freeze_epochs and not emissions_frozen:
                print(f"Epoch {epoch}: Freezing K and pi.")
                for p in emission_params:
                    p.requires_grad_(False)
                emissions_frozen = True
            elif epoch == freeze_epochs and emissions_frozen:
                print(f"Epoch {epoch}: Unfreezing K and pi.")
                for p in emission_params:
                    p.requires_grad_(True)
                emissions_frozen = False

        # Update g and sigma2 before ELBO
        g, sigma2 = update_emission_means_variances(X, posterior, K, traj_graph, edge_tuple_to_index)

        if use_lagging:
            for _ in range(inference_steps):
                optimizer_inf.zero_grad()
                cell_indices = torch.arange(X.shape[0], device=device)
                elbo, metrics = compute_elbo(
                                X, cell_indices, traj_graph, posterior, edge_tuple_to_index,
                                g, K, sigma2, pi=pi,
                                belief_propagator=belief_propagator,
                                n_samples=n_samples,
                                kl_weight=kl_weight,
                                kl_p_weight=kl_p_weight,
                                t_cont_weight=t_cont_weight,
                                transition_weight=transition_weight,
                                l1_weight=l1_weight,
                                branch_entropy_weight=branch_entropy_weight,
                                tau=tau
                            )
                elbo.backward()
                optimizer_inf.step()

            for _ in range(generative_steps):
                optimizer_gen.zero_grad()
                cell_indices = torch.arange(X.shape[0], device=device)
                elbo, metrics = compute_elbo(
                                X, cell_indices, traj_graph, posterior, edge_tuple_to_index,
                                g, K, sigma2, pi=pi,
                                belief_propagator=belief_propagator,
                                n_samples=n_samples,
                                kl_weight=kl_weight,
                                kl_p_weight=kl_p_weight,
                                t_cont_weight=t_cont_weight,
                                transition_weight=transition_weight,
                                l1_weight=l1_weight,
                                branch_entropy_weight=branch_entropy_weight,
                                tau=tau 
                            )
                elbo.backward()
                optimizer_gen.step()

        else:
            optimizer.zero_grad()
            cell_indices = torch.arange(X.shape[0], device=device)
            elbo, metrics = compute_elbo(
                X, cell_indices, traj_graph, posterior, edge_tuple_to_index,
                g, K, sigma2, pi=pi,
                belief_propagator=belief_propagator,
                n_samples=n_samples,
                kl_weight=kl_weight,
                kl_p_weight=kl_p_weight,
                t_cont_weight=t_cont_weight,
                transition_weight=transition_weight,
                l1_weight=l1_weight,
                branch_entropy_weight=branch_entropy_weight,
                tau = tau
            )
            elbo.backward()
            optimizer.step()

        _log_stats(epoch, elbo.item(), metrics, time.time() - start_time, g, sigma2, K, pi)


def _get_batches(X, batch_size):
    for batch in batch_indices(X.shape[0], batch_size=batch_size):
        yield batch


def _log_stats(epoch, loss, metrics, elapsed, g, sigma2, K, pi):
    q_eff = metrics.get("q_eff")
    entropy = 0.0
    if q_eff is not None:
        q_eff = q_eff.detach()
        entropy = -(q_eff * q_eff.clamp(min=1e-6).log()).sum(dim=1).mean().item()

    print(f"\n[Epoch {epoch}] Loss: {loss:.4e}")
    print(f"  NLL:        {metrics['nll']:.4e}")
    print(f"  KL(t):      {metrics['kl_t']:.4f}")
    print(f"  KL(p):      {metrics['kl_p']:.4f}")
    print(f"  t_cont:     {metrics['t_cont']:.4f}")
    print(f"  Transition: {metrics.get('transition', 0.0):.4f}")
    print(f"  Emis Cont:  {metrics.get('emission_cont', 0.0):.4f}")
    print(f"  Br. Entropy:{metrics.get('entropy', 0.0):.4f}")
    if q_eff is not None:
        print(f"  q_eff Entr: {entropy:.4f}")
    print(f"  g range:    ({g.min():.2f}, {g.max():.2f}), mean: {g.mean():.2f}")
    print(f"  σ² range:   ({sigma2.min():.2f}, {sigma2.max():.2f}), mean: {sigma2.mean():.2f}")
    print(f"  K range:    ({K.min():.2f}, {K.max():.2f}), mean: {K.mean():.2f}")
    if pi is not None:
        print(f"  π range:    ({pi.min():.2f}, {pi.max():.2f}), mean: {pi.mean():.2f}")
    print(f"  Time:       {elapsed:.2f}s")
