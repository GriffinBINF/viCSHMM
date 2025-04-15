import torch
import torch.nn.functional as F
import time
import numpy as np
import math # For ceiling division
from models.loss import compute_elbo # Ensure this is the batch version
from models.emission import update_emission_means_variances
from utils.inference import batch_indices
from utils.constants import EPSILON # For sigma2 clamping
from models.proposal import compute_proposal_distribution


def _set_requires_grad(params, value):
    """Helper function to set requires_grad for a list of parameters."""
    if params: # Check if the list is not empty
        for param in params:
            if param is not None: # Check if the parameter itself exists
                 param.requires_grad_(value)

def _log_stats(
    epoch, avg_loss_epoch, metrics_avg_epoch, elapsed, g, sigma2, K, pi=None,
    training_mode="N/A", phase="N/A", tau=None,
    kl_weight=1.0, kl_p_weight=1.0, t_cont_weight=0.0, # t_cont_weight is unused but keep signature
    transition_weight=1.0, # transition_weight is unused but keep signature
    branch_entropy_weight=1.0
):
    """Logs training statistics with both raw and weighted ELBO terms."""
    # Use the metrics dict from the *last batch* for components breakdown
    # avg_loss is the epoch average of the total loss
    metrics_detached = metrics_avg_epoch  # These are already detached and float


    # --- Removed q_eff entropy calculation as q_eff is not returned ---

    def fmt_term(name, raw, weight):
        weighted = raw * weight
        return f"{name}: {raw:.4f} * {weight:.1e} = {weighted:.4e}"

    # --- Modified Log String ---
    log_str = (
        f"\n[Epoch {epoch}] Mode: {training_mode}" + (f" | Phase: {phase}" if phase != "N/A" else "") + "\n"
        # Report the averaged loss clearly
        f"  Avg. Loss (-ELBO): {avg_loss_epoch:.4e}\n"
        # Components from the last batch's metrics dict:
        f"  NLL (IS):      {metrics_detached.get('nll_weighted', 0.0):.4e}\n"
        f"  {fmt_term('KL(t)', metrics_detached.get('kl_t', 0.0), kl_weight)}\n"
        f"  {fmt_term('KL(p)', metrics_detached.get('kl_p', 0.0), kl_p_weight)}\n"
        # Only show branch entropy if weight > 0
        f"  {fmt_term('BranchEntropy', metrics_detached.get('branch_entropy', 0.0), branch_entropy_weight) if branch_entropy_weight > 0 else ''}\n"
        # --- Removed Transition, EmissionCont, q_eff Entr ---
        f"  g range:       ({g.min():.2f}, {g.max():.2f}), mean: {g.mean():.2f}\n"
        f"  σ² range:      ({sigma2.min():.2f}, {sigma2.max():.2f}), mean: {sigma2.mean():.2f}\n"
        f"  K range:       ({K.min():.2f}, {K.max():.2f}), mean: {K.mean():.2f}"
    )
    if pi is not None:
        log_str += f"\n  π range:       ({pi.min():.2f}, {pi.max():.2f}), mean: {pi.mean():.2f}"
    if tau is not None:
        # tau is passed to compute_elbo but not currently used there, log it anyway as it was intended
        log_str += f"\n  Tau (temp):    {tau:.3f}"
    log_str += f"\n  Time:          {elapsed:.2f}s"
    print(log_str)



def train_model(
    # Core data and model components
    X, traj_graph, posterior, belief_propagator,
    pi_init,
    edge_tuple_to_index,
    # Training configuration
    training_mode='joint', # 'joint', 'lagging', 'phase_switching'
    use_pi=False,
    num_epochs=100,
    batch_size=512,
    lr=1e-2,
    # Curriculum / Mode specific parameters
    freeze_posterior_epochs=10,
    inference_steps=1,
    generative_steps=5,
    phase_epochs_inf=10,
    phase_epochs_gen=10,
    # ELBO / Loss hyperparameters
    n_samples=3,
    kl_weight=1.0,
    kl_p_weight=1.0,
    t_cont_weight=1.0,
    transition_weight=1.0,
    l1_weight=0.0,
    branch_entropy_weight=1.0,
    # Tau annealing
    tau_start=5.0,
    tau_end=0.5,
    tau_anneal_mode='exponential',
    tau_anneal_rate=0.05,
    # Proposal distribution hyperparameters
    proposal_edge_temp=1.0,
    proposal_diffusion_alpha=0.5,
    proposal_diffusion_steps=2,
    # Other
    device=None,
    gradient_clip_norm=None
):
    effective_device = device or X.device
    X = X.to(effective_device)
    posterior = posterior.to(effective_device)

    edge_to_index = edge_tuple_to_index
    n_edges = len(edge_to_index)
    n_genes = X.shape[1]
    
    K_vals = []
    sigma2_vals = []
    g_vals = []
    
    for edge_idx in range(posterior.n_edges):
        u_idx, v_idx = posterior.index_to_edge[edge_idx]
        params = traj_graph.emission_params.get(edge_idx, None)
        if params is None:
            raise ValueError(f"Missing emission parameters for edge {edge_idx} ({u_idx}->{v_idx}) in traj_graph.")
        K_vals.append(params['K'])
        sigma2_vals.append(params['r2'])
        g_vals.append(traj_graph.node_emission.get(v_idx, np.zeros(X.shape[1])))

    
    K_init = torch.tensor(np.stack(K_vals), dtype=torch.float32, device=X.device)
    # Derive K_raw from K_init using inverse softplus
    K_init_clamped = K_init.clone().clamp(min=1e-6)  # Avoid log(0) or log(negative)
    K_raw_init = torch.log((torch.exp(K_init_clamped) - 1.0).clamp(min=1e-6))
    K_raw = torch.nn.Parameter(K_raw_init.to(effective_device))

    sigma2_init = torch.tensor(np.stack(sigma2_vals), dtype=torch.float32, device=X.device)
    g_init = torch.tensor(np.stack(g_vals), dtype=torch.float32, device=X.device)

    pi = None
    if use_pi:
        if pi_init is None:
            print("Warning: use_pi=True but pi_init is None. Initializing pi to zeros.")
            pi_init = torch.zeros_like(K_init)
        pi = torch.nn.Parameter(pi_init.clone().to(effective_device))

    posterior_params = list(posterior.parameters())
    emission_params_optim = [K_raw]
    if use_pi and pi is not None:
        emission_params_optim.append(pi)

    optimizer_inf = torch.optim.Adam(posterior_params, lr=lr) if posterior_params else None
    optimizer_gen = torch.optim.Adam(emission_params_optim, lr=lr) if emission_params_optim else None
    all_trainable_params = posterior_params + emission_params_optim
    optimizer_joint = torch.optim.Adam(all_trainable_params, lr=lr) if all_trainable_params else None

    if not optimizer_inf and not optimizer_gen:
        raise ValueError("No parameters found for optimization in either posterior or emission model.")

    posterior_frozen = False
    current_phase = 'generative'
    epochs_in_current_phase = 0

    log_history = {
            "loss": [],          # Store the main averaged loss here
            "nll_weighted": [],
            "kl_t": [],
            "kl_p": [],
            "branch_entropy": [],
            # Optional: Add diagnostics if needed
            "log_weight_mean": [],
            "log_weight_std": [],
        }

    print(f"--- Starting Training ---")
    print(f"Mode: {training_mode}, Epochs: {num_epochs}, LR: {lr}, Batch Size: {batch_size}")
    print(f"Device: {effective_device}, Use Pi: {use_pi}")
    print(f"Tau Annealing: {tau_anneal_mode}, Start: {tau_start:.2f}, End: {tau_end:.2f}, Rate: {tau_anneal_rate:.3f}")
    print(f"Optimizing K_raw (K > 0 via Softplus): {K_raw.requires_grad}")
    if use_pi and pi is not None: print(f"Optimizing pi: {pi.requires_grad}")
    print(f"Proposal Edge Temp: {proposal_edge_temp}, Diffusion Alpha: {proposal_diffusion_alpha}, Steps: {proposal_diffusion_steps}")

    print(f"-------------------------")

    for epoch in range(num_epochs):
        start_time = time.time()
        posterior.train()

        current_tau = tau_end
        if epoch < num_epochs:
            if tau_anneal_mode == 'linear':
                tau_range = tau_start - tau_end
                current_tau = max(tau_end, tau_start - tau_range * (epoch / max(1, num_epochs - 1)))
            elif tau_anneal_mode == 'exponential':
                current_tau = tau_end + (tau_start - tau_end) * math.exp(-tau_anneal_rate * epoch)
                current_tau = max(tau_end, current_tau)
            elif tau_anneal_mode is None or tau_anneal_mode.lower() == 'none':
                current_tau = tau_start
            else:
                raise ValueError(f"Unsupported tau_anneal_mode: {tau_anneal_mode}")

        K = F.softplus(K_raw)

        with torch.no_grad():
            g, sigma2 = update_emission_means_variances(
                X, posterior, K.detach(), traj_graph, edge_tuple_to_index,
                epsilon=EPSILON
            )
        g = g.detach().requires_grad_(False)
        sigma2 = sigma2.detach().clamp(min=EPSILON).requires_grad_(False)

        total_loss_accum = 0.0
        total_batches = 0
        epoch_metrics_agg = {k: 0.0 for k in log_history.keys()}
        last_batch_metrics = {}

        num_batches = math.ceil(X.shape[0] / batch_size)
        batch_generator = batch_indices(X.shape[0], batch_size=batch_size)
        with torch.no_grad():
            prop_edge_probs_all, prop_alpha_all, prop_beta_all = compute_proposal_distribution(
                posterior,
                belief_propagator,
                proposal_beta_const=10.0,
                proposal_target_temp=1.0,
                proposal_edge_temp=proposal_edge_temp,
                diffusion_alpha=proposal_diffusion_alpha,
                diffusion_steps=proposal_diffusion_steps
            )


        for batch_idx, cell_indices_batch in enumerate(batch_generator):
            X_batch = X[cell_indices_batch].to(effective_device)
            batch_loss = 0.0
            K_used = F.softplus(K_raw)
            pi_used = torch.sigmoid(pi) if use_pi and pi is not None else None

            if training_mode == 'joint':
                freeze_posterior_now = epoch < freeze_posterior_epochs
                if freeze_posterior_now:
                    _set_requires_grad(posterior_params, False)
                    _set_requires_grad(emission_params_optim, True)
                    current_optimizer = optimizer_gen
                    active_params = emission_params_optim
                else:
                    _set_requires_grad(posterior_params, True)
                    _set_requires_grad(emission_params_optim, True)
                    current_optimizer = optimizer_joint
                    active_params = all_trainable_params

                if current_optimizer:
                    current_optimizer.zero_grad()
                    loss, metrics = compute_elbo(
                                                X=X_batch,                     # Use keyword args for clarity
                                                cell_indices=cell_indices_batch,
                                                traj=traj_graph,
                                                posterior=posterior,
                                                edge_tuple_to_index=edge_tuple_to_index,
                                                g=g,
                                                K=K_used,
                                                sigma2=sigma2,
                                                pi=pi_used,                 # Correctly assign pi
                                                belief_propagator=belief_propagator, # Correctly assign belief_propagator
                                                prop_edge_probs_all=prop_edge_probs_all,
                                                prop_alpha_all=prop_alpha_all,
                                                prop_beta_all=prop_beta_all,
                                                n_samples=n_samples,           # Correctly assign n_samples
                                                kl_weight=kl_weight,
                                                kl_p_weight=kl_p_weight,
                                                branch_entropy_weight=branch_entropy_weight,
                                                # Pass other relevant weights/params using keywords
                                                t_cont_weight=t_cont_weight,
                                                transition_weight=transition_weight,
                                                l1_weight=l1_weight,
                                                tau=current_tau
                                                # proposal_beta_const and proposal_target_temp will use defaults from compute_elbo
                                                # eps will use default from compute_elbo
                                            )
                    if torch.isnan(loss): raise ValueError(f"NaN loss detected in joint mode, epoch {epoch}, batch {batch_idx}")
                    loss.backward()
                    if gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(active_params, max_norm=gradient_clip_norm)
                    current_optimizer.step()
                    batch_loss = loss.item()
                    last_batch_metrics = metrics

            elif training_mode == 'lagging':
                if optimizer_gen:
                    _set_requires_grad(posterior_params, False)
                    _set_requires_grad(emission_params_optim, True)
                    for _ in range(generative_steps):
                        optimizer_gen.zero_grad()
                        K = F.softplus(K_raw)
                        loss_gen, metrics_gen = compute_elbo(
                                                X=X_batch,                     # Use keyword args for clarity
                                                cell_indices=cell_indices_batch,
                                                traj=traj_graph,
                                                posterior=posterior,
                                                edge_tuple_to_index=edge_tuple_to_index,
                                                g=g,
                                                K=K_used,
                                                sigma2=sigma2,
                                                pi=pi_used,                 # Correctly assign pi
                                                belief_propagator=belief_propagator, # Correctly assign belief_propagator
                                                prop_edge_probs_all=prop_edge_probs_all,
                                                prop_alpha_all=prop_alpha_all,
                                                prop_beta_all=prop_beta_all,
                                                n_samples=n_samples,           # Correctly assign n_samples
                                                kl_weight=kl_weight,
                                                kl_p_weight=kl_p_weight,
                                                branch_entropy_weight=branch_entropy_weight,
                                                # Pass other relevant weights/params using keywords
                                                t_cont_weight=t_cont_weight,
                                                transition_weight=transition_weight,
                                                l1_weight=l1_weight,
                                                tau=current_tau
                                                # proposal_beta_const and proposal_target_temp will use defaults from compute_elbo
                                                # eps will use default from compute_elbo
                                            )
                        if torch.isnan(loss_gen): raise ValueError(f"NaN loss detected in lagging (gen), epoch {epoch}, batch {batch_idx}")
                        loss_gen.backward()
                        if gradient_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(emission_params_optim, max_norm=gradient_clip_norm)
                        optimizer_gen.step()
                        batch_loss = loss_gen.item()
                        last_batch_metrics = metrics_gen

                if optimizer_inf:
                    _set_requires_grad(posterior_params, True)
                    _set_requires_grad(emission_params_optim, False)
                    for _ in range(inference_steps):
                        optimizer_inf.zero_grad()
                        K_used = F.softplus(K_raw).detach()
                        pi_used = torch.sigmoid(pi).detach() if pi is not None else None

                        loss, metrics = compute_elbo(
                                                X=X_batch,                     # Use keyword args for clarity
                                                cell_indices=cell_indices_batch,
                                                traj=traj_graph,
                                                posterior=posterior,
                                                edge_tuple_to_index=edge_tuple_to_index,
                                                g=g,
                                                K=K_used,
                                                sigma2=sigma2,
                                                pi=pi_used,                 # Correctly assign pi
                                                belief_propagator=belief_propagator, # Correctly assign belief_propagator
                                                prop_edge_probs_all=prop_edge_probs_all,
                                                prop_alpha_all=prop_alpha_all,
                                                prop_beta_all=prop_beta_all,
                                                n_samples=n_samples,           # Correctly assign n_samples
                                                kl_weight=kl_weight,
                                                kl_p_weight=kl_p_weight,
                                                branch_entropy_weight=branch_entropy_weight,
                                                # Pass other relevant weights/params using keywords
                                                t_cont_weight=t_cont_weight,
                                                transition_weight=transition_weight,
                                                l1_weight=l1_weight,
                                                tau=current_tau
                                                # proposal_beta_const and proposal_target_temp will use defaults from compute_elbo
                                                # eps will use default from compute_elbo
                                            )
                        if torch.isnan(loss_inf): raise ValueError(f"NaN loss detected in lagging (inf), epoch {epoch}, batch {batch_idx}")
                        loss_inf.backward()
                        if gradient_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(posterior_params, max_norm=gradient_clip_norm)
                        optimizer_inf.step()
                        if not optimizer_gen:
                            batch_loss = loss_inf.item()
                            last_batch_metrics = metrics_inf

            elif training_mode == 'phase_switching':
                if current_phase == 'generative':
                    phase_optimizer = optimizer_gen
                    _set_requires_grad(posterior_params, False)
                    _set_requires_grad(emission_params_optim, True)
                    active_params = emission_params_optim
                    K_used = F.softplus(K_raw)
                    pi_used = torch.sigmoid(pi) if pi is not None else None
                else:
                    phase_optimizer = optimizer_inf
                    _set_requires_grad(posterior_params, True)
                    _set_requires_grad(emission_params_optim, False)
                    active_params = posterior_params
                    K_used = F.softplus(K_raw).detach()
                    pi_used = torch.sigmoid(pi) if pi is not None else None


                if phase_optimizer:
                    phase_optimizer.zero_grad()
                    loss, metrics = compute_elbo(
                                                X=X_batch,                     # Use keyword args for clarity
                                                cell_indices=cell_indices_batch,
                                                traj=traj_graph,
                                                posterior=posterior,
                                                edge_tuple_to_index=edge_tuple_to_index,
                                                g=g,
                                                K=K_used,
                                                sigma2=sigma2,
                                                pi=pi_used,                 # Correctly assign pi
                                                belief_propagator=belief_propagator, # Correctly assign belief_propagator
                                                prop_edge_probs_all=prop_edge_probs_all,
                                                prop_alpha_all=prop_alpha_all,
                                                prop_beta_all=prop_beta_all,
                                                n_samples=n_samples,           # Correctly assign n_samples
                                                kl_weight=kl_weight,
                                                kl_p_weight=kl_p_weight,
                                                branch_entropy_weight=branch_entropy_weight,
                                                # Pass other relevant weights/params using keywords
                                                t_cont_weight=t_cont_weight,
                                                transition_weight=transition_weight,
                                                l1_weight=l1_weight,
                                                tau=current_tau
                                                # proposal_beta_const and proposal_target_temp will use defaults from compute_elbo
                                                # eps will use default from compute_elbo
                                            )
                    if torch.isnan(loss): raise ValueError(f"NaN loss detected in phase_switching ({current_phase}), epoch {epoch}, batch {batch_idx}")
                    loss.backward()
                    if gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(active_params, max_norm=gradient_clip_norm)
                    phase_optimizer.step()
                    batch_loss = loss.item()
                    last_batch_metrics = metrics

            batch_size_actual = len(cell_indices_batch)
            total_loss_accum += batch_loss * batch_size_actual
            for key in log_history.keys(): # Iterate over the NEW keys in log_history
             # Get value from the last batch's metrics for this key
                val = last_batch_metrics.get(key, 0.0) # Default to 0.0 if key not present (e.g., 'loss')
             # Aggregate only the component terms, not the total loss again
                if key != 'loss':
                    epoch_metrics_agg[key] += val * len(cell_indices_batch)
            total_batches += 1

        avg_loss_epoch = total_loss_accum / X.shape[0] if total_batches > 0 else 0.0
        log_history["loss"].append(avg_loss_epoch) # Use the new key 'loss'
        for key in log_history.keys():
            if key != "loss": # Already saved avg_loss_epoch
                # Use the aggregated sum divided by batches
                averaged_metric = epoch_metrics_agg.get(key, float('nan')) / X.shape[0]
                log_history[key].append(averaged_metric)

        K_final_epoch = F.softplus(K_raw).detach()
        avg_metrics_epoch = {
        k: v / X.shape[0] if k != 'loss' else avg_loss_epoch
        for k, v in epoch_metrics_agg.items()
        }

        _log_stats(
            epoch, avg_loss_epoch, avg_metrics_epoch, time.time() - start_time,
            g, sigma2.detach(), K_final_epoch,
            pi.detach() if use_pi and pi is not None else None,
            training_mode=training_mode,
            phase=current_phase if training_mode == 'phase_switching' else "N/A",
            tau=current_tau,
            kl_weight=kl_weight,
            kl_p_weight=kl_p_weight,
            t_cont_weight=t_cont_weight,
            transition_weight=transition_weight,
            branch_entropy_weight=branch_entropy_weight
        )

        if training_mode == 'phase_switching':
            epochs_in_current_phase += 1
            phase_limit = phase_epochs_gen if current_phase == 'generative' else phase_epochs_inf
            if epochs_in_current_phase >= phase_limit and epoch < num_epochs - 1:
                current_phase = 'inference' if current_phase == 'generative' else 'generative'
                epochs_in_current_phase = 0
                print(f"--- [Epoch {epoch+1}] Switching to {current_phase} phase ---")

    print(f"--- Training Finished ---")
    posterior.eval()
    return posterior, g, F.softplus(K_raw).detach(), sigma2, pi.detach() if pi is not None else None, log_history




