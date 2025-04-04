import torch
import time
import math # For ceiling division
from models.loss import compute_elbo_batch # Ensure this is the batch version
from models.emission import update_emission_means_variances
from utils.inference import batch_indices
from utils.constants import EPSILON # For sigma2 clamping

def _set_requires_grad(params, value):
    """Helper function to set requires_grad for a list of parameters."""
    if params: # Check if the list is not empty
        for param in params:
            if param is not None: # Check if the parameter itself exists
                 param.requires_grad_(value)

def _log_stats(epoch, avg_loss, metrics, elapsed, g, sigma2, K, pi=None, training_mode="N/A", phase="N/A", tau=None):
    """Logs training statistics."""
    # Detach metrics that might have grad
    metrics_detached = {k: v.detach().item() if isinstance(v, torch.Tensor) and v.requires_grad else v for k, v in metrics.items()}

    q_eff = metrics_detached.get("q_eff") # q_eff should ideally be detached already from compute_elbo
    q_eff_entropy = 0.0
    if q_eff is not None and isinstance(q_eff, torch.Tensor): # Check if it's a tensor
        q_eff = q_eff.detach() # Ensure detached
        q_eff_clamped = q_eff.clamp(min=EPSILON)
        q_eff_entropy = -(q_eff * q_eff_clamped.log()).sum(dim=-1).mean().item() # Ensure sum over correct dim if needed

    log_str = (
        f"\n[Epoch {epoch}] Mode: {training_mode}" + (f" | Phase: {phase}" if phase != "N/A" else "") + "\n"
        f"  Avg. ELBO:  {avg_loss:.4e} (raw loss: {metrics_detached.get('loss', 0.0):.4e})\n"
        f"  NLL (IS):   {metrics_detached.get('nll_weighted', 0.0):.4e}\n"
        f"  KL(t):      {metrics_detached.get('kl_t', 0.0):.4f}\n"
        f"  KL(p):      {metrics_detached.get('kl_p', 0.0):.4f}\n"
        # f"  t_cont:     {metrics_detached.get('t_cont', 0.0):.4f}\n" # Often zero if not used
        f"  Transition: {metrics_detached.get('transition', 0.0):.4f}\n"
        f"  Emis Cont:  {metrics_detached.get('emission_cont', 0.0):.4f}\n"
        f"  Br. Entropy:{metrics_detached.get('branch_entropy', 0.0):.4f}\n" # Corrected key if needed
        f"  q_eff Entr: {q_eff_entropy:.4f}\n"
        f"  g range:    ({g.min():.2f}, {g.max():.2f}), mean: {g.mean():.2f}\n"
        f"  σ² range:   ({sigma2.min():.2f}, {sigma2.max():.2f}), mean: {sigma2.mean():.2f}\n"
        f"  K range:    ({K.min():.2f}, {K.max():.2f}), mean: {K.mean():.2f}"
    )
    if pi is not None:
        log_str += f"\n  π range:    ({pi.min():.2f}, {pi.max():.2f}), mean: {pi.mean():.2f}"
    if tau is not None:
         log_str += f"\n  Tau (temp): {tau:.3f}"
    log_str += f"\n  Time:       {elapsed:.2f}s"
    print(log_str)


def train_model(
    # Core data and model components
    X, traj_graph, posterior, belief_propagator,
    g_init, K_init, sigma2_init, pi_init,
    edge_tuple_to_index,
    # Training configuration
    training_mode='joint', # 'joint', 'lagging', 'phase_switching'
    use_pi=False,
    num_epochs=100,
    batch_size=512,
    lr=1e-2,
    # Curriculum / Mode specific parameters
    freeze_posterior_epochs=10, # Renamed from freeze_epochs, now freezes posterior
    inference_steps=1,      # Default to 1 step for posterior per batch in lagging
    generative_steps=5,     # Default to more steps for generative per batch in lagging
    phase_epochs_inf=10,    # Epochs per posterior phase in 'phase_switching'
    phase_epochs_gen=10,    # Epochs per emission phase in 'phase_switching'
    # ELBO / Loss hyperparameters
    n_samples=3,
    kl_weight=1.0,
    kl_p_weight=1.0,
    t_cont_weight=1.0,      # Set to 0 if not used
    transition_weight=1.0,
    l1_weight=0.0,          # Usually applied to K/pi if needed
    branch_entropy_weight=1.0,
    # Tau annealing for Gumbel-Softmax (if posterior uses it)
    tau_start=5.0,
    tau_end=0.5,
    tau_anneal_mode='exponential', # 'linear' or 'exponential'
    tau_anneal_rate=0.05,          # Rate for annealing
    # Other
    device=None
):
    """
    Trains the viCSHMM model with flexible strategies, prioritizing emission
    learning early on when using freezing or phasing.

    Args:
        X (Tensor): Expression data [N, G].
        traj_graph (TrajectoryGraph): Initialized trajectory graph.
        posterior (TreeVariationalPosterior): Posterior model instance.
        belief_propagator (BeliefPropagator): Belief propagator instance.
        g_init, K_init, sigma2_init, pi_init (Tensor): Initial emission parameters.
        edge_tuple_to_index (dict): Mapping from (u_idx, v_idx) to edge index.
        training_mode (str): Strategy ('joint', 'lagging', 'phase_switching').
        use_pi (bool): If True, include and learn pi (zero-inflation).
        num_epochs (int): Total training epochs.
        batch_size (int): Minibatch size.
        lr (float): Learning rate.
        freeze_posterior_epochs (int): Epochs to freeze POSTERIOR params in 'joint'.
                                       Emission params (K, pi) are trained first.
        inference_steps (int): Posterior updates per batch in 'lagging'.
        generative_steps (int): Emission (K, pi) updates per batch in 'lagging'.
        phase_epochs_inf (int): Epochs per posterior phase in 'phase_switching'.
        phase_epochs_gen (int): Epochs per emission phase in 'phase_switching'.
        n_samples (int): Number of Monte Carlo samples for ELBO estimation.
        kl_weight, kl_p_weight, ... (float): Weights for ELBO terms.
        tau_start, tau_end, ... : Gumbel-Softmax temperature annealing parameters.
        device (torch.device, optional): Device to run on. Defaults to X.device.

    Returns:
        tuple: (posterior, g, K, sigma2, pi, log_history) trained components and logs.
    """
    effective_device = device or X.device
    X = X.to(effective_device)
    posterior = posterior.to(effective_device)

    # --- Initialize Parameters ---
    # g, sigma2 are updated via M-step, not direct optimization on ELBO
    g = g_init.clone().to(effective_device)
    sigma2 = sigma2_init.clone().to(effective_device).clamp(min=EPSILON)

    # K, pi are nn.Parameters optimized via gradient descent on ELBO
    K = torch.nn.Parameter(K_init.clone().to(effective_device))
    pi = None
    if use_pi:
        if pi_init is None:
            print("Warning: use_pi=True but pi_init is None. Initializing pi to zeros.")
            pi_init = torch.zeros_like(K_init)
        pi = torch.nn.Parameter(pi_init.clone().to(effective_device))

    # --- Parameter Groups for Optimizers ---
    posterior_params = list(posterior.parameters())
    emission_params_optim = [K] # Parameters optimized by gradient descent
    if use_pi and pi is not None:
        emission_params_optim.append(pi)

    if not posterior_params:
         warnings.warn("Posterior model has no parameters to optimize.")
    if not emission_params_optim:
         warnings.warn("No emission parameters (K, pi) marked for gradient-based optimization.")

    # --- Optimizers ---
    # Optimizer for posterior parameters only
    optimizer_inf = torch.optim.Adam(posterior_params, lr=lr) if posterior_params else None
    # Optimizer for emission parameters (K, pi) only
    optimizer_gen = torch.optim.Adam(emission_params_optim, lr=lr) if emission_params_optim else None
    # Optimizer for all trainable parameters
    all_trainable_params = posterior_params + emission_params_optim
    optimizer_joint = torch.optim.Adam(all_trainable_params, lr=lr) if all_trainable_params else None

    if not optimizer_inf and not optimizer_gen:
        raise ValueError("No parameters found for optimization in either posterior or emission model.")

    # --- Training State Variables ---
    # For joint mode freezing:
    posterior_frozen = False
    # For phase switching mode:
    current_phase = 'generative' # Start by optimizing emission params K, pi
    epochs_in_current_phase = 0

    log_history = {
        "elbo": [], "loss": [], "nll_weighted": [], "kl_t": [], "kl_p": [],
        "transition": [], "emission_cont": [], "branch_entropy": [], "q_eff": []
        # Add other metrics as needed, ensure keys match 'compute_elbo_batch' output
    }

    print(f"--- Starting Training ---")
    print(f"Mode: {training_mode}, Epochs: {num_epochs}, LR: {lr}, Batch Size: {batch_size}")
    print(f"Device: {effective_device}, Use Pi: {use_pi}")
    print(f"Tau Annealing: {tau_anneal_mode}, Start: {tau_start:.2f}, End: {tau_end:.2f}, Rate: {tau_anneal_rate:.3f}")
    if training_mode == 'joint': print(f"Freeze Posterior Epochs: {freeze_posterior_epochs}")
    if training_mode == 'lagging': print(f"Lagging Steps (Inf/Gen): {inference_steps}/{generative_steps}")
    if training_mode == 'phase_switching': print(f"Phase Epochs (Gen/Inf): {phase_epochs_gen}/{phase_epochs_inf} (Starting with Generative)")
    print(f"Optimizing K: {K.requires_grad}")
    if use_pi and pi is not None: print(f"Optimizing pi: {pi.requires_grad}")
    print(f"-------------------------")

    # --- Main Training Loop ---
    for epoch in range(num_epochs):
        start_time = time.time()
        posterior.train() # Set posterior model to training mode

        # --- Tau Annealing ---
        current_tau = tau_end # Default to end value if annealing is off or finished
        if epoch < num_epochs: # Apply annealing during training
            if tau_anneal_mode == 'linear':
                tau_range = tau_start - tau_end
                current_tau = max(tau_end, tau_start - tau_range * (epoch / max(1, num_epochs - 1))) # Avoid division by zero if num_epochs=1
            elif tau_anneal_mode == 'exponential':
                # Decay factor calculation: rate^epoch assumes rate < 1, let's use exp decay
                current_tau = tau_end + (tau_start - tau_end) * math.exp(-tau_anneal_rate * epoch)
                current_tau = max(tau_end, current_tau) # Ensure it doesn't go below min
            elif tau_anneal_mode is None or tau_anneal_mode.lower() == 'none':
                 current_tau = tau_start # Use fixed start tau if no annealing
            else:
                raise ValueError(f"Unsupported tau_anneal_mode: {tau_anneal_mode}")

        # --- M-Step for g and sigma2 (before batch loop) ---
        # Uses posterior expectations from previous epoch and current K
        with torch.no_grad():
             # Detach K for the M-step update, as g/sigma2 don't get gradients from ELBO
             K_detached = K.detach()
             # Ensure posterior is in eval mode for deterministic expectations if needed
             # posterior.eval() # Temporarily switch to eval if dropout/batchnorm used
             g, sigma2 = update_emission_means_variances(
                 X, posterior, K_detached, traj_graph, edge_tuple_to_index,
 #                pi=pi.detach() if pi is not None else None, # Use detached pi if applicable
                 epsilon=EPSILON
             )
             # posterior.train() # Switch back to train mode
        # Detach results as they are treated as fixed for the E-step (ELBO computation)
        g = g.detach().requires_grad_(False)
        sigma2 = sigma2.detach().clamp(min=EPSILON).requires_grad_(False)

        # --- Batch Processing ---
        total_loss_accum = 0.0 # Accumulator for the primary loss objective (ELBO)
        total_batches = 0
        # Aggregate metrics over the epoch
        epoch_metrics_agg = {k: 0.0 for k in log_history.keys()}
        last_batch_metrics = {} # Store metrics from the last operation in a batch

        num_batches = math.ceil(X.shape[0] / batch_size)
        batch_generator = batch_indices(X.shape[0], batch_size=batch_size)

        for batch_idx, cell_indices_batch in enumerate(batch_generator):
            X_batch = X[cell_indices_batch].to(effective_device)
            batch_loss = 0.0 # Loss for the current batch

            # Determine which parameters are active based on mode and state
            if training_mode == 'joint':
                freeze_posterior_now = epoch < freeze_posterior_epochs
                if freeze_posterior_now:
                    if not posterior_frozen:
                        print(f"Epoch {epoch}: Freezing posterior parameters. Training Emission (K{', pi' if use_pi else ''}).")
                        _set_requires_grad(posterior_params, False)
                        _set_requires_grad(emission_params_optim, True)
                        current_optimizer = optimizer_gen
                        posterior_frozen = True
                    # Continue using optimizer_gen
                else: # Not freezing posterior
                    if posterior_frozen:
                        print(f"Epoch {epoch}: Unfreezing posterior parameters. Training Jointly.")
                        _set_requires_grad(posterior_params, True)
                        _set_requires_grad(emission_params_optim, True)
                        current_optimizer = optimizer_joint
                        posterior_frozen = False
                    # Continue using optimizer_joint

                if current_optimizer:
                    current_optimizer.zero_grad()
                    loss, metrics = compute_elbo_batch(
                        X_batch, cell_indices_batch, traj_graph, posterior, edge_tuple_to_index,
                        g, K, sigma2, belief_propagator=belief_propagator, n_samples=n_samples,
                        kl_weight=kl_weight, kl_p_weight=kl_p_weight, t_cont_weight=t_cont_weight,
                        transition_weight=transition_weight, l1_weight=l1_weight,
                        branch_entropy_weight=branch_entropy_weight, tau=current_tau,
                        pi=pi if use_pi else None
                    )
                    if torch.isnan(loss): raise ValueError(f"NaN loss detected in joint mode, epoch {epoch}, batch {batch_idx}")
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(all_trainable_params if not freeze_posterior_now else emission_params_optim, 1.0) # Optional gradient clipping
                    current_optimizer.step()
                    batch_loss = loss.item()
                    last_batch_metrics = metrics
                else:
                     warnings.warn(f"Epoch {epoch}, Batch {batch_idx}: No active optimizer in joint mode.")
                     last_batch_metrics = {}


            elif training_mode == 'lagging':
                # Generative (Emission K, pi) steps first
                if optimizer_gen:
                    _set_requires_grad(posterior_params, False)
                    _set_requires_grad(emission_params_optim, True)
                    for _ in range(generative_steps):
                        optimizer_gen.zero_grad()
                        loss_gen, metrics_gen = compute_elbo_batch(
                            X_batch, cell_indices_batch, traj_graph, posterior, edge_tuple_to_index,
                            g, K, sigma2, belief_propagator=belief_propagator, n_samples=n_samples,
                            kl_weight=kl_weight, kl_p_weight=kl_p_weight, t_cont_weight=t_cont_weight,
                            transition_weight=transition_weight, l1_weight=l1_weight,
                            branch_entropy_weight=branch_entropy_weight, tau=current_tau,
                            pi=pi if use_pi else None
                        )
                        if torch.isnan(loss_gen): raise ValueError(f"NaN loss detected in lagging (gen), epoch {epoch}, batch {batch_idx}")
                        loss_gen.backward()
                        # torch.nn.utils.clip_grad_norm_(emission_params_optim, 1.0) # Optional
                        optimizer_gen.step()
                        batch_loss = loss_gen.item() # Use loss from last generative step
                        last_batch_metrics = metrics_gen

                # Inference (Posterior) steps second
                if optimizer_inf:
                    _set_requires_grad(posterior_params, True)
                    _set_requires_grad(emission_params_optim, False)
                    for _ in range(inference_steps):
                        optimizer_inf.zero_grad()
                        loss_inf, metrics_inf = compute_elbo_batch(
                            X_batch, cell_indices_batch, traj_graph, posterior, edge_tuple_to_index,
                            g, K.detach(), sigma2, belief_propagator=belief_propagator, n_samples=n_samples, # Use detached K/pi for posterior update
                            kl_weight=kl_weight, kl_p_weight=kl_p_weight, t_cont_weight=t_cont_weight,
                            transition_weight=transition_weight, l1_weight=l1_weight,
                            branch_entropy_weight=branch_entropy_weight, tau=current_tau,
                            pi=pi.detach() if pi is not None else None
                        )
                        if torch.isnan(loss_inf): raise ValueError(f"NaN loss detected in lagging (inf), epoch {epoch}, batch {batch_idx}")
                        loss_inf.backward()
                        # torch.nn.utils.clip_grad_norm_(posterior_params, 1.0) # Optional
                        optimizer_inf.step()
                        # Don't overwrite batch_loss, report metrics from generative step or last inference if no gen step ran
                        if not optimizer_gen: # If only inference ran
                             batch_loss = loss_inf.item()
                             last_batch_metrics = metrics_inf


            elif training_mode == 'phase_switching':
                if current_phase == 'generative':
                    phase_optimizer = optimizer_gen
                    _set_requires_grad(posterior_params, False)
                    _set_requires_grad(emission_params_optim, True)
                    active_params = emission_params_optim
                    use_detached_emission = False
                else: # current_phase == 'inference'
                    phase_optimizer = optimizer_inf
                    _set_requires_grad(posterior_params, True)
                    _set_requires_grad(emission_params_optim, False)
                    active_params = posterior_params
                    use_detached_emission = True # Posterior depends on fixed emission params

                if phase_optimizer:
                    phase_optimizer.zero_grad()
                    # Use detached K/pi when optimizing posterior
                    current_K = K.detach() if use_detached_emission else K
                    current_pi = pi.detach() if use_detached_emission and pi is not None else pi

                    loss, metrics = compute_elbo_batch(
                        X_batch, cell_indices_batch, traj_graph, posterior, edge_tuple_to_index,
                        g, current_K, sigma2, belief_propagator=belief_propagator, n_samples=n_samples,
                        kl_weight=kl_weight, kl_p_weight=kl_p_weight, t_cont_weight=t_cont_weight,
                        transition_weight=transition_weight, l1_weight=l1_weight,
                        branch_entropy_weight=branch_entropy_weight, tau=current_tau,
                        pi=current_pi
                    )
                    if torch.isnan(loss): raise ValueError(f"NaN loss detected in phase_switching ({current_phase}), epoch {epoch}, batch {batch_idx}")
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(active_params, 1.0) # Optional
                    phase_optimizer.step()
                    batch_loss = loss.item()
                    last_batch_metrics = metrics
                else:
                     warnings.warn(f"Epoch {epoch}, Batch {batch_idx}: No active optimizer for phase {current_phase}.")
                     last_batch_metrics = {}


            # --- Accumulate Batch Results ---
            total_loss_accum += batch_loss
            for key in epoch_metrics_agg.keys():
                # Use .item() for tensors, handle non-tensor metrics
                metric_val = last_batch_metrics.get(key, 0.0)
                if isinstance(metric_val, torch.Tensor):
                    metric_val = metric_val.detach().item()
                epoch_metrics_agg[key] += metric_val # Aggregate raw metric value
            total_batches += 1

        # --- End of Epoch ---
        avg_loss_epoch = total_loss_accum / total_batches if total_batches > 0 else 0.0
        for key in epoch_metrics_agg:
            epoch_metrics_agg[key] /= total_batches if total_batches > 0 else 1.0

        # Store aggregated epoch metrics (use 'elbo' key for the primary objective)
        log_history["elbo"].append(avg_loss_epoch)
        for key in log_history.keys():
             if key != "elbo":
                  # Use the aggregated value, provide NaN if key missing (shouldn't happen with init)
                  log_history[key].append(epoch_metrics_agg.get(key, float('nan')))

        _log_stats(
            epoch, avg_loss_epoch, epoch_metrics_agg, time.time() - start_time,
            g, sigma2.detach(), K.detach(), pi.detach() if use_pi and pi is not None else None, # Pass detached tensors
            training_mode=training_mode,
            phase=current_phase if training_mode == 'phase_switching' else "N/A",
            tau=current_tau
        )

        # Update phase for 'phase_switching' mode
        if training_mode == 'phase_switching':
            epochs_in_current_phase += 1
            phase_limit = phase_epochs_gen if current_phase == 'generative' else phase_epochs_inf
            if epochs_in_current_phase >= phase_limit and epoch < num_epochs -1 : # Don't switch after last epoch
                current_phase = 'inference' if current_phase == 'generative' else 'generative'
                epochs_in_current_phase = 0
                print(f"--- [Epoch {epoch+1}] Switching to {current_phase} phase ---")


    print(f"--- Training Finished ---")
    # Return trained components and logs
    # Ensure returned posterior is on the correct device and potentially in eval mode
    posterior.eval()
    return posterior, g, K, sigma2, pi, log_history


