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

def _log_stats(epoch, avg_loss, metrics, elapsed, g, sigma2, K, pi=None, training_mode="N/A", phase="N/A"):
    """Logs training statistics."""
    q_eff = metrics.get("q_eff")
    q_eff_entropy = 0.0
    if q_eff is not None:
        q_eff = q_eff.detach()
        q_eff_entropy = -(q_eff * q_eff.clamp(min=EPSILON).log()).sum(dim=1).mean().item()

    log_str = (
        f"\n[Epoch {epoch}] Mode: {training_mode}" + (f" | Phase: {phase}" if phase != "N/A" else "") + "\n"
        f"  Avg. ELBO:  {avg_loss:.4e}\n"
        f"  NLL:        {metrics['nll']:.4e}\n"
        f"  KL(t):      {metrics['kl_t']:.4f}\n"
        f"  KL(p):      {metrics['kl_p']:.4f}\n"
        f"  t_cont:     {metrics['t_cont']:.4f}\n"
        f"  Transition: {metrics.get('transition', 0.0):.4f}\n"
        f"  Emis Cont:  {metrics.get('emission_cont', 0.0):.4f}\n"
        f"  Br. Entropy:{metrics.get('entropy', 0.0):.4f}\n"
        f"  q_eff Entr: {q_eff_entropy:.4f}\n" # Corrected key
        f"  g range:    ({g.min():.2f}, {g.max():.2f}), mean: {g.mean():.2f}\n"
        f"  σ² range:   ({sigma2.min():.2f}, {sigma2.max():.2f}), mean: {sigma2.mean():.2f}\n"
        f"  K range:    ({K.min():.2f}, {K.max():.2f}), mean: {K.mean():.2f}"
    )
    if pi is not None:
        log_str += f"\n  π range:    ({pi.min():.2f}, {pi.max():.2f}), mean: {pi.mean():.2f}"
    log_str += f"\n  Time:       {elapsed:.2f}s"
    print(log_str)


def train_model(
    # Core data and model components
    X, traj_graph, posterior, belief_propagator,
    g_init, K_init, sigma2_init, pi_init,
    edge_tuple_to_index,
    # Training configuration
    training_mode='joint', # 'joint', 'lagging', 'phase_switching'
    use_pi=False, # Whether to use and learn the zero-inflation parameter pi
    num_epochs=100,
    batch_size=512,
    lr=1e-2,
    # Curriculum / Mode specific parameters
    freeze_epochs=10,        # For 'joint' and 'lagging' modes
    inference_steps=5,       # For 'lagging' mode
    generative_steps=1,      # For 'lagging' mode
    phase_epochs_inf=10,     # For 'phase_switching' mode
    phase_epochs_gen=10,     # For 'phase_switching' mode
    # ELBO / Loss hyperparameters
    n_samples=3,
    kl_weight=1.0,
    kl_p_weight=1.0,
    t_cont_weight=1.0,
    transition_weight=1.0,
    l1_weight=0.0,           # Note: L1 regularization on parameters is not explicitly implemented here yet
    branch_entropy_weight=1.0,
    tau=1.0,
    # Other
    device=None # Optional device override
):
    """
    Trains the viCSHMM model with flexible strategies.

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
        freeze_epochs (int): Epochs to freeze K (and pi) in 'joint'/'lagging'.
        inference_steps (int): Posterior updates per epoch in 'lagging'.
        generative_steps (int): Emission updates per epoch in 'lagging'.
        phase_epochs_inf (int): Epochs per posterior phase in 'phase_switching'.
        phase_epochs_gen (int): Epochs per emission phase in 'phase_switching'.
        n_samples (int): Number of Monte Carlo samples for ELBO estimation.
        kl_weight, kl_p_weight, ... (float): Weights for ELBO terms.
        tau (float): Temperature for Gumbel-Softmax sampling.
        device (torch.device, optional): Device to run on. Defaults to X.device.
    """
    effective_device = device or X.device
    X = X.to(effective_device)
    posterior = posterior.to(effective_device)
    # Note: belief_propagator uses posterior.device internally

    # --- Parameter Initialization ---
    # g and sigma2 are updated non-gradient based, so no nn.Parameter needed initially
    g = g_init.clone().to(effective_device)
    sigma2 = sigma2_init.clone().to(effective_device).clamp(min=EPSILON) # Clamp initial variance

    # K is always learnable (unless frozen initially)
    K = torch.nn.Parameter(K_init.clone().to(effective_device))

    # Pi is learnable only if use_pi is True
    pi = None
    if use_pi:
        if pi_init is None:
            print("Warning: use_pi=True but pi_init is None. Initializing pi to zeros.")
            pi_init = torch.zeros_like(K_init) # Default initialization if needed
        pi = torch.nn.Parameter(pi_init.clone().to(effective_device))

    # --- Optimizer Setup (Outside Loop) ---
    posterior_params = list(posterior.parameters())
    emission_params = [K]
    if use_pi and pi is not None:
        emission_params.append(pi)

    optimizer_inf = torch.optim.Adam(posterior_params, lr=lr)
    optimizer_gen = torch.optim.Adam(emission_params, lr=lr) if emission_params else None # Handle case where emission_params might be empty if not learning K/pi
    optimizer_joint = torch.optim.Adam(posterior_params + emission_params, lr=lr) if emission_params else torch.optim.Adam(posterior_params, lr=lr)

    # --- Training State ---
    current_phase = 'inference' # Start phase for phase_switching
    epochs_in_current_phase = 0
    emissions_frozen = False # For joint/lagging curriculum
    
    log_history = {
        "elbo": [], "nll": [], "kl_t": [], "kl_p": [], "t_cont": [],
        "transition": [], "l1": [], "emission_cont": [], "entropy": [],
        "q_eff_entropy": [] # Added for completeness
    }
    
    print(f"--- Starting Training ---")
    print(f"Mode: {training_mode}, Epochs: {num_epochs}, LR: {lr}, Batch Size: {batch_size}")
    print(f"Device: {effective_device}, Use Pi: {use_pi}")
    if training_mode in ['joint', 'lagging']: print(f"Freeze Epochs: {freeze_epochs}")
    if training_mode == 'lagging': print(f"Lagging Steps (Inf/Gen): {inference_steps}/{generative_steps}")
    if training_mode == 'phase_switching': print(f"Phase Epochs (Inf/Gen): {phase_epochs_inf}/{phase_epochs_gen}")
    print(f"-------------------------")

    # --- Main Training Loop ---
    for epoch in range(num_epochs):
        start_time = time.time()
        posterior.train() # Set posterior module to training mode

        # 1. Update g and sigma2 (non-gradient based) using current posterior expectations
        # Detach K as it's used only for expectation calculation here, not for g/sigma2 grads
        with torch.no_grad(): # Ensure K's gradient isn't affected by this calculation if K requires grad
             K_detached = K.detach()
        g, sigma2 = update_emission_means_variances(
             X, posterior, K_detached, traj_graph, edge_tuple_to_index, epsilon=EPSILON
        )
        # Ensure g and sigma2 remain regular tensors, not parameters
        g = g.detach().requires_grad_(False) # Make sure they don't accumulate grads
        sigma2 = sigma2.detach().clamp(min=EPSILON).requires_grad_(False)

        total_loss = 0.0
        total_batches = 0
        # Initialize metrics dict for the epoch
        epoch_metrics_agg = {k: 0.0 for k in ["nll", "kl_t", "kl_p", "t_cont", "transition", "l1", "emission_cont", "entropy"]}
        last_batch_metrics = {} # To store metrics from the last step for logging

        # --- Batch Iteration ---
        num_batches = math.ceil(X.shape[0] / batch_size)
        batch_generator = batch_indices(X.shape[0], batch_size=batch_size)

        for batch_idx, cell_indices_batch in enumerate(batch_generator):
            X_batch = X[cell_indices_batch].to(effective_device)

            # --- Mode-Specific Optimization ---
            if training_mode == 'joint':
                # Handle initial freezing curriculum
                freeze_now = epoch < freeze_epochs
                if freeze_now and not emissions_frozen:
                    print(f"Epoch {epoch}: Freezing emission params (K{', pi' if use_pi else ''})")
                    _set_requires_grad(emission_params, False)
                    # Re-create optimizer only with active parameters IF NECESSARY
                    # (Adam might handle this, but explicit re-creation is safer)
                    active_params = posterior_params
                    optimizer_joint = torch.optim.Adam(active_params, lr=lr)
                    emissions_frozen = True
                elif not freeze_now and emissions_frozen:
                    print(f"Epoch {epoch}: Unfreezing emission params")
                    _set_requires_grad(emission_params, True)
                    active_params = posterior_params + emission_params
                    optimizer_joint = torch.optim.Adam(active_params, lr=lr)
                    emissions_frozen = False

                optimizer_joint.zero_grad()
                loss, metrics = compute_elbo_batch(
                    X_batch, cell_indices_batch, traj_graph, posterior, edge_tuple_to_index,
                    g, K, sigma2, # Pass potentially frozen K/pi
                    belief_propagator=belief_propagator, n_samples=n_samples,
                    kl_weight=kl_weight, kl_p_weight=kl_p_weight, t_cont_weight=t_cont_weight,
                    transition_weight=transition_weight, l1_weight=l1_weight,
                    branch_entropy_weight=branch_entropy_weight, tau=tau,
                    pi=pi if use_pi else None # Pass pi only if used
                )
                if torch.isnan(loss): raise ValueError(f"NaN loss detected in joint mode, epoch {epoch}, batch {batch_idx}")
                loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(optimizer_joint.param_groups[0]['params'], max_norm=1.0)
                optimizer_joint.step()
                last_batch_metrics = metrics # Store for logging

            elif training_mode == 'lagging':
                 freeze_emission_now = epoch < freeze_epochs

                 # Inference Steps (Posterior Update)
                 _set_requires_grad(posterior_params, True)
                 _set_requires_grad(emission_params, False) # Ensure K, pi are frozen during inf step
                 for _ in range(inference_steps):
                     optimizer_inf.zero_grad()
                     # Need to recompute ELBO for each step if using MC sampling
                     loss_inf, metrics_inf = compute_elbo_batch(
                         X_batch, cell_indices_batch, traj_graph, posterior, edge_tuple_to_index,
                         g, K, sigma2, # K/pi grads won't flow back
                         belief_propagator=belief_propagator, n_samples=n_samples,
                         kl_weight=kl_weight, kl_p_weight=kl_p_weight, t_cont_weight=t_cont_weight,
                         transition_weight=transition_weight, l1_weight=l1_weight,
                         branch_entropy_weight=branch_entropy_weight, tau=tau,
                         pi=pi if use_pi else None
                     )
                     if torch.isnan(loss_inf): raise ValueError(f"NaN loss detected in lagging (inf), epoch {epoch}, batch {batch_idx}")
                     loss_inf.backward()
                     optimizer_inf.step()

                 # Generative Steps (Emission Update)
                 _set_requires_grad(posterior_params, False) # Freeze posterior
                 if not freeze_emission_now and optimizer_gen: # Only update if not frozen and params exist
                     _set_requires_grad(emission_params, True)
                     for _ in range(generative_steps):
                         optimizer_gen.zero_grad()
                         loss_gen, metrics_gen = compute_elbo_batch(
                             X_batch, cell_indices_batch, traj_graph, posterior, edge_tuple_to_index,
                             g, K, sigma2, # Grads flow to K/pi now
                             belief_propagator=belief_propagator, n_samples=n_samples,
                             kl_weight=kl_weight, kl_p_weight=kl_p_weight, t_cont_weight=t_cont_weight,
                             transition_weight=transition_weight, l1_weight=l1_weight,
                             branch_entropy_weight=branch_entropy_weight, tau=tau,
                             pi=pi if use_pi else None
                         )
                         if torch.isnan(loss_gen): raise ValueError(f"NaN loss detected in lagging (gen), epoch {epoch}, batch {batch_idx}")
                         loss_gen.backward()
                         optimizer_gen.step()
                         last_batch_metrics = metrics_gen # Log metrics from generative step
                 else:
                     # If frozen or no emission params, just use inference metrics for logging
                     last_batch_metrics = metrics_inf
                 loss = last_batch_metrics.get('loss', loss_inf) # Use last computed loss


            elif training_mode == 'phase_switching':
                if current_phase == 'inference':
                    _set_requires_grad(posterior_params, True)
                    _set_requires_grad(emission_params, False)
                    optimizer_inf.zero_grad()
                    loss, metrics = compute_elbo_batch(
                        X_batch, cell_indices_batch, traj_graph, posterior, edge_tuple_to_index,
                        g, K, sigma2, # K/pi grads won't flow
                        belief_propagator=belief_propagator, n_samples=n_samples,
                        kl_weight=kl_weight, kl_p_weight=kl_p_weight, t_cont_weight=t_cont_weight,
                        transition_weight=transition_weight, l1_weight=l1_weight,
                        branch_entropy_weight=branch_entropy_weight, tau=tau,
                        pi=pi if use_pi else None
                    )
                    if torch.isnan(loss): raise ValueError(f"NaN loss detected in phase_switching (inf), epoch {epoch}, batch {batch_idx}")
                    loss.backward()
                    optimizer_inf.step()
                    last_batch_metrics = metrics

                elif current_phase == 'generative':
                     if optimizer_gen: # Only if there are emission params to optimize
                        _set_requires_grad(posterior_params, False)
                        _set_requires_grad(emission_params, True)
                        optimizer_gen.zero_grad()
                        loss, metrics = compute_elbo_batch(
                            X_batch, cell_indices_batch, traj_graph, posterior, edge_tuple_to_index,
                            g, K, sigma2, # Grads flow to K/pi
                            belief_propagator=belief_propagator, n_samples=n_samples,
                            kl_weight=kl_weight, kl_p_weight=kl_p_weight, t_cont_weight=t_cont_weight,
                            transition_weight=transition_weight, l1_weight=l1_weight,
                            branch_entropy_weight=branch_entropy_weight, tau=tau,
                            pi=pi if use_pi else None
                        )
                        if torch.isnan(loss): raise ValueError(f"NaN loss detected in phase_switching (gen), epoch {epoch}, batch {batch_idx}")
                        loss.backward()
                        optimizer_gen.step()
                        last_batch_metrics = metrics
                     else: # No emission params, just skip generative phase calculations
                         loss = torch.tensor(0.0, device=effective_device) # Avoid error later
                         last_batch_metrics = {}


            # --- Accumulate Metrics ---
            # Use metrics from the relevant step (last step in lagging/phase)
            total_loss += loss.item() # Use item() to detach loss from graph
            for key in epoch_metrics_agg:
                 epoch_metrics_agg[key] += last_batch_metrics.get(key, 0.0) # Gracefully handle missing keys
            total_batches += 1
        # End of batch loop

        # --- Calculate Average Metrics for Epoch ---
        avg_loss_epoch = total_loss / total_batches if total_batches > 0 else 0.0
        for key in epoch_metrics_agg:
            epoch_metrics_agg[key] /= total_batches if total_batches > 0 else 1.0

        for key in log_history.keys():
            log_history[key].append(epoch_metrics_agg.get(key, float('nan'))) # Use nan for missing values
        # --- Log Epoch Statistics ---
        _log_stats(
            epoch, avg_loss_epoch, epoch_metrics_agg, time.time() - start_time,
            g, sigma2, K, pi if use_pi else None,
            training_mode=training_mode,
            phase=current_phase if training_mode == 'phase_switching' else "N/A"
        )

        # --- Phase Switching Logic ---
        if training_mode == 'phase_switching':
            epochs_in_current_phase += 1
            phase_limit = phase_epochs_inf if current_phase == 'inference' else phase_epochs_gen
            if epochs_in_current_phase >= phase_limit:
                current_phase = 'generative' if current_phase == 'inference' else 'inference'
                epochs_in_current_phase = 0
                print(f"--- Switching to {current_phase} phase ---")

    print(f"--- Training Finished ---")
    # Return trained posterior and learned emission parameters
    return posterior, g, K, sigma2, pi, log_history # Return potentially learned pi


def _get_batches(X, batch_size):
    for batch in batch_indices(X.shape[0], batch_size=batch_size):
        yield batch


