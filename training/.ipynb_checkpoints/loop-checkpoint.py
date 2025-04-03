# training/loop.py

import torch
import time
import networkx as nx
import math
import torch.nn.functional as F
from collections import defaultdict
import time # Ensure time is imported
import numpy as np # Import numpy

# --- Model & Utility Imports ---
from models.posterior import TreeVariationalPosterior
from models.belief import BeliefPropagator
from models.emission import pack_emission_params, emission_nll # Ensure correct path
from models.loss import compute_elbo_batch # Use the updated version
from utils.inference import batch_indices # Ensure correct path
from utils.constants import EPSILON, EPSILON_LOG # Use constants

# ============================================
# === Helper Functions (L1 Penalties) ===
# ============================================

def compute_l1_penalty_K(K, weight):
    """ Computes L1 penalty on K parameter tensor. """
    if weight == 0.0 or K is None:
        return torch.tensor(0.0, device=K.device if K is not None else 'cpu', dtype=torch.float32)
    # L1 penalty encourages K to be zero (simpler dynamics)
    # Sum of absolute values across all elements
    return weight * torch.linalg.norm(K.reshape(-1), ord=1)


# ============================================
# ======== VIRunner Class =========
# ============================================

class VIRunner:
    """
    Manages the Variational Inference optimization process.

    Key Features:
    - Handles lagging vs. standard VI.
    - Recalculates non-learned 'g' (node means) and 'sigma2' (edge variances)
      each epoch based on current posterior assignments.
    - Learns 'K' (edge rates) via optimization.
    - Supports annealing schedules and various ELBO term weights.
    """
    def __init__(self, default_config=None, device=None):
        """ Initializes the VIRunner. """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {
            # *** Learnability Control ***
            "learn_K": True,              # K is typically learned
            "learn_g": False,             # Fixed to False: g recalculated from data/posterior
            "learn_sigma2": False,        # Fixed to False: sigma2 recalculated from data/posterior
            # --- Optimization Strategy ---
            "use_lagging": False,
            "minibatch": False,
            "batch_size": 512,
            "lr": 1e-2,
            # --- Training Duration & Sampling ---
            "num_epochs": 100,
            "n_samples": 3,               # Samples for ELBO expectation
            "inference_steps": 5,         # E-steps per epoch if lagging
            "generative_steps": 1,        # M-steps per epoch if lagging
            # --- Curriculum Learning (Applies only to learn_K if True) ---
            "freeze_epochs": 10,          # Freeze K for first N epochs (if learn_K=True & use_lagging=False)
            # --- Loss Term Weights ---
            "kl_weight": 1.0,             # Weight for KL(t)
            "kl_p_weight": 1.0,             # Weight for KL(p)
            "t_cont_weight": 1.0,           # Weight for time continuity penalty
            "l1_K_weight": 0.0,           # L1 penalty on K (rate parameter)
            "branch_entropy_weight": 1.0,   # Weight for branch entropy regularization
            "struct_prior_weight": 0.0,     # Weight for structural KL divergence term in ELBO
            # --- Annealing Schedules ---
            "anneal_start_epoch": 10,     # Epoch to start annealing temperature/struct_prior
            "anneal_epochs": 50,          # Number of epochs to anneal over
            "logit_temp_start": 5.0,      # Starting temperature for edge logits softmax
            "logit_temp_end": 1.0,        # Final temperature
            "struct_prior_anneal_start": 0.0, # Starting weight for struct_prior_weight
            "struct_prior_anneal_end": 1.0,   # Final weight for struct_prior_weight
            # --- Other ---
            "use_belief_propagator": True,# Use BeliefPropagator for q_eff
            "recalc_prior_std": 0.1,    # Prior std dev added when recalculating sigma2
            "verbose": True,
            "short_run": False,           # Run fewer epochs (e.g., for evaluation)
            "eval_epochs": 20             # Number of epochs for short_run
        }
        if default_config: self.config.update(default_config)

        # Ensure correct types for config values
        self._cast_config_types()

        if self.config['verbose']: print(f"VIRunner initialized on device: {self.device}")
        if self.config['verbose']: print(f"Learn K: {self.config['learn_K']}. g & sigma2 recalculated each epoch.")

    def _cast_config_types(self):
        """Ensures config values have the correct numeric types."""
        float_keys = ['lr', 'kl_weight', 'kl_p_weight', 't_cont_weight', 'l1_K_weight',
                      'branch_entropy_weight', 'struct_prior_weight', 'logit_temp_start',
                      'logit_temp_end', 'struct_prior_anneal_start', 'struct_prior_anneal_end',
                      'recalc_prior_std']
        int_keys = ['batch_size', 'num_epochs', 'n_samples', 'inference_steps',
                    'generative_steps', 'freeze_epochs', 'anneal_start_epoch',
                    'anneal_epochs', 'eval_epochs']

        for key in float_keys:
            self.config[key] = float(self.config.get(key, 0.0))
        for key in int_keys:
            self.config[key] = int(self.config.get(key, 0))

    def _get_annealing_value(self, epoch, start_epoch, total_anneal_epochs, start_val, end_val):
        """ Calculates linear annealing value. """
        if epoch < start_epoch: return start_val
        if total_anneal_epochs <= 0: return end_val
        # Ensure total_anneal_epochs is positive to avoid division by zero
        total_anneal_epochs = max(1, total_anneal_epochs)
        if epoch >= start_epoch + total_anneal_epochs: return end_val
        progress = max(0.0, min(1.0, (epoch - start_epoch) / total_anneal_epochs))
        return start_val + (end_val - start_val) * progress

    def _initialize_params_and_posterior(self, traj_graph, X, initial_params=None):
        """
        Initializes parameters and posterior.
        g and sigma2 NEVER require grad. K requires grad based on config['learn_K'].
        Returns: g, K, log_sigma2, pi, posterior, belief_propagator, node_to_index, edge_to_index
        """
        n_cells, n_genes = X.shape
        g, K, log_sigma2, pi = None, None, None, None

        # Determine requires_grad for K
        learn_K = self.config['learn_K']

        required_init_keys = ['g', 'K', 'log_sigma2']
        if initial_params and all(k in initial_params for k in required_init_keys):
            if self.config['verbose']: print("Using provided initial emission parameters.")
            try:
                # Initialize g, K, log_sigma2 from provided params
                g = initial_params['g'].clone().detach().to(self.device).requires_grad_(False) # Never learns
                K = initial_params['K'].clone().detach().to(self.device).requires_grad_(learn_K)
                log_sigma2 = initial_params['log_sigma2'].clone().detach().to(self.device).requires_grad_(False) # Never learns
                pi_init = initial_params.get('pi')
                pi = pi_init.clone().detach().to(self.device) if pi_init is not None else None
            except AttributeError as e:
                raise ValueError(f"Invalid initial_params structure or type: {e}") from e
        else:
            if self.config['verbose']: print("Initializing emission parameters from TrajectoryGraph.")
            try:
                edge_to_index_init, g_node_init, K_init, sigma2_init, pi_init = pack_emission_params(traj_graph, device=self.device)
                g = g_node_init.clone().detach().to(self.device).requires_grad_(False) # Never learns
                K = K_init.clone().detach().to(self.device).requires_grad_(learn_K)
                log_sigma2 = torch.log(sigma2_init.clamp(min=EPSILON)).requires_grad_(False) # Never learns
                pi = pi_init.clone().detach().to(self.device) if pi_init is not None else None
            except Exception as e:
                raise RuntimeError(f"Failed to initialize parameters via pack_emission_params: {e}") from e

        if self.config['verbose']: print(f"Requires Grad -> g: {g.requires_grad}, K: {K.requires_grad}, log_sigma2: {log_sigma2.requires_grad}")

        # Initialize Posterior and BeliefPropagator
        try:
            posterior = TreeVariationalPosterior(traj_graph, n_cells=n_cells, device=self.device)
            for p in posterior.parameters(): p.requires_grad_(True) # Posterior params always learned
            belief_propagator = BeliefPropagator(traj_graph, posterior) if self.config.get("use_belief_propagator", True) else None
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Posterior/BeliefPropagator: {e}") from e

        # --- Validation ---
        node_to_index_map = posterior.traj.node_to_index
        edge_to_index_map = posterior.edge_to_index
        n_nodes_graph = len(node_to_index_map)
        n_edges_graph = posterior.n_edges
        if g is not None and g.shape[0] != n_nodes_graph: raise ValueError(f"Shape mismatch: g nodes {g.shape[0]} vs graph {n_nodes_graph}")
        if K is not None and K.shape[0] != n_edges_graph: raise ValueError(f"Shape mismatch: K edges {K.shape[0]} vs graph {n_edges_graph}")
        if log_sigma2 is not None and log_sigma2.shape[0] != n_edges_graph: raise ValueError(f"Shape mismatch: log_sigma2 edges {log_sigma2.shape[0]} vs graph {n_edges_graph}")
        if pi is not None and pi.shape[0] != n_edges_graph: raise ValueError(f"Shape mismatch: pi edges {pi.shape[0]} vs graph {n_edges_graph}")
        if self.config['verbose']: print(f"Initialization complete: {n_nodes_graph} nodes, {n_edges_graph} edges.")

        return g, K, log_sigma2, pi, posterior, belief_propagator, node_to_index_map, edge_to_index_map


    def _recalculate_g_sigma2(self, X, posterior, traj_graph, node_to_index, edge_to_index):
        """
        Recalculates g (node means) and sigma2 (edge variances) based on
        current posterior expectations (q_eff). Uses weighted averages/variances of data X.

        Args:
            X (Tensor): Full data [N_cells, N_genes].
            posterior (TreeVariationalPosterior): Current posterior state.
            traj_graph (TrajectoryGraph): Graph structure.
            node_to_index, edge_to_index (dict): Mappings.

        Returns:
            tuple: (recalculated_g, recalculated_sigma2) - Detached tensors on the correct device.
        """
        # if self.config['verbose']: print("Recalculating g and sigma2...") # Can be too frequent
        n_cells, n_genes = X.shape
        n_nodes = len(node_to_index)
        n_edges = len(edge_to_index)
        device = self.device

        recalculated_g = torch.zeros(n_nodes, n_genes, device=device)
        recalculated_sigma2 = torch.zeros(n_edges, n_genes, device=device)

        with torch.no_grad(): # Perform calculation without tracking gradients
            # 1. Get current expected edge assignments (q_eff)
            raw_logits = posterior.edge_logits
            # Use temperature=1 for recalculation weights
            q_e_raw = torch.softmax(raw_logits / 1.0, dim=1)
            if self.config['use_belief_propagator']:
                # Ensure belief_propagator uses the *current* posterior state for A
                # A temporary propagator is safer if the main one isn't updated.
                temp_bp = BeliefPropagator(traj_graph, posterior)
                q_eff = temp_bp.diffuse(q_e_raw, alpha=0.5, steps=2)
            else:
                q_eff = q_e_raw
            q_eff = q_eff / q_eff.sum(dim=1, keepdim=True).clamp(min=EPSILON)
            q_eff = q_eff.clamp(min=EPSILON_LOG) # [N_cells, N_edges]

            # 2. Recalculate g (Node Means) - Weighted average of X based on edge assignments
            g_numerator = torch.zeros(n_nodes, n_genes, device=device)
            g_denominator = torch.zeros(n_nodes, 1, device=device).fill_(EPSILON) # Add epsilon for stability

            # Aggregate contributions from cells assigned to edges connected to each node
            for node_idx in range(n_nodes):
                node_name = posterior.index_to_node.get(node_idx)
                if node_name is None: continue

                total_weight_for_node = torch.zeros(n_cells, device=device)

                # Sum weights from incoming edges (contribution as target 'v')
                for pred_name in traj_graph.G_traj.predecessors(node_name):
                    edge = (pred_name, node_name)
                    edge_idx = edge_to_index.get(edge)
                    if edge_idx is not None:
                        # Simple model: Cells on edge (u,v) contribute fully to node v mean
                        total_weight_for_node += q_eff[:, edge_idx]

                # Sum weights from outgoing edges (contribution as source 'u')
                for succ_name in traj_graph.G_traj.successors(node_name):
                    edge = (node_name, succ_name)
                    edge_idx = edge_to_index.get(edge)
                    if edge_idx is not None:
                         # Simple model: Cells on edge (u,v) contribute fully to node u mean
                        total_weight_for_node += q_eff[:, edge_idx]

                # Calculate weighted sum for node_idx
                node_total_weight = total_weight_for_node.sum()
                if node_total_weight > EPSILON:
                    weighted_X_sum = (X * total_weight_for_node.unsqueeze(1)).sum(dim=0)
                    g_numerator[node_idx] = weighted_X_sum
                    g_denominator[node_idx] = node_total_weight

            recalculated_g = g_numerator / g_denominator
            # Handle nodes with zero effective weight
            zero_weight_nodes = (g_denominator <= EPSILON).squeeze()
            if torch.any(zero_weight_nodes):
                global_mean_X = X.mean(dim=0)
                recalculated_g[zero_weight_nodes] = global_mean_X


            # 3. Recalculate sigma2 (Edge Variances) - Weighted variance of X for cells assigned to edge
            edge_denominator = q_eff.sum(dim=0).unsqueeze(1).clamp(min=EPSILON) # [N_edges, 1] Total weight per edge

            # Weighted mean: E[X_e] = sum_i(q_eff_ie * X_i) / sum_i(q_eff_ie)
            edge_mean_numerator = torch.matmul(q_eff.T, X) # [N_edges, N_genes] = [E, N] x [N, G]
            edge_mean = edge_mean_numerator / edge_denominator # [N_edges, N_genes]

            # Weighted variance: E[X_e^2] - (E[X_e])^2
            # E[X_e^2] = sum_i(q_eff_ie * X_i^2) / sum_i(q_eff_ie)
            edge_mean_sq_numerator = torch.matmul(q_eff.T, X**2) # [N_edges, N_genes]
            edge_mean_sq = edge_mean_sq_numerator / edge_denominator # [N_edges, N_genes]

            recalculated_sigma2 = (edge_mean_sq - edge_mean**2).clamp(min=EPSILON) # Ensure non-negative

            # Handle edges with zero weight
            zero_weight_edges = (edge_denominator <= EPSILON).squeeze()
            if torch.any(zero_weight_edges):
                recalculated_sigma2[zero_weight_edges] = 1.0 # Assign default variance 1.0

            # Add a small prior variance
            prior_var = self.config['recalc_prior_std']**2
            recalculated_sigma2 += prior_var

        # if self.config['verbose']: print("Recalculation finished.")
        return recalculated_g.detach(), recalculated_sigma2.detach()


    def optimize_on_graph(self, traj_graph, X, initial_params=None, run_config=None):
        """ Runs the VI optimization loop. """
        config = self.config.copy()
        if run_config: config.update(run_config)
        X = X.to(self.device)
        n_cells = X.shape[0]
        if config['verbose']: print("\n" + "=" * 20 + " Starting Optimization Run " + "=" * 20)
        if config['verbose']: print(f"Mode: {'Lagging' if config['use_lagging'] else 'Standard'}, Learn K: {config['learn_K']}, Minibatch: {config['minibatch']}, LR: {config['lr']}")

        try:
            # Initialize params. g and log_sigma2 will have requires_grad=False.
            g, K, log_sigma2, pi, posterior, belief_propagator, node_to_index, edge_to_index = \
                self._initialize_params_and_posterior(traj_graph, X, initial_params)
        except (RuntimeError, ValueError) as e:
            print(f"ERROR: Stopping optimization due to initialization error: {e}")
            return None, None, -float('inf')

        # --- Setup Optimizer ---
        # Posterior parameters are always learned
        posterior_params = [p for p in posterior.parameters() if p.requires_grad]
        # Generative parameters only include K if learn_K is True
        learnable_gen_params = []
        if K is not None and config['learn_K']:
            learnable_gen_params.append(K)

        if not learnable_gen_params and not posterior_params:
            print("ERROR: No parameters require gradients. Cannot optimize.")
            opt_params = {p_name: v.detach().cpu() if v is not None else None
                          for p_name, v in zip(['g','K','log_sigma2','pi'], [g,K,log_sigma2,pi])}
            return opt_params, posterior, -float('inf')

        optimizer_inf, optimizer_gen, optimizer = None, None, None
        if config['use_lagging']:
            optimizer_inf = torch.optim.Adam(posterior_params, lr=config['lr']) if posterior_params else None
            optimizer_gen = torch.optim.Adam(learnable_gen_params, lr=config['lr']) if learnable_gen_params else None
            if not optimizer_inf: print("Warning: Lagging VI - No posterior parameters to optimize.")
            if not optimizer_gen: print("Warning: Lagging VI - No learnable generative parameters (K) to optimize.")
            if not optimizer_inf and not optimizer_gen:
                 print("ERROR: Lagging VI - No optimizers could be created."); return None, None, -float('inf')
            if config['verbose']: print(f"Using Lagging VI (Inf Opt: {optimizer_inf is not None}, Gen Opt: {optimizer_gen is not None})")
        else: # Standard VI
            all_learnable_params = posterior_params + learnable_gen_params
            if not all_learnable_params:
                print("ERROR: Standard VI - No learnable parameters found."); return None, None, -float('inf')
            optimizer = torch.optim.Adam(all_learnable_params, lr=config['lr'])
            if config['verbose']: print(f"Using Standard VI optimizer ({len(all_learnable_params)} param groups)")


        # --- Training Loop ---
        num_epochs = config['eval_epochs'] if config.get('short_run', False) else config['num_epochs']
        if config['verbose']: print(f"Starting VI for {num_epochs} epochs...")

        metrics_history = []
        final_elbo_value = -float('inf')

        # --- Annealing start epoch (consider freezing K) ---
        anneal_start_epoch_actual = config['anneal_start_epoch']
        if not config['use_lagging'] and config['learn_K']:
            anneal_start_epoch_actual = max(config['anneal_start_epoch'], config['freeze_epochs'])

        # --- Freezing state for K (Standard VI) ---
        K_frozen_state = config['learn_K'] and (config['freeze_epochs'] > 0)

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_metrics_sum = defaultdict(float); epoch_elbo_sum = 0.0; epoch_l1k_sum = 0.0
            samples_processed_elbo = 0; samples_processed_nll = 0; samples_processed_l1k = 0

            # --- Annealing ---
            current_temperature = self._get_annealing_value(epoch, anneal_start_epoch_actual, config['anneal_epochs'], config['logit_temp_start'], config['logit_temp_end'])
            current_struct_prior_weight = self._get_annealing_value(epoch, anneal_start_epoch_actual, config['anneal_epochs'], config['struct_prior_anneal_start'], config['struct_prior_anneal_end'])

            # --- Freezing/Thawing K (Standard VI only, if learn_K=True) ---
            K_frozen_this_epoch = False
            if not config['use_lagging'] and config['learn_K'] and K is not None:
                 should_be_frozen = epoch < config['freeze_epochs']
                 if should_be_frozen != K_frozen_state: # State change needed
                     verb = "Freezing" if should_be_frozen else "Unfreezing"
                     if config['verbose'] and (epoch == 0 or epoch == config['freeze_epochs']):
                         print(f"Epoch {epoch}: {verb} parameter K.")
                     K.requires_grad_(not should_be_frozen)
                     # Re-create optimizer with potentially changed K status
                     current_learnable_gen = [K] if K.requires_grad else []
                     current_optim_params = posterior_params + current_learnable_gen
                     if current_optim_params:
                          optimizer = torch.optim.Adam(current_optim_params, lr=config['lr'])
                     else: optimizer = None # Only posterior params left
                     K_frozen_state = should_be_frozen
                 K_frozen_this_epoch = K_frozen_state


            # ================== OPTIMIZATION STEP ==================

            if config['use_lagging']:
                # === E-Step: Optimize Posterior ===
                if optimizer_inf:
                    posterior.train()
                    # Ensure K grad is off for E-step
                    if K is not None: K.requires_grad_(False)
                    e_step_batches = 0
                    for _inf_step in range(config['inference_steps']):
                        optimizer_inf.zero_grad(set_to_none=True)
                        for batch_idx in self._get_batches(X, config['batch_size'], config['minibatch']):
                            # --- Batch ELBO Calculation (E-step) ---
                            X_batch = X[batch_idx]; batch_size_actual = X_batch.shape[0]
                            if batch_size_actual == 0: continue
                            e_step_batches += 1
                            # Use DETACHED g, K, log_sigma2 for E-step ELBO
                            g_d, K_d, ls2_d = (p.detach() if p is not None else None for p in [g, K, log_sigma2])
                            sigma2_d = ls2_d.exp() if ls2_d is not None else None
                            pi_d = pi.detach() if pi is not None else None
                            elbo_inf, batch_metrics_inf = compute_elbo_batch(
                                X_batch, batch_idx, traj_graph, posterior, edge_to_index,
                                g=g_d, K=K_d, sigma2=sigma2_d, pi=pi_d,
                                belief_propagator=belief_propagator, n_samples=config['n_samples'],
                                kl_weight=config['kl_weight'], kl_p_weight=config['kl_p_weight'], t_cont_weight=config['t_cont_weight'],
                                branch_entropy_weight=config['branch_entropy_weight'],
                                temperature=current_temperature, struct_prior_weight=current_struct_prior_weight
                            )
                            loss_e = -elbo_inf
                            if torch.isnan(loss_e).any(): continue
                            loss_e_avg = loss_e / batch_size_actual
                            loss_e_avg.backward()
                            # --- Accumulate E-step Metrics ---
                            epoch_elbo_sum += elbo_inf.item()
                            for k, v_val in batch_metrics_inf.items():
                                if k not in ['q_e_raw', 'q_eff', 't_sampled', 'edge_idx_sampled', 'nll', 'l1_K']: # Exclude NLL/L1K (M-step)
                                     scalar_v = v_val.item() if isinstance(v_val, torch.Tensor) else float(v_val)
                                     epoch_metrics_sum[k] += scalar_v * batch_size_actual
                            samples_processed_elbo += batch_size_actual
                        if e_step_batches > 0: optimizer_inf.step()

                # === M-Step: Optimize K (if learnable) ===
                if optimizer_gen: # Only run if K is learnable
                    posterior.eval()
                    if K is not None: K.requires_grad_(True) # Ensure K grad is ON
                    # g and log_sigma2 grads remain OFF
                    m_step_batches = 0
                    for _m_step in range(config['generative_steps']):
                        optimizer_gen.zero_grad(set_to_none=True)
                        for batch_idx in self._get_batches(X, config['batch_size'], config['minibatch']):
                            # --- Batch NLL + L1(K) Calculation (M-step) ---
                            X_batch = X[batch_idx]; batch_size_actual = X_batch.shape[0]
                            if batch_size_actual == 0: continue
                            m_step_batches += 1
                            # --- Sample (No Grad) ---
                            with torch.no_grad():
                                # (Sampling logic remains the same as previous response)
                                raw_logits_batch = posterior.edge_logits[batch_idx]
                                q_e_batch = torch.softmax(raw_logits_batch / current_temperature, dim=1)
                                q_eff_batch = belief_propagator.diffuse(q_e_batch) if belief_propagator else q_e_batch
                                q_eff_batch = (q_eff_batch / q_eff_batch.sum(dim=1, keepdim=True).clamp(min=EPSILON)).clamp(min=EPSILON_LOG)
                                try: edge_idx_samples = torch.multinomial(q_eff_batch, num_samples=config['n_samples'], replacement=True)
                                except RuntimeError: continue
                                alpha_g = posterior.alpha[batch_idx].unsqueeze(1).expand(-1, config['n_samples'], -1).gather(2, edge_idx_samples.unsqueeze(-1)).squeeze(-1).clamp(min=EPSILON)
                                beta_g = posterior.beta[batch_idx].unsqueeze(1).expand(-1, config['n_samples'], -1).gather(2, edge_idx_samples.unsqueeze(-1)).squeeze(-1).clamp(min=EPSILON)
                                try: t_samples = torch.distributions.Beta(alpha_g, beta_g).rsample().clamp(min=EPSILON, max=1.0-EPSILON)
                                except ValueError: continue
                            # --- Reshape ---
                            flat_X = X_batch.unsqueeze(1).expand(-1, config['n_samples'], -1).reshape(-1, X.shape[1])
                            flat_edge_idx = edge_idx_samples.reshape(-1); flat_t = t_samples.reshape(-1)
                            if flat_X.numel() == 0: continue
                            # --- Calculate NLL (using current K, fixed g/sigma2) ---
                            # Use detached g and log_sigma2.exp()
                            g_d = g.detach() if g is not None else None
                            sigma2_d = log_sigma2.exp().detach() if log_sigma2 is not None else None
                            mean_nll = emission_nll(flat_X, flat_edge_idx, flat_t, g_d, K, sigma2_d, posterior.index_to_edge, node_to_index, pi=pi)
                            # --- Calculate L1 Penalty on K ---
                            l1k_penalty = compute_l1_penalty_K(K, config['l1_K_weight'])
                            l1k_penalty_value = l1k_penalty.item()
                            # --- Loss & Backward ---
                            loss_m = mean_nll + l1k_penalty # NLL avg + Global L1(K)
                            if torch.isnan(loss_m).any(): continue
                            loss_m.backward() # Calculate gradients w.r.t. K
                            # --- Accumulate M-step Metrics ---
                            epoch_metrics_sum['nll'] += mean_nll.item() * batch_size_actual
                            samples_processed_nll += batch_size_actual
                            if l1k_penalty_value > 0.0:
                                epoch_l1k_sum += l1k_penalty_value * batch_size_actual # L1 is global, associate with batch
                                samples_processed_l1k += batch_size_actual
                        if m_step_batches > 0: optimizer_gen.step()


            else: # ================ Standard VI Step ================
                if optimizer is None: continue
                posterior.train()
                # Set K grad based on freezing state
                if K is not None: K.requires_grad_(config['learn_K'] and not K_frozen_this_epoch)
                # g and log_sigma2 grads remain False
                optimizer.zero_grad(set_to_none=True)
                std_step_batches = 0
                for batch_idx in self._get_batches(X, config['batch_size'], config['minibatch']):
                     # --- Batch ELBO + L1(K) Calculation ---
                    X_batch = X[batch_idx]; batch_size_actual = X_batch.shape[0]
                    if batch_size_actual == 0: continue
                    std_step_batches += 1
                    # --- Calculate ELBO ---
                    # Use current K (potentially learnable), detached g/log_sigma2.exp()
                    g_d = g.detach() if g is not None else None
                    sigma2_d = log_sigma2.exp().detach() if log_sigma2 is not None else None
                    elbo, batch_metrics = compute_elbo_batch(
                        X_batch, batch_idx, traj_graph, posterior, edge_to_index,
                        g=g_d, K=K, sigma2=sigma2_d, pi=pi, # K might have grad
                        belief_propagator=belief_propagator, n_samples=config['n_samples'],
                        kl_weight=config['kl_weight'], kl_p_weight=config['kl_p_weight'], t_cont_weight=config['t_cont_weight'],
                        branch_entropy_weight=config['branch_entropy_weight'],
                        temperature=current_temperature, struct_prior_weight=current_struct_prior_weight
                    )
                    loss_elbo_part = -elbo
                    # --- Calculate L1 Penalty on K (if learnable and not frozen) ---
                    l1k_penalty = torch.tensor(0.0, device=device); l1k_penalty_value = 0.0
                    if config['l1_K_weight'] > 0.0 and config['learn_K'] and not K_frozen_this_epoch and K is not None:
                        l1k_penalty = compute_l1_penalty_K(K, config['l1_K_weight'])
                        l1k_penalty_value = l1k_penalty.item()
                    # --- Loss & Backward ---
                    total_batch_loss = loss_elbo_part + l1k_penalty
                    if torch.isnan(total_batch_loss).any(): continue
                    total_batch_loss_avg = total_batch_loss / batch_size_actual
                    total_batch_loss_avg.backward()
                    # --- Accumulate All Metrics ---
                    epoch_elbo_sum += elbo.item()
                    samples_processed_elbo += batch_size_actual
                    samples_processed_nll += batch_size_actual # NLL from ELBO
                    for k, v_val in batch_metrics.items():
                        if k not in ['q_e_raw', 'q_eff', 't_sampled', 'edge_idx_sampled']:
                             scalar_v = v_val.item() if isinstance(v_val, torch.Tensor) else float(v_val)
                             epoch_metrics_sum[k] += scalar_v * batch_size_actual
                    if l1k_penalty_value > 0.0:
                        epoch_l1k_sum += l1k_penalty_value * batch_size_actual
                        samples_processed_l1k += batch_size_actual
                if std_step_batches > 0: optimizer.step()

            # ================== RECALCULATE g/sigma2 ==================
            try:
                recalc_g, recalc_sigma2 = self._recalculate_g_sigma2(
                    X, posterior, traj_graph, node_to_index, edge_to_index
                )
                # Update parameters in-place (detached)
                if g is not None: g.data.copy_(recalc_g)
                if log_sigma2 is not None: log_sigma2.data.copy_(torch.log(recalc_sigma2.clamp(min=EPSILON)))
            except Exception as e:
                print(f"ERROR during g/sigma2 recalculation: {e}. Using previous values for epoch {epoch}.")
                # import traceback; traceback.print_exc() # Optional: print traceback


            # ================== END OF EPOCH - AVERAGING & LOGGING ==================
            epoch_time = time.time() - epoch_start_time
            current_epoch_metrics_avg = {}

            # Average ELBO and related terms (KL, t_cont, etc.)
            if samples_processed_elbo > 0:
                avg_elbo = epoch_elbo_sum / samples_processed_elbo
                final_elbo_value = avg_elbo
                for k, v_sum in epoch_metrics_sum.items():
                    if k not in ['nll', 'l1_K']: # Handle NLL/L1K below
                         current_epoch_metrics_avg[k] = v_sum / samples_processed_elbo
            else: avg_elbo = 0.0; final_elbo_value = max(final_elbo_value, -float('inf'))

            # Average NLL
            if samples_processed_nll > 0:
                 current_epoch_metrics_avg['nll'] = epoch_metrics_sum.get('nll', 0.0) / samples_processed_nll
            else: current_epoch_metrics_avg['nll'] = 0.0

            # Average L1(K) penalty
            if samples_processed_l1k > 0:
                current_epoch_metrics_avg['l1_K'] = epoch_l1k_sum / samples_processed_l1k
            else: current_epoch_metrics_avg['l1_K'] = 0.0

            metrics_history.append(current_epoch_metrics_avg.copy())

            # Logging
            if config['verbose'] and (epoch % 10 == 0 or epoch == num_epochs - 1):
                 last_batch_metrics = {} # Get last batch metrics for entropy logging
                 if 'batch_metrics' in locals(): last_batch_metrics = batch_metrics
                 elif 'batch_metrics_inf' in locals(): last_batch_metrics = batch_metrics_inf
                 current_epoch_metrics_avg['q_e_raw_last_batch'] = last_batch_metrics.get('q_e_raw')
                 current_epoch_metrics_avg['q_eff_last_batch'] = last_batch_metrics.get('q_eff')
                 self._log_stats(epoch, avg_elbo, current_epoch_metrics_avg, epoch_time, temp=current_temperature, struct_weight=current_struct_prior_weight)


        # --- End of Training Loop ---
        if config['verbose']: print(f"Optimization finished. Final ELBO: {final_elbo_value:.4f}")

        # --- Return Results ---
        optimized_params = {
            'g': g.detach().cpu() if g is not None else None,
            'K': K.detach().cpu() if K is not None else None,
            'log_sigma2': log_sigma2.detach().cpu() if log_sigma2 is not None else None,
            'pi': pi.detach().cpu() if pi is not None else None
        }
        posterior.eval(); [p.requires_grad_(False) for p in posterior.parameters()]
        return optimized_params, posterior, final_elbo_value

    # --- _log_stats (updated format) ---
    def _log_stats(self, epoch, elbo_value, metrics_avg, elapsed, temp=1.0, struct_weight=1.0):
        """ Logs averaged metrics consistently. """
        nll = metrics_avg.get('nll', 0.0); kl_t = metrics_avg.get('kl_t', 0.0)
        kl_p = metrics_avg.get('kl_p', 0.0); t_cont = metrics_avg.get('t_cont', 0.0)
        b_ent = metrics_avg.get('branch_entropy', 0.0); s_kl = metrics_avg.get('struct_kl', 0.0)
        l1k = metrics_avg.get('l1_K', 0.0) # Use L1(K) now
        q_eff = metrics_avg.get("q_eff_last_batch", None); q_raw = metrics_avg.get("q_e_raw_last_batch", None)

        def calc_ent(probs):
            if not isinstance(probs, torch.Tensor) or probs.dim()!=2 or probs.numel()==0: return "N/A"
            try:
                probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=EPSILON)
                entr = -(probs.clamp(min=EPSILON_LOG) * torch.log(probs.clamp(min=EPSILON_LOG))).sum(dim=1)
                return f"{entr[torch.isfinite(entr)].mean().item():.3f}" if torch.isfinite(entr).any() else "NaN"
            except Exception: return "Err"

        ent_q_eff, ent_q_raw = calc_ent(q_eff), calc_ent(q_raw)
        # Updated format string for brevity and clarity
        log_items = [
            f"[Ep {epoch:>3}]", f"ELBO:{elbo_value:.3e}", f"NLL:{nll:.3e}",
            f"KLt:{kl_t:.2f}", f"KLp:{kl_p:.2f}", f"Tcont:{t_cont:.2f}",
            f"L1(K):{l1k:.2e}", # Log L1(K)
            f"Bent:{b_ent:.2f}", f"Skl:{s_kl:.2f}",
            f"Eq*:{ent_q_eff}", f"Eq:{ent_q_raw}",
            f"T:{temp:.2f}", f"Wstr:{struct_weight:.2f}", f"Time:{elapsed:.1f}s"
        ]
        print(" | ".join(log_items))

    # --- _get_batches (Keep as is) ---
    def _get_batches(self, X, batch_size, minibatch_flag):
        N = X.shape[0]
        if N == 0: yield torch.tensor([], dtype=torch.long, device=X.device); return
        if minibatch_flag and batch_size > 0 and batch_size < N:
            perm = torch.randperm(N, device=X.device)
            for i in range(0, N, batch_size): yield perm[i:min(i + batch_size, N)]
        else: yield torch.arange(N, device=X.device)