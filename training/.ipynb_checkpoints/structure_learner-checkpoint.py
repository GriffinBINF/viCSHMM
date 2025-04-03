import torch
import numpy as np
import networkx as nx
from models.emission import emission_nll # Need this for mu calculation logic
from models.posterior import TreeVariationalPosterior # For type hinting
from utils.constants import EPSILON, EPSILON_LOG
import time
import traceback

# Import VIRunner from the same directory
from .loop import VIRunner

# Optional: If seeding posterior from assignments is needed later
# from utils.inference import initialize_beta_from_cell_assignment


class TrajectoryStructureLearner:
    def __init__(self, initial_graph, data_X, vi_runner: VIRunner, config=None):
        """
        Initializes the TrajectoryStructureLearner.

        Args:
            initial_graph (TrajectoryGraph): The starting trajectory graph object.
            data_X (torch.Tensor): Expression data [n_cells, n_genes].
            vi_runner (VIRunner): An instance of the VIRunner class.
            config (dict, optional): Configuration dictionary for thresholds, penalties, etc.
        """
        if not hasattr(initial_graph, 'G_traj') or not isinstance(initial_graph.G_traj, nx.DiGraph):
             raise ValueError("initial_graph must be a TrajectoryGraph object with a valid G_traj attribute.")

        self.current_graph = initial_graph # This will be modified if structure changes
        self.data_X = data_X.to(vi_runner.device) # Ensure data is on the correct device
        self.vi_runner = vi_runner
        self.device = vi_runner.device # Inherit device from runner

        # --- Default Configuration ---
        self.config = {
            "max_structure_iterations": 1, # Default to 1 for Phase 1 (analyze -> optimize -> analyze)
            "analysis_thresholds": {
                "time_boundary": 0.1,    # Mode < 0.1 or > 0.9 defines boundary cells
                "discontinuity": 1.0,    # Propose SPLIT if D(v) > this
                "prune_mass_fraction": 0.05, # Propose PRUNE if Mass(e) < this * avg_edge_mass
                "merge_similarity": 0.5, # Propose MERGE if Sim(u,v) < this (Note: Low similarity means *more* likely to merge?) - Let's flip: merge if ||g_u - g_v|| is SMALLER than threshold
                "min_cells_for_discontinuity": 10 # Minimum cells needed at boundary to calculate D(v)
            },
            "structure_modification": { # Config for future phases
                "enabled": False, # Only analyze in Phase 1
                "max_proposals_to_evaluate": 3,
                "evaluation_epochs": 20 # Short run for evaluating modifications
            },
            "verbose": True,
        }
        if config:
            # Simple update for top-level keys
            self.config.update({k: v for k, v in config.items() if k not in ['analysis_thresholds', 'structure_modification']})
            # Merge nested dictionaries carefully
            if 'analysis_thresholds' in config:
                self.config['analysis_thresholds'].update(config['analysis_thresholds'])
            if 'structure_modification' in config:
                 self.config['structure_modification'].update(config['structure_modification'])

        self.history = [] # Stores list of dicts: {iteration, graph_snapshot, params, posterior, analysis_results, elbo}

    def _get_initial_components(self, graph_to_analyze):
        """Generates reasonable 'best guess' parameters and posterior for analysis BEFORE training."""
        if self.config['verbose']: print("Generating initial components for pre-optimization analysis...")
        try:
            # Use the runner's initializer - it handles from scratch or provided params
            # We want from scratch here, so pass initial_params=None
            # It returns 8 values now
            g, K, log_sigma2, pi, posterior, _belief_propagator, _node_to_index, edge_to_index = \
                self.vi_runner._initialize_params_and_posterior(
                    graph_to_analyze, self.data_X, initial_params=None
                )

            # Check if initialization succeeded
            if g is None or K is None or log_sigma2 is None or posterior is None:
                 raise RuntimeError("VI Runner initialization returned None for essential parameters/posterior.")

            initial_params = {
                'g': g.detach(), 'K': K.detach(), 'log_sigma2': log_sigma2.detach(), 'pi': pi.detach() if pi is not None else None
            }

            # Seed posterior to a vague initial state (more informative than random)
            with torch.no_grad():
                 # Set edge logits to be nearly uniform (zeros) -> uniform softmax
                 if hasattr(posterior, 'edge_logits') and posterior.edge_logits is not None:
                    posterior.edge_logits.zero_()

                 # Set Beta distribution centered around 0.5 (alpha=beta approx)
                 # alpha=beta=2 gives mean=0.5, variance=1/12 approx
                 if hasattr(posterior, 'alpha') and posterior.alpha is not None:
                     posterior.alpha.fill_(2.0)
                 if hasattr(posterior, 'beta') and posterior.beta is not None:
                     posterior.beta.fill_(2.0)
                 # Ensure requires_grad is False for this analysis posterior
                 for p in posterior.parameters(): p.requires_grad_(False)

            if self.config['verbose']: print("Initial components generated successfully.")
            return initial_params, posterior
        except Exception as e:
            print(f"ERROR during initial component generation: {type(e).__name__}: {e}")
            traceback.print_exc() # Print full traceback for debugging
            return None, None


    def learn_structure(self):
        """
        Main loop for learning structure.
        Phase 1: Analyze initial -> Optimize -> Analyze final.
        Phase 2+: Propose -> Evaluate -> Apply -> Optimize -> Analyze ... (Not implemented yet)
        """
        print("\n" + "*"*15 + " Starting Structure Learning Process " + "*"*15)

        current_graph = self.current_graph # Start with the initial graph

        for struct_iter in range(self.config['max_structure_iterations']):
            print(f"\n--- Structure Learning Iteration {struct_iter} ---")
            graph_nodes = current_graph.G_traj.number_of_nodes()
            graph_edges = current_graph.G_traj.number_of_edges()
            print(f"Current Graph: {graph_nodes} nodes, {graph_edges} edges")

            # --- Optional: Analyze structure *before* this iteration's optimization ---
            # Useful for seeing effect of modifications if struct_iter > 0
            if struct_iter == 0: # Only do detailed pre-analysis on the very first iteration
                print("\n--- Analyzing Initial Structure (Before Optimization) ---")
                initial_params, initial_posterior = self._get_initial_components(current_graph)
                if initial_params and initial_posterior:
                    initial_analysis_results = self.analyze_structure(
                        current_graph, initial_params, initial_posterior
                    )
                    self.log_analysis("Initial Analysis", initial_analysis_results)
                else:
                    print("Skipping initial analysis due to component generation error.")
                    initial_analysis_results = None # Mark as failed

            # --- 1. Optimize parameters for the current graph ---
            print(f"\n--- Optimizing Parameters for Iteration {struct_iter} ---")
            # Run optimization using the VI runner
            # Use initial_params=None so runner initializes fresh or from graph
            # Pass current_graph
            start_optim_time = time.time()
            opt_params, opt_posterior, final_elbo = self.vi_runner.optimize_on_graph(
                current_graph, self.data_X, initial_params=None # Runner handles initialization
            )
            optim_time = time.time() - start_optim_time
            print(f"Optimization completed in {optim_time:.2f}s. Final ELBO: {final_elbo:.4f}")

            if opt_params is None or opt_posterior is None:
                print("ERROR: VI optimization failed in structure iteration {struct_iter}. Stopping.")
                # Store failure state?
                self.history.append({
                    "iteration": struct_iter, "status": "Optimization Failed",
                    "graph_nodes": graph_nodes, "graph_edges": graph_edges
                 })
                # Return the *last successful* state if available, or initial graph
                last_successful_graph = self.history[-2]['graph_snapshot'] if len(self.history) > 1 and 'graph_snapshot' in self.history[-2] else self.current_graph
                return last_successful_graph, self.history

            # Update current state with optimized results
            current_params = opt_params # Keep params on CPU
            current_posterior = opt_posterior # Posterior is already detached, on CPU/Device? Runner should return CPU state? Let's assume CPU for params, device for posterior.
            # Ensure posterior is on the correct device for analysis
            current_posterior.to(self.device)
            # Ensure params are on device for analysis
            current_params_device = {k: v.to(self.device) if v is not None else None for k, v in current_params.items()}


            # --- 2. Analyze the structure *after* optimization ---
            print("\n--- Analyzing Structure (After Optimization) ---")
            try:
                post_analysis_results = self.analyze_structure(
                    current_graph, current_params_device, current_posterior
                )
                self.log_analysis(f"Post-Optimization Analysis (Iter {struct_iter})", post_analysis_results)
            except Exception as e:
                 print(f"ERROR during post-optimization analysis: {e}")
                 traceback.print_exc()
                 post_analysis_results = {"error": str(e)} # Store error info


            # --- Store History for this iteration ---
            history_entry = {
                "iteration": struct_iter,
                "status": "Completed",
                "graph_snapshot": current_graph, # Store the graph used for this iteration
                "graph_nodes": graph_nodes,
                "graph_edges": graph_edges,
                "final_elbo": final_elbo,
                "params_optimized": current_params, # Store CPU params
                "posterior_optimized": current_posterior, # Store posterior (likely on device)
                "analysis_post_optim": post_analysis_results,
                "optimization_time": optim_time
            }
            # Add initial analysis only for the first iteration's history
            if struct_iter == 0 and initial_analysis_results:
                 history_entry["analysis_pre_optim"] = initial_analysis_results
            self.history.append(history_entry)


            # --- Phase 2+: Propose, Evaluate, Apply Modifications ---
            if not self.config['structure_modification']['enabled'] or struct_iter >= self.config['max_structure_iterations'] -1:
                 if self.config['verbose']: print(f"Structure modification disabled or max iterations reached. Stopping.")
                 break # End loop

            print(f"\n--- Proposing/Evaluating Modifications for Iteration {struct_iter} ---")
            proposals = post_analysis_results.get('proposals', [])
            if not proposals:
                print("No modifications proposed based on analysis. Structure converged.")
                break

            # --- Evaluate Top N Proposals (Placeholder) ---
            # This requires:
            # 1. A way to apply a proposal to create a candidate graph.
            # 2. A way to initialize parameters for the candidate graph.
            # 3. Running a short optimization (e.g., using vi_runner with short_run=True).
            # 4. Comparing the resulting ELBO or another metric (e.g., BIC/AIC).
            print("Evaluating proposals (NOT IMPLEMENTED YET)...")
            best_modification_proposal = None # Assume no modification is chosen for now

            if best_modification_proposal:
                 print(f"Selected modification: {best_modification_proposal['type']} on {best_modification_proposal.get('node') or best_modification_proposal.get('edge') or best_modification_proposal.get('nodes')}")
                 try:
                     # Apply modification needs careful implementation
                     # new_graph, new_initial_params = self.apply_modification(current_graph, current_params_device, best_modification_proposal)
                     # current_graph = new_graph
                     # print("Modification applied. Proceeding to next optimization iteration.")
                     print("Apply modification (NOT IMPLEMENTED YET)...")
                     # For now, we stop after one iteration if mods are enabled but not implemented
                     break
                 except Exception as e:
                     print(f"ERROR applying modification: {e}. Stopping.")
                     break
            else:
                 print("No modification selected or improved score. Structure converged.")
                 break # Stop if no good modification found

        # --- End of Structure Learning Loop ---
        print("\n" + "*"*15 + " Structure Learning Process Complete " + "*"*15)

        # Return the final graph state and history
        final_graph = self.history[-1]['graph_snapshot'] if self.history else self.current_graph
        return final_graph, self.history


    # ------------------------------------------
    # ----- Analysis Methods (Core Logic) ----
    # ------------------------------------------

    def _calculate_mu(self, edge_idx_tensor, t_tensor, g, K, node_to_index, index_to_edge):
        """
        Calculates the expected expression mu = g_b + (g_a - g_b) * exp(-K*t).
        Handles potential errors in index resolution.

        Returns:
            Tensor [batch_size, n_genes] or Tensor filled with NaNs on error.
        """
        device = g.device
        if edge_idx_tensor.ndim == 0: # Handle scalar input
             edge_idx_tensor = edge_idx_tensor.unsqueeze(0)
             t_tensor = t_tensor.unsqueeze(0)

        batch_size = edge_idx_tensor.shape[0]
        n_genes = g.shape[1]
        n_nodes = g.shape[0]
        n_edges = K.shape[0]

        # Ensure indices are long type
        edge_idx_long = edge_idx_tensor.long()

        # Resolve node indices (vectorized, with error checking)
        u_idx_list, v_idx_list = [], []
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            edge_i = edge_idx_long[i].item()
            # Check edge index bounds first
            if not (0 <= edge_i < n_edges):
                 # print(f"Error: Edge index {edge_i} out of bounds [0, {n_edges-1}] in _calculate_mu. Masking.")
                 valid_mask[i] = False
                 u_idx_list.append(0); v_idx_list.append(0) # Dummy append
                 continue

            edge = index_to_edge.get(edge_i)
            if not edge:
                 # print(f"Error: Edge index {edge_i} not found in index_to_edge map in _calculate_mu. Masking.")
                 valid_mask[i] = False
                 u_idx_list.append(0); v_idx_list.append(0)
                 continue

            u_name, v_name = edge
            u_idx = node_to_index.get(u_name, -1) # Use get with default -1
            v_idx = node_to_index.get(v_name, -1)

            # Check node index bounds
            if not (0 <= u_idx < n_nodes and 0 <= v_idx < n_nodes):
                # print(f"Error: Resolved node index out of bounds for edge {edge} (u={u_idx}, v={v_idx}). Max node idx={n_nodes-1}. Masking.")
                valid_mask[i] = False
                u_idx_list.append(0); v_idx_list.append(0)
            else:
                u_idx_list.append(u_idx)
                v_idx_list.append(v_idx)

        if not torch.all(valid_mask):
            # print(f"Warning: {torch.sum(~valid_mask)} items masked during node index resolution in _calculate_mu.")
            if torch.sum(valid_mask) == 0:
                print("Error: All items masked in _calculate_mu. Returning NaNs.")
                return torch.full((batch_size, n_genes), float('nan'), device=device)
            # Filter inputs (create new tensors for valid items)
            edge_idx_long = edge_idx_long[valid_mask]
            t_tensor = t_tensor[valid_mask]
            u_idx_list = [idx for i, idx in enumerate(u_idx_list) if valid_mask[i]]
            v_idx_list = [idx for i, idx in enumerate(v_idx_list) if valid_mask[i]]
            # Note: We need to return a tensor of the original batch_size, filling invalid entries with NaN
            # Alternatively, return only valid mu and the mask. Let's fill with NaN.

        u_idx_tensor = torch.tensor(u_idx_list, device=device, dtype=torch.long)
        v_idx_tensor = torch.tensor(v_idx_list, device=device, dtype=torch.long)

        # Gather parameters for the valid indices
        g_a = g[u_idx_tensor] # [n_valid, G]
        g_b = g[v_idx_tensor] # [n_valid, G]
        K_edge = K[edge_idx_long] # [n_valid, G]

        # Ensure t_tensor has correct shape [n_valid, 1] for broadcasting
        t_col = t_tensor.reshape(-1, 1)

        # Calculate mu for valid entries
        mu_valid = g_b + (g_a - g_b) * torch.exp(-K_edge * t_col) # [n_valid, G]

        # Create the full output tensor and fill with valid results / NaNs
        mu_full = torch.full((batch_size, n_genes), float('nan'), device=device)
        mu_full[valid_mask] = mu_valid

        return mu_full


    def _calculate_boundary_discontinuity(self, graph, params, posterior):
        """ Calculates D(v) = || mean(mu_end) - mean(mu_start) || for non-root/leaf nodes. """
        if self.config['verbose']: print("Calculating boundary discontinuity...")
        g = params['g'] # Assumed on device
        K = params['K'] # Assumed on device
        if g is None or K is None:
             print("Warning: Missing 'g' or 'K' in params for discontinuity calc. Returning empty.")
             return {}

        alpha = posterior.alpha.to(self.device)
        beta = posterior.beta.to(self.device)
        edge_logits = posterior.edge_logits.to(self.device)

        # Use mappings consistent with the posterior
        node_to_index = posterior.traj.node_to_index
        index_to_node = {v: k for k, v in node_to_index.items()}
        edge_to_index = posterior.edge_to_index
        index_to_edge = posterior.index_to_edge
        G_networkx = posterior.traj.G_traj # The networkx graph

        n_cells = alpha.shape[0]
        cell_indices = torch.arange(n_cells, device=self.device)

        discontinuity_scores = {}
        boundary_thresh = self.config['analysis_thresholds']['time_boundary']
        min_cells_thresh = self.config['analysis_thresholds']['min_cells_for_discontinuity']

        # --- Calculate time mode robustly ---
        t_mode_num = alpha - 1
        t_mode_den = alpha + beta - 2
        # Mode is 0 if alpha <= 1, 1 if beta <= 1, otherwise (a-1)/(a+b-2)
        # Initialize mode based on boundaries
        t_mode = torch.zeros_like(alpha)
        t_mode[alpha > 1] = (t_mode_num[alpha > 1] / t_mode_den[alpha > 1].clamp(min=EPSILON)) # Avoid div by zero if a+b=2
        t_mode[(alpha <= 1)] = 0.0
        t_mode[(beta <= 1) & (alpha > 1)] = 1.0 # Prioritize beta=1 case
        t_mode = torch.clamp(t_mode, 0.0, 1.0) # Final clamp

        # Get highest probability edge assignment for each cell
        with torch.no_grad(): # No need for gradients here
            q_e = torch.softmax(edge_logits, dim=1)
            max_prob_edge_idx = torch.argmax(q_e, dim=1) # [n_cells]

        # Identify nodes that are intermediate points in the graph structure
        nodes_to_analyze = [
            node_name for node_name in G_networkx.nodes()
            if G_networkx.in_degree(node_name) > 0 and G_networkx.out_degree(node_name) > 0
            and node_name in node_to_index # Make sure the node exists in the model
        ]
        if not nodes_to_analyze:
             print("No intermediate nodes found in the graph for discontinuity analysis.")
             return {}

        # Pre-calculate mu for all cells, modes, and assigned edges (potential speedup)
        # This might be memory intensive if n_cells is large
        # mu_all_cells = self._calculate_mu(max_prob_edge_idx, t_mode[cell_indices, max_prob_edge_idx], g, K, node_to_index, index_to_edge)

        for node_name in nodes_to_analyze:
            # Find incoming and outgoing edge indices *present in the model*
            incoming_edge_indices = [
                edge_idx for pred_name in G_networkx.predecessors(node_name)
                if (edge_idx := edge_to_index.get((pred_name, node_name))) is not None
            ]
            outgoing_edge_indices = [
                edge_idx for succ_name in G_networkx.successors(node_name)
                if (edge_idx := edge_to_index.get((node_name, succ_name))) is not None
            ]

            if not incoming_edge_indices or not outgoing_edge_indices:
                # print(f"Node {node_name} has no valid incoming/outgoing edges in the model. Skipping.")
                discontinuity_scores[node_name] = 0.0 # No discontinuity if effectively root/leaf
                continue

            # Find cells primarily assigned to these edges AND near the boundary time
            cells_at_end_of_incoming = torch.zeros(n_cells, dtype=torch.bool, device=self.device)
            cells_at_start_of_outgoing = torch.zeros(n_cells, dtype=torch.bool, device=self.device)

            for edge_i_idx in incoming_edge_indices:
                assigned_to_inc_edge = (max_prob_edge_idx == edge_i_idx)
                # Use mode specific to this edge for the check
                near_end_time = (t_mode[cell_indices, edge_i_idx] > (1.0 - boundary_thresh))
                cells_at_end_of_incoming |= (assigned_to_inc_edge & near_end_time)

            for edge_o_idx in outgoing_edge_indices:
                assigned_to_out_edge = (max_prob_edge_idx == edge_o_idx)
                near_start_time = (t_mode[cell_indices, edge_o_idx] < boundary_thresh)
                cells_at_start_of_outgoing |= (assigned_to_out_edge & near_start_time)

            end_indices = cell_indices[cells_at_end_of_incoming]
            start_indices = cell_indices[cells_at_start_of_outgoing]

            # Check if enough cells meet the criteria
            if len(end_indices) >= min_cells_thresh and len(start_indices) >= min_cells_thresh:
                # Calculate average mu for these specific boundary cells
                # Get the edge and time mode corresponding to each cell in the boundary sets
                edge_idx_end = max_prob_edge_idx[end_indices]
                t_mode_end = t_mode[end_indices, edge_idx_end] # Get mode for the assigned edge
                mu_end = self._calculate_mu(edge_idx_end, t_mode_end, g, K, node_to_index, index_to_edge)

                edge_idx_start = max_prob_edge_idx[start_indices]
                t_mode_start = t_mode[start_indices, edge_idx_start]
                mu_start = self._calculate_mu(edge_idx_start, t_mode_start, g, K, node_to_index, index_to_edge)

                # Filter out NaNs before averaging
                mu_end_valid = mu_end[~torch.any(torch.isnan(mu_end), dim=1)]
                mu_start_valid = mu_start[~torch.any(torch.isnan(mu_start), dim=1)]

                if mu_end_valid.shape[0] < min_cells_thresh or mu_start_valid.shape[0] < min_cells_thresh:
                    # print(f"Node {node_name}: Not enough valid mu vectors after NaN filtering (end={mu_end_valid.shape[0]}, start={mu_start_valid.shape[0]}). Score=0.")
                    discontinuity_scores[node_name] = 0.0
                    continue

                # Calculate average expression vectors from valid mus
                mean_expr_end = mu_end_valid.mean(dim=0)
                mean_expr_start = mu_start_valid.mean(dim=0)

                # Calculate L2 distance
                score = torch.linalg.norm(mean_expr_end - mean_expr_start).item()
                discontinuity_scores[node_name] = score
            else:
                # Not enough cells at boundary
                # print(f"Node {node_name}: Not enough cells at boundary (end={len(end_indices)}, start={len(start_indices)}). Min required={min_cells_thresh}. Score=0.")
                discontinuity_scores[node_name] = 0.0 # Assign 0 if insufficient data

        if self.config['verbose']: print("Boundary discontinuity calculation finished.")
        return discontinuity_scores

    def _calculate_edge_usage(self, posterior):
        """ Calculates Mass(e) = sum_cells q(assign=e | cell) for each edge. """
        if self.config['verbose']: print("Calculating edge usage...")
        edge_logits = posterior.edge_logits.to(self.device)
        with torch.no_grad():
            q_e = torch.softmax(edge_logits, dim=1)
            mass_e = q_e.sum(dim=0) # Sum probability mass over all cells for each edge
        edge_usage = {
            edge: mass_e[idx].item()
            for edge, idx in posterior.edge_to_index.items()
        }
        if self.config['verbose']: print("Edge usage calculation finished.")
        return edge_usage

    def _calculate_node_similarity(self, graph, params):
        """ Calculates Sim(u, v) = ||g[u] - g[v]|| for connected nodes. """
        if self.config['verbose']: print("Calculating node similarity...")
        g = params.get('g')
        if g is None:
             print("Warning: Missing 'g' in params for node similarity. Returning empty.")
             return {}
        g = g.to(self.device) # Ensure g is on device

        node_to_index = graph.node_to_index # Use node mapping from the graph being analyzed
        G_networkx = graph.G_traj
        similarity_scores = {}
        n_nodes = g.shape[0]

        for u_name, v_name in G_networkx.edges():
             u_idx = node_to_index.get(u_name, -1)
             v_idx = node_to_index.get(v_name, -1)

             # Ensure indices are valid for g
             if 0 <= u_idx < n_nodes and 0 <= v_idx < n_nodes:
                 score = torch.linalg.norm(g[u_idx] - g[v_idx]).item()
                 similarity_scores[(u_name, v_name)] = score
             else:
                  # print(f"Warning: Invalid node index for edge ({u_name}, {v_name}) in _calculate_node_similarity.")
                  similarity_scores[(u_name, v_name)] = float('nan') # Indicate error

        if self.config['verbose']: print("Node similarity calculation finished.")
        return similarity_scores

    def _calculate_unassigned_fraction(self, posterior):
        """ Calculates fraction assigned to 'Unassigned' state (Phase 2+). """
        # This functionality depends on adding an extra column to edge_logits
        # for the unassigned state during posterior initialization and training.
        # For Phase 1, assume no unassigned state exists.
        return {"mean_fraction": 0.0, "high_fraction_cell_indices": []}


    def analyze_structure(self, graph, params, posterior):
        """
        Performs analysis using optimized parameters and posterior.
        Generates proposals for structural modifications based on thresholds.
        """
        if self.config['verbose']: print("--- Starting Structure Analysis ---")
        analysis_results = {}
        start_time = time.time()

        # 1. Calculate Metrics
        analysis_results['discontinuity_scores'] = self._calculate_boundary_discontinuity(graph, params, posterior)
        analysis_results['edge_usage'] = self._calculate_edge_usage(posterior)
        analysis_results['node_similarity'] = self._calculate_node_similarity(graph, params)
        analysis_results['unassigned_info'] = self._calculate_unassigned_fraction(posterior) # Placeholder

        # 2. Generate Proposals based on Metrics and Thresholds
        proposals = []
        thresholds = self.config['analysis_thresholds']
        edge_to_index = posterior.edge_to_index # Use posterior's mapping

        # Proposal: Split Node (High Discontinuity)
        disc_thresh = thresholds['discontinuity']
        for node_name, score in analysis_results.get('discontinuity_scores', {}).items():
            if not np.isnan(score) and score > disc_thresh:
                proposals.append({
                    'type': 'SPLIT_NODE', 'target': node_name,
                    'reason': 'High Discontinuity', 'metric_value': score, 'threshold': disc_thresh
                })

        # Proposal: Prune Edge (Low Usage)
        all_edge_mass = list(analysis_results.get('edge_usage', {}).values())
        if all_edge_mass:
            avg_usage = np.mean(all_edge_mass)
            usage_threshold = thresholds['prune_mass_fraction'] * avg_usage
            for edge, mass in analysis_results['edge_usage'].items():
                 # Add checks: Don't prune if it disconnects graph? Don't prune sole incoming/outgoing?
                 u_name, v_name = edge
                 # Check if nodes still exist in the graph definition
                 if u_name in graph.node_to_index and v_name in graph.node_to_index:
                    if mass < usage_threshold:
                         proposals.append({
                             'type': 'PRUNE_EDGE', 'target': edge,
                             'reason': 'Low Usage', 'metric_value': mass, 'threshold': usage_threshold
                         })
        else: print("Warning: No edge usage data to propose pruning.")


        # Proposal: Merge Nodes (Low Similarity = High feature similarity)
        # We merge if ||g_u - g_v|| is SMALLER than the threshold.
        sim_thresh = thresholds['merge_similarity']
        for edge, similarity in analysis_results.get('node_similarity', {}).items():
             u_name, v_name = edge
             # Avoid merging if similarity is NaN or high
             if not np.isnan(similarity) and similarity < sim_thresh:
                 # Add checks: Don't merge nodes involved in high discontinuity?
                 # Ensure nodes still exist
                 if u_name in graph.node_to_index and v_name in graph.node_to_index:
                     proposals.append({
                         'type': 'MERGE_NODES', 'target': edge, # Edge tuple represents nodes
                         'reason': 'High Similarity', 'metric_value': similarity, 'threshold': sim_thresh
                     })

        # --- Rank proposals (example logic) ---
        # Prioritize splits > merges > prunes? Or based on metric severity?
        def sort_key(p):
            order = {'SPLIT_NODE': 0, 'MERGE_NODES': 1, 'PRUNE_EDGE': 2}
            # Severity: Higher is "worse" for discontinuity, lower is "worse" for usage/similarity
            severity = 0
            if p['type'] == 'SPLIT_NODE': severity = p['metric_value'] / p['threshold'] # Ratio > 1 is bad
            if p['type'] == 'PRUNE_EDGE': severity = p['threshold'] / max(p['metric_value'], 1e-9) # Ratio > 1 is bad
            if p['type'] == 'MERGE_NODES': severity = p['threshold'] / max(p['metric_value'], 1e-9)# Ratio > 1 is bad (similarity is low)
            return (order.get(p['type'], 9), -severity) # Sort by type, then descending severity

        proposals.sort(key=sort_key)
        analysis_results['proposals'] = proposals
        analysis_time = time.time() - start_time
        if self.config['verbose']: print(f"--- Structure Analysis Finished ({analysis_time:.2f}s) ---")

        return analysis_results

    def log_analysis(self, analysis_name, analysis_results):
        """ Prints a summary of the analysis results. """
        print(f"\n--- Structure Analysis Report: {analysis_name} ---")
        if "error" in analysis_results:
            print(f"  ERROR during analysis: {analysis_results['error']}")
            return

        print(" Boundary Discontinuity Scores (D(v)):")
        scores = analysis_results.get('discontinuity_scores')
        if scores:
            for node, score in sorted(scores.items()): print(f"  Node {node}: {score:.4f}")
        else: print("  (No scores calculated or available)")

        print("\n Edge Usage (Mass(e)):")
        usage = analysis_results.get('edge_usage')
        if usage:
            for edge, mass in sorted(usage.items()): print(f"  Edge {edge}: {mass:.4f}")
        else: print("  (No usage calculated or available)")

        print("\n Node Similarity (||g_u-g_v||):")
        similarity = analysis_results.get('node_similarity')
        if similarity:
             for edge, sim in sorted(similarity.items()): print(f"  Edge {edge}: {sim:.4f}")
        else: print("  (No similarity calculated or available)")

        print("\n Proposed Modifications (Ranked):")
        proposals = analysis_results.get('proposals')
        if proposals:
            for i, p in enumerate(proposals):
                target = p.get('target', 'N/A')
                reason = p.get('reason', '')
                print(f"  {i+1}. Type: {p['type']:<12} Target: {str(target):<25} Reason: {reason:<20} Metric: {p['metric_value']:.4f} (Thresh: {p['threshold']:.4f})")
        else:
            print("  No modifications proposed based on current thresholds.")
        print("--------------------------------------------\n")


    # --------------------------------------------------
    # ----- Placeholders for Structure Modification ----
    # --------------------------------------------------

    def select_best_modification(self, proposals, current_params, current_posterior, current_elbo):
        """ Selects the best modification by evaluating candidates (Phase 2+). """
        print("WARNING: select_best_modification - Structure modification evaluation not implemented yet.")
        return None # No modification selected

    def apply_modification(self, graph, params, proposal):
        """ Applies a modification and returns new graph + initial params (Phase 2+). """
        print(f"WARNING: apply_modification - Applying {proposal['type']} not implemented yet.")
        # Needs to perform graph surgery (add/remove nodes/edges in G_traj)
        # Needs to update node_to_index, edge_list, etc. in the graph object
        # Needs to intelligently initialize parameters (g, K, sigma2, pi) for the new structure,
        # potentially inheriting from the old structure where possible.
        # Returns a *new* TrajectoryGraph object and potentially *new* initial params dictionary.
        # For now, return unchanged.
        import copy
        # Return deep copies to avoid modifying the original in place during evaluation
        return copy.deepcopy(graph), {k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k, v in params.items()}