import torch
import torch.nn as nn
from torch.distributions import Categorical, Beta
from collections import defaultdict
import networkx as nx
from utils.constants import EPSILON
# Remove the import related to initialize_edge_logits... - it's done externally now

class TreeVariationalPosterior(nn.Module):
    def __init__(self, traj_graph, n_cells: int, device='cpu'):
        super().__init__()
        self.traj = traj_graph # Store the TrajectoryGraph object
        self.device = device
        self.n_cells = n_cells

        # Use edge_list from the passed traj_graph object
        self.edge_list = self.traj.edge_list # Should be [(u_idx, v_idx)]
        self.n_edges = len(self.edge_list)

        # Initialize parameters
        self.alpha = nn.Parameter(torch.ones(n_cells, self.n_edges, device=device))
        self.beta = nn.Parameter(torch.ones(n_cells, self.n_edges, device=device))

        # Initialize edge_logits to small random values or zeros.
        # The actual meaningful initialization happens *externally* using
        # initialize_edge_logits_from_assignment AFTER the posterior object is created.
        # Using small random values helps break symmetry if no assignment is provided.
        self.edge_logits = nn.Parameter(1e-2 * torch.randn(n_cells, self.n_edges, device=device))

        self.branch_logits = nn.Parameter(torch.zeros(self.n_edges, device=device)) # For transition/branching logic

        # Mappings using the stored traj_graph and its properties
        # Ensure edge_list uses indices
        if not self.edge_list or not isinstance(self.edge_list[0][0], int):
             raise ValueError("traj_graph.edge_list must contain index-based tuples (u_idx, v_idx)")

        self.edge_to_index = {(u, v): i for i, (u, v) in enumerate(self.edge_list)}
        self.index_to_edge = {i: (u, v) for (u, v), i in self.edge_to_index.items()}

        # node_to_index should be on the traj_graph object
        if not hasattr(self.traj, 'node_to_index'):
             raise AttributeError("traj_graph object must have a 'node_to_index' attribute.")
        self.index_to_name = {v: k for k, v in self.traj.node_to_index.items()}

        # Children map based on index-based edge_list
        self.children_map = defaultdict(list)
        for u_idx, v_idx in self.edge_list:
            self.children_map[u_idx].append((u_idx, v_idx)) # Use indices

        # Reachable paths using the actual graph G_traj from the traj_graph object
        if not hasattr(self.traj, 'G_traj'):
             raise AttributeError("traj_graph object must have a 'G_traj' attribute (NetworkX graph).")
        self.reachable_paths = self._precompute_reachable_paths(self.traj.G_traj)

    def _precompute_reachable_paths(self, G_traj):
        # This method seems okay, relies on internal mappings correctly
        edge_to_index = self.edge_to_index
        index_to_edge = self.index_to_edge
        index_to_name = self.index_to_name # Map indices back to names for nx path finding

        reachable_paths = {} # Stores paths between *edge indices*

        for i, (u_idx, v_idx) in index_to_edge.items(): # Starting edge i: u -> v
            u_name = index_to_name.get(u_idx)
            v_name = index_to_name.get(v_idx)
            if u_name is None or v_name is None: continue # Skip if node mapping fails

            reachable_paths[i] = {}

            for j, (x_idx, w_idx) in index_to_edge.items(): # Potential target edge j: x -> w
                x_name = index_to_name.get(x_idx)
                if x_name is None: continue

                # Find path from the *end* of edge i (v_name) to the *start* of edge j (x_name)
                try:
                    # Use the actual NetworkX graph from the trajectory object
                    path_nodes_names = nx.shortest_path(G_traj, source=v_name, target=x_name)

                    # Convert node names in path back to edge *indices*
                    edge_path_names = [(path_nodes_names[k], path_nodes_names[k + 1]) for k in range(len(path_nodes_names) - 1)]

                    # Map name-based edges to index-based edges and then to edge indices
                    path_indices = []
                    valid_path = True
                    for u_n, v_n in edge_path_names:
                        u_i = self.traj.node_to_index.get(u_n)
                        v_i = self.traj.node_to_index.get(v_n)
                        if u_i is None or v_i is None:
                            valid_path = False; break
                        edge_idx = edge_to_index.get((u_i, v_i))
                        if edge_idx is None:
                            valid_path = False; break
                        path_indices.append(edge_idx)

                    if valid_path and path_indices: # Ensure the path exists in our edge list
                        reachable_paths[i][j] = path_indices
                    elif not edge_path_names and v_name == x_name: # Handle case where v == x (direct connection implied)
                         # No intermediate path needed, the transition depends only on branching from v_name/v_idx
                         # The logic in compute_transition_probs handles B_tensor, which covers this.
                         # We don't need an explicit empty path here. Let compute_transition_probs figure it out.
                         pass


                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue # No path exists in the underlying graph

        return reachable_paths


    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def compute_branch_probs(self):
        """Computes branching probabilities B_e for each edge e based on branch_logits."""
        # B_tensor[edge_idx] = P(choose edge e | parent is common source node)
        B_tensor = torch.zeros_like(self.branch_logits) # Size [E]

        for parent_idx, child_edges_tuples in self.children_map.items(): # key=parent_idx, val=list of (parent_idx, child_idx) tuples
            if not child_edges_tuples:
                continue

            # Get indices of edges originating from this parent_idx
            child_edge_indices = torch.tensor(
                [self.edge_to_index[edge_tuple] for edge_tuple in child_edges_tuples],
                dtype=torch.long, # Ensure long type for indexing
                device=self.branch_logits.device
            )

            if len(child_edge_indices) > 0:
                # Get the logits corresponding to these children edges
                logits_at_branch = self.branch_logits[child_edge_indices]
                # Compute softmax over these children
                probs_at_branch = torch.softmax(logits_at_branch, dim=0)
                # Assign the computed probabilities back to the B_tensor
                B_tensor[child_edge_indices] = probs_at_branch
            # If only one child, softmax([]) results in value 1, which is correct implicitly if B_tensor is initialized smartly, but explicit handling is safer.
            # If len=1, softmax([logit]) = [1.0]. This seems correct.

        # Edges originating from root nodes might need special handling if they aren't in children_map naturally
        # Assuming graph structure ensures all edges have a parent node index captured by children_map keys

        return B_tensor # Shape [E]

    def compute_transition_probs(self):
        """Computes the edge-to-edge transition probability matrix A[i, j] = P(next is edge j | current is edge i). Revised to avoid inplace ops."""
        B_tensor = self.compute_branch_probs() # Shape [E], requires grad via branch_logits
        n_edges = self.n_edges
        device = self.device

        row_indices = []
        col_indices = []
        values = []
        Z_vals = [] # Store normalizing constants

        # Determine the appropriate dtype from B_tensor or default to float32
        compute_dtype = B_tensor.dtype if B_tensor is not None else torch.float32

        for i in range(n_edges): # For each source edge i
            u_i, v_i = self.index_to_edge[i]

            stay_weight = torch.tensor(0.5, device=device, dtype=compute_dtype)
            current_row_targets = [i] # Target index
            current_row_weights = [stay_weight] # Weight value (tensor)

            # Transitions based on branching at node v_i
            outgoing_edges_from_v = self.children_map.get(v_i, [])
            outgoing_edge_indices = [self.edge_to_index[edge_tuple] for edge_tuple in outgoing_edges_from_v]

            if outgoing_edge_indices:
                # Select the relevant branch probabilities (maintains grad)
                branch_probs_at_v = B_tensor[outgoing_edge_indices] # Shape [num_outgoing]

                for k_idx, prob_k in zip(outgoing_edge_indices, branch_probs_at_v):
                    current_row_targets.append(k_idx)
                    current_row_weights.append(prob_k) # Add tensor potentially requiring grad

            # Stack weights into a tensor to allow sum() to work correctly with autograd
            if current_row_weights:
                # Stack the list of tensors directly. Ensure they have requires_grad info preserved.
                weights_tensor = torch.stack(current_row_weights)
                total_weight = weights_tensor.sum() # Summation is differentiable
            else:
                # Create a zero tensor with the correct dtype and device if no weights
                total_weight = torch.tensor(0.0, device=device, dtype=compute_dtype)

            # Ensure Z_i preserves dtype and grad if total_weight requires grad
            # Use total_weight.clamp(min=EPSILON) for normalization stability
            Z_i = total_weight.clamp(min=EPSILON)
            Z_vals.append(Z_i) # Store Z_i (tensor)

            # Normalize the weights for this row and add to lists for final assembly
            if current_row_weights: # Check if weights_tensor was created and Z_i is non-zero
                 normalized_weights = weights_tensor / Z_i # Division is differentiable
                 for target_j, norm_w in zip(current_row_targets, normalized_weights):
                     row_indices.append(i)
                     col_indices.append(target_j)
                     values.append(norm_w) # Append the normalized tensor value

        # Construct the dense matrix A_probs *after* collecting all values
        A_probs = torch.zeros(n_edges, n_edges, device=device, dtype=compute_dtype) # Initialize with correct dtype

        if values:
            row_indices_t = torch.tensor(row_indices, device=device, dtype=torch.long)
            col_indices_t = torch.tensor(col_indices, device=device, dtype=torch.long)
            # Stack the list of tensor values into a single tensor for index_put_
            values_t = torch.stack(values)

            # Use index_put_ : This is inplace on A_probs, but A_probs was just created
            # and doesn't have a grad history itself. The grad history comes from values_t.
            A_probs.index_put_((row_indices_t, col_indices_t), values_t, accumulate=False) # Use accumulate=False to overwrite zeros

        # Stack the list of Z tensors. Ensure requires_grad is maintained if any Z_i required grad.
        Z = torch.stack(Z_vals)

        return A_probs, Z

    # sample, log_q, compute_edge_probs seem okay.
    def sample(self, cell_idx, n_samples=1):
        """Sample edge index and latent time for a given cell."""
        # Ensure cell_idx is valid and refers to a row in the parameters
        if isinstance(cell_idx, torch.Tensor):
             cell_idx = cell_idx.item() # Get scalar index if tensor
        if not (0 <= cell_idx < self.n_cells):
             raise IndexError(f"cell_idx {cell_idx} out of bounds for n_cells {self.n_cells}")

        with torch.no_grad(): # Sampling should not require gradients
            # Use logits directly for stability with Categorical
            edge_dist = Categorical(logits=self.edge_logits[cell_idx])
            edge_idx = edge_dist.sample((n_samples,)) # Shape [n_samples]

            # Ensure alpha and beta are positive for Beta distribution
            alpha_clamped = self.alpha[cell_idx, edge_idx].clamp(min=EPSILON) # Shape [n_samples]
            beta_clamped = self.beta[cell_idx, edge_idx].clamp(min=EPSILON)  # Shape [n_samples]

            # Use reparameterization trick (rsample) if gradients needed through samples
            # But sampling is often for evaluation, so .sample() might be sufficient.
            # Let's stick to rsample as ELBO uses it.
            t = Beta(alpha_clamped, beta_clamped).rsample() # Shape [n_samples]

            # Squeeze if n_samples is 1 for consistency? Or always return list/tensor?
            # Current usage seems to expect tensors, so keep shape [n_samples].
            if n_samples == 1:
                 edge_idx = edge_idx.squeeze(0)
                 t = t.squeeze(0)

        return edge_idx, t

    def log_q(self, cell_idx, edge_idx, t):
        """Compute log q(edge_idx, t | cell_idx)"""
        # Ensure inputs are correctly shaped/indexed
        if isinstance(cell_idx, torch.Tensor): cell_idx = cell_idx.item()
        if isinstance(edge_idx, torch.Tensor): edge_idx = edge_idx.long() # Ensure long for indexing
        if isinstance(t, torch.Tensor): t = t.clamp(EPSILON, 1.0 - EPSILON) # Clamp t for log_prob stability

        # Log probability of edge choice
        # Use log_softmax for numerical stability
        log_edge_probs = torch.log_softmax(self.edge_logits[cell_idx], dim=0)
        log_edge_prob = log_edge_probs[edge_idx] # Select the probability for the chosen edge(s)

        # Log probability of time choice
        alpha_clamped = self.alpha[cell_idx, edge_idx].clamp(min=EPSILON)
        beta_clamped = self.beta[cell_idx, edge_idx].clamp(min=EPSILON)
        log_t_prob = Beta(alpha_clamped, beta_clamped).log_prob(t)

        return log_edge_prob + log_t_prob


    def compute_edge_probs(self):
        """Compute q(edge | cell) = softmax(edge_logits)."""
        return torch.softmax(self.edge_logits, dim=1) # Shape [N, E]