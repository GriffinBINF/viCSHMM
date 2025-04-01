import torch
import numpy as np
from torch.distributions import Categorical, Beta
from collections import defaultdict
import networkx as nx


class TreeVariationalPosterior:
    def __init__(self, traj_graph, n_cells: int, device='cpu'):
        self.traj = traj_graph
        self.device = device
        self.n_cells = n_cells

        self.edge_list = traj_graph.edge_list  # (u_idx, v_idx)
        self.n_edges = len(self.edge_list)

        # Variational parameters per cell per edge
        self.alpha = torch.nn.Parameter(torch.ones(n_cells, self.n_edges, device=device))
        self.beta = torch.nn.Parameter(torch.ones(n_cells, self.n_edges, device=device))
        self.edge_logits = torch.nn.Parameter(torch.zeros(n_cells, self.n_edges, device=device))

        # Learnable branch logits per edge (for B)
        self.branch_logits = torch.nn.Parameter(torch.zeros(self.n_edges, device=device))

        # Edge ↔ index mapping
        self.edge_to_index = {(u, v): i for i, (u, v) in enumerate(self.edge_list)}
        self.index_to_edge = {i: (u, v) for (u, v), i in self.edge_to_index.items()}

        # Parent → [children] edge mapping (used in transitions)
        self.children_map = defaultdict(list)
        for (u, v) in self.edge_list:
            self.children_map[u].append((u, v))

    def compute_branch_probs(self):
        """Compute softmax-normalized branch probabilities B(edge) over children of each node."""
        B = {}
        for parent, child_edges in self.children_map.items():
            logits = torch.stack([self.branch_logits[self.edge_to_index[e]] for e in child_edges])
            probs = torch.softmax(logits, dim=0)
            for edge, prob in zip(child_edges, probs):
                B[edge] = prob
        return B

    def compute_transition_probs(self):
        """
        Compute transition probabilities A(sp,t → sq,t′) using learned branch logits.
        Returns:
            A_probs: dict mapping (from_edge_idx, to_edge_idx) → A probability
            Z: dict mapping from_edge_idx → normalization constant
        """
        B = self.compute_branch_probs()
        A_probs = {}
        Z = {}
    
        G = self.traj.G_traj
        index_to_name = {v: k for k, v in self.traj.node_to_index.items()}
        
        for i, (u_idx, v_idx) in enumerate(self.edge_list):
            u = index_to_name[u_idx]
            v = index_to_name[v_idx]
    
            reachable = []
            branch_path_weights = {}
    
            for j, (x_idx, y_idx) in enumerate(self.edge_list):
                x = index_to_name[x_idx]
                y = index_to_name[y_idx]
    
                # Check if (x, y) is reachable from v
                try:
                    path_nodes = nx.shortest_path(G, source=v, target=x)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
    
                # Convert to edge path
                edge_path = [(path_nodes[k], path_nodes[k + 1]) for k in range(len(path_nodes) - 1)]
    
                # Compute product of B's along the path
                product = 1.0
                for edge in edge_path:
                    product *= B.get(edge, 1e-8)
    
                branch_path_weights[j] = product
                reachable.append(j)
    
            # Z_p,t normalization constant: on-edge stay + all transitions
            z_val = (1 - 0.5) + sum(branch_path_weights.values())  # assume midpoint t = 0.5
            Z[i] = z_val
    
            # Stay on same edge
            A_probs[(i, i)] = 1.0 / z_val
    
            # Cross-edge transitions
            for j in reachable:
                if j == i:
                    continue
                A_probs[(i, j)] = branch_path_weights[j] / z_val
    
        return A_probs, Z


    def sample(self, cell_idx, n_samples=1):
        """Sample edge and t ∈ [0,1] from q(S) for a given cell."""
        edge_probs = torch.softmax(self.edge_logits[cell_idx], dim=0)
        edge_dist = Categorical(edge_probs)
        edge_idx = edge_dist.sample((n_samples,))  # shape: [n_samples]
        alpha = self.alpha[cell_idx, edge_idx]
        beta = self.beta[cell_idx, edge_idx]
        t = Beta(alpha, beta).rsample()
        return edge_idx, t

    def log_q(self, cell_idx, edge_idx, t):
        """Log q(S = (edge_idx, t))"""
        edge_probs = torch.softmax(self.edge_logits[cell_idx], dim=0)
        log_edge_prob = torch.log(edge_probs[edge_idx] + 1e-10)
        alpha = self.alpha[cell_idx, edge_idx]
        beta = self.beta[cell_idx, edge_idx]
        log_t_prob = Beta(alpha, beta).log_prob(t)
        return log_edge_prob + log_t_prob

    def compute_edge_probs(self):
        """Return q(edge) for all cells."""
        return torch.softmax(self.edge_logits, dim=1)
