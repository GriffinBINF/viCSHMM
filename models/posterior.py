import torch
import torch.nn as nn
from torch.distributions import Categorical, Beta
from collections import defaultdict
import networkx as nx


class TreeVariationalPosterior(nn.Module):
    def __init__(self, traj_graph, n_cells: int, device='cpu'):
        super().__init__()
        self.traj = traj_graph
        self.device = device
        self.n_cells = n_cells

        self.edge_list = traj_graph.edge_list
        self.n_edges = len(self.edge_list)

        self.alpha = nn.Parameter(torch.ones(n_cells, self.n_edges, device=device))
        self.beta = nn.Parameter(torch.ones(n_cells, self.n_edges, device=device))
        self.edge_logits = nn.Parameter(1.0 * torch.randn(n_cells, self.n_edges, device=device))
        self.branch_logits = nn.Parameter(torch.zeros(self.n_edges, device=device))

        self.edge_to_index = {(u, v): i for i, (u, v) in enumerate(self.edge_list)}
        self.index_to_edge = {i: (u, v) for (u, v), i in self.edge_to_index.items()}

        self.children_map = defaultdict(list)
        for (u, v) in self.edge_list:
            self.children_map[u].append((u, v))

        self.reachable_paths = self._precompute_reachable_paths(self.traj.G_traj)

    def _precompute_reachable_paths(self, G_traj):
        edge_to_index = self.edge_to_index
        reachable_paths = {}
    
        for i, (u_name, v_name) in self.index_to_edge.items():
            reachable_paths[i] = {}
    
            for j, (x_name, _) in self.index_to_edge.items():
                try:
                    path_nodes = nx.shortest_path(G_traj, source=v_name, target=x_name)
                    edge_path = [(path_nodes[k], path_nodes[k + 1]) for k in range(len(path_nodes) - 1)]
                    path_indices = [edge_to_index[edge] for edge in edge_path if edge in edge_to_index]
    
                    if len(path_indices) == len(edge_path):
                        reachable_paths[i][j] = path_indices
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
    
        return reachable_paths


    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def compute_branch_probs(self):
        B_tensor = torch.zeros_like(self.branch_logits)
    
        for parent, child_edges in self.children_map.items():
            if not child_edges:
                continue
    
            # Get indices of child edges
            child_idxs = torch.tensor(
                [self.edge_to_index[edge] for edge in child_edges],
                device=self.branch_logits.device
            )
    
            logits = self.branch_logits[child_idxs]
            probs = torch.softmax(logits, dim=0)
    
            # Use in-place update (preserves graph)
            B_tensor[child_idxs] = probs
    
        return B_tensor


    def compute_transition_probs(self):
        B_tensor = self.compute_branch_probs()
        A_probs = torch.zeros(self.n_edges, self.n_edges, device=self.device)
        Z = torch.zeros(self.n_edges, device=self.device)

        active_paths = sum(len(v) for v in self.reachable_paths.values())
        #print(f"Reachable paths count: {active_paths}")


        for i in range(self.n_edges):
            weights = []
            targets = []

            for j, path_indices in self.reachable_paths.get(i, {}).items():
                b_vals = B_tensor[path_indices]
                weights.append(b_vals.prod())
                targets.append(j)

            if weights:
                stacked_weights = torch.stack(weights)
                z_val = 0.5 + stacked_weights.sum()
            else:
                z_val = torch.tensor(0.5, device=self.device)

            Z[i] = z_val
            A_probs[i, i] = 1.0 / z_val

            for j, w in zip(targets, weights):
                if j != i:
                    A_probs[i, j] = w / z_val

        return A_probs, Z

    def sample(self, cell_idx, n_samples=1):
        edge_probs = torch.softmax(self.edge_logits[cell_idx], dim=0)
        edge_dist = Categorical(edge_probs)
        edge_idx = edge_dist.sample((n_samples,))
        alpha = self.alpha[cell_idx, edge_idx]
        beta = self.beta[cell_idx, edge_idx]
        t = Beta(alpha, beta).rsample()
        return edge_idx, t

    def log_q(self, cell_idx, edge_idx, t):
        edge_probs = torch.softmax(self.edge_logits[cell_idx], dim=0)
        log_edge_prob = torch.log(edge_probs[edge_idx] + 1e-10)
        alpha = self.alpha[cell_idx, edge_idx]
        beta = self.beta[cell_idx, edge_idx]
        log_t_prob = Beta(alpha, beta).log_prob(t)
        return log_edge_prob + log_t_prob

    def compute_edge_probs(self):
        return torch.softmax(self.edge_logits, dim=1)
