import torch
import numpy as np
import networkx as nx


class TreeVariationalPosterior:
    def __init__(self, traj_graph, n_cells: int, device='cpu'):
        self.traj = traj_graph
        self.device = device
        self.n_cells = n_cells

        # Enumerate all root-to-leaf paths as sequences of edges
        self.paths = self._enumerate_paths()
        self.path_ids = list(self.paths.keys())  # (root, leaf) tuples
        self.n_paths = len(self.path_ids)

        # Variational Beta parameters per cell per path
        self.alpha = torch.nn.Parameter(torch.ones(n_cells, self.n_paths, device=device))
        self.beta = torch.nn.Parameter(torch.ones(n_cells, self.n_paths, device=device))

        # Map from path ID to index
        self.path_to_index = {path_id: i for i, path_id in enumerate(self.path_ids)}
        self.index_to_path = {i: path_id for i, path_id in enumerate(self.path_ids)}

        # Learnable logits for categorical path assignment
        self.path_logits = torch.nn.Parameter(torch.zeros(n_cells, self.n_paths, device=device))

    def _enumerate_paths(self):
        """Enumerate all root-to-leaf paths in the trajectory graph."""
        paths = {}
        root_nodes = [n for n in self.traj.G_traj.nodes if self.traj.G_traj.in_degree(n) == 0]
        leaf_nodes = [n for n in self.traj.G_traj.nodes if self.traj.G_traj.out_degree(n) == 0]

        for root in root_nodes:
            for leaf in leaf_nodes:
                try:
                    path = nx.shortest_path(self.traj.G_traj, source=root, target=leaf)
                    edge_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                    paths[(root, leaf)] = edge_path
                except nx.NetworkXNoPath:
                    continue
        return paths

    def compute_path_probs(self):
        """Compute softmax-normalized probabilities for each path based on branch probabilities."""
        log_probs = torch.zeros(self.n_paths, device=self.device)
        for i, path_id in enumerate(self.path_ids):
            path = self.paths[path_id]
            logp = 0.0
            for (u, v) in path:
                logp += np.log(self.traj.branch_probabilities.get((u, v), 1e-10))
            log_probs[i] = logp
        probs = torch.softmax(log_probs, dim=0)
        return probs  # shape: [n_paths]

    def sample(self, cell_idx, n_samples=1):
        """Sample (path_idx, t) from the posterior for a given cell."""
        path_probs = self.compute_path_probs()  # [n_paths]
        path_idx = torch.multinomial(path_probs, num_samples=n_samples, replacement=True).squeeze()
        alpha = self.alpha[cell_idx, path_idx]
        beta = self.beta[cell_idx, path_idx]
        t = torch.distributions.Beta(alpha, beta).rsample()
        return path_idx, t  # shape: [n_samples], [n_samples]
