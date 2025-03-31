import torch
import numpy as np

class BeliefPropagator:
    def __init__(self, traj_graph, posterior):
        self.traj = traj_graph
        self.posterior = posterior
        self.device = posterior.device

        self.path_ids = posterior.path_ids  # List of (root, leaf) tuples
        self.n_paths = len(self.path_ids)

        self.path_to_index = {pid: i for i, pid in enumerate(self.path_ids)}
        self.index_to_path = {i: pid for i, pid in enumerate(self.path_ids)}

        self.A = self._build_path_adjacency()

    def _build_path_adjacency(self):
        """
        Builds symmetric adjacency matrix A where A[i,j] = 1 if
        path_i and path_j share a node in their trajectory.
        """
        A = torch.zeros((self.n_paths, self.n_paths), device=self.device)

        for i, path_i in enumerate(self.path_ids):
            nodes_i = {u for u, _ in self.posterior.paths[path_i]}
            nodes_i.add(self.posterior.paths[path_i][-1][1])

            for j, path_j in enumerate(self.path_ids):
                if i == j:
                    continue
                nodes_j = {u for u, _ in self.posterior.paths[path_j]}
                nodes_j.add(self.posterior.paths[path_j][-1][1])

                if nodes_i & nodes_j:
                    A[i, j] = 1.0

        A = A / (A.sum(dim=1, keepdim=True) + 1e-8)  # Row-normalize
        return A

    def diffuse(self, q_p, alpha=0.5, steps=1):
        """
        Diffuse q_p: [N, P] over path topology.

        Args:
            q_p: tensor of shape [N, P], base categorical distribution
            alpha: retention strength
            steps: number of diffusion steps

        Returns:
            q_eff: diffused posterior [N, P]
        """
        q_eff = q_p.clone()
        for _ in range(steps):
            q_eff = alpha * q_p + (1 - alpha) * (q_eff @ self.A)
        return q_eff
