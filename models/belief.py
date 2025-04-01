import torch


class BeliefPropagator:
    def __init__(self, traj_graph, posterior):
        self.traj = traj_graph
        self.posterior = posterior
        self.device = posterior.device

        self.edge_list = traj_graph.edge_list
        self.n_edges = len(self.edge_list)
        self.edge_to_index = {(u, v): i for i, (u, v) in enumerate(self.edge_list)}
        self.index_to_edge = {i: (u, v) for (u, v), i in self.edge_to_index.items()}

        # Use posterior-derived differentiable transitions
        self.A = self._build_edge_adjacency()

    def _build_edge_adjacency(self):
        """
        Builds [E, E] adjacency matrix where A[i, j] reflects the
        probability of transitioning from edge i to edge j.
        Uses posterior-derived branch logits for differentiability.
        """
        A_probs, _ = self.posterior.compute_transition_probs()  # [E, E]
        A = A_probs.clone()

        # Ensure rows sum to 1 (stability clamp)
        A = A / (A.sum(dim=1, keepdim=True) + 1e-8)
        return A

    def diffuse(self, q_edge, alpha=0.5, steps=1):
        """
        Diffuse edge-level probabilities q_edge: [N, E] using adjacency A.
        alpha: retention factor between original and diffused distribution.
        Returns q_eff: [N, E]
        """
        q_eff = q_edge.clone()
        for _ in range(steps):
            q_eff = alpha * q_edge + (1 - alpha) * (q_eff @ self.A)
        return q_eff
