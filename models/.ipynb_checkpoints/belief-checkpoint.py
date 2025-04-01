import torch


class BeliefPropagator:
    def __init__(self, traj_graph, posterior):
        self.traj = traj_graph
        self.posterior = posterior
        self.device = posterior.device

        self.edge_list = traj_graph.edge_list  # [(u_idx, v_idx)]
        self.n_edges = len(self.edge_list)
        self.edge_to_index = {
            (u, v): i for i, (u, v) in enumerate(traj_graph.edge_list)
        }
        self.index_to_edge = {i: (u, v) for (u, v), i in self.edge_to_index.items()}

        self.A = self._build_edge_adjacency()

    def _build_edge_adjacency(self):
        """
        Builds edge adjacency matrix [E, E] where A[i, j] reflects the probability
        of transitioning from edge i to edge j, based on shared nodes and learned transition weights.
        """
        E = self.n_edges
        A = torch.zeros((E, E), device=self.device)

        for i in range(E):
            u_i, v_i = self.index_to_edge[i]

            # Always allow self-loop (retain mass)
            A[i, i] = 1.0

            # Check all other edges j
            for j in range(E):
                if i == j:
                    continue
                u_j, v_j = self.index_to_edge[j]

                # Edges are connectable if they share a node (v_i == u_j or v_i == v_j)
                shared = (v_i == u_j or v_i == v_j or u_i == v_j or u_i == u_j)
                if not shared:
                    continue

                # Look up transition probability A((v_i, v_j)) = B / Z
                # Only valid if there's an actual trajectory edge between v_i → v_j
                if (v_i, v_j) in self.traj.branch_probabilities:
                    B_val = self.traj.branch_probabilities[(v_i, v_j)]
                    Z_val = self.traj.normalizing_constants.get(v_i, 1.0)
                    A_prob = B_val / (Z_val + 1e-8)
                    A[i, j] = A_prob

        # Normalize rows so they sum to 1
        A = A / (A.sum(dim=1, keepdim=True) + 1e-8)
        return A

    def diffuse(self, q_edge, alpha=0.5, steps=1):
        """
        Diffuse edge-level probabilities q_edge: [N, E] using adjacency A.
        alpha: retention factor.
        Returns q_eff: [N, E]
        """
        q_eff = q_edge.clone()
        for _ in range(steps):
            q_eff = alpha * q_edge + (1 - alpha) * (q_eff @ self.A)
        return q_eff
