import numpy as np
import pandas as pd
import networkx as nx
from viz.trajectory import plot_cells_on_trajectory


class TrajectoryGraph:
    def __init__(self, adata, cluster_key='leiden', random_state=0, laplace_h=1.0):
        self.adata = adata
        self.random_state = random_state
        self.laplace_h = laplace_h
        self.cluster_key = cluster_key

        self.G_traj = None
        self.levels = {}
        self.parent = {}
        self.split_nodes = []
        self.roots = []
        self.branch_probabilities = {}
        self.transition_probabilities = {}
        self.normalizing_constants = {}
        self.emission_params = {}
        self.node_emission = {}
        self.node_to_index = {}
        self.edge_list = []
        self.edge_segments = []

        self._initialize_graph()
        self._initialize_transition_probabilities()
        self._compute_edge_segments()

    def _initialize_graph(self):
        rng = np.random.default_rng(self.random_state)
        conn = self.adata.uns['paga']['connectivities']
        clusters = sorted(self.adata.obs[self.cluster_key].unique(), key=int)

        G_conn = nx.Graph()
        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters):
                w = conn[i, j]
                if w > 0:
                    G_conn.add_edge(c1, c2, weight=w)

        components = list(nx.connected_components(G_conn))
        parent = {}
        levels = {}
        children = {}

        for comp in components:
            comp = list(comp)
            root_cluster = rng.choice(comp)
            self.roots.append(root_cluster)
            queue = [(root_cluster, 0)]
            parent[root_cluster] = None
            levels[root_cluster] = 0
            children[root_cluster] = []

            while queue:
                cur, lvl = queue.pop(0)
                neigh = [n for n in G_conn.neighbors(cur) if n not in parent]
                if not neigh:
                    continue
                weights = np.array([G_conn[cur][n]['weight'] for n in neigh])
                probs = weights / weights.sum()
                num_children = min(2, len(neigh))
                chosen = rng.choice(neigh, size=num_children, replace=False, p=probs)
                children[cur] = list(chosen)
                for nb in chosen:
                    parent[nb] = cur
                    levels[nb] = lvl + 1
                    queue.append((nb, lvl+1))
                    children[nb] = []

        self.parent = parent
        self.levels = levels
        self.split_nodes = [c for c in children if len(children[c]) > 1]
        G_traj = nx.DiGraph()

        def get_node_type(c):
            is_root = (parent[c] is None)
            num_children = len(children[c])
            node_kind = "split" if num_children > 1 else "leaf" if num_children == 0 else "int"
            return f"root_{node_kind}" if is_root else node_kind

        node_for_cluster = {}
        for c in clusters:
            label = f"{get_node_type(c)}_{c}"
            node_for_cluster[c] = label
            G_traj.add_node(label, type=get_node_type(c))

        comp_map = {cl: i for i, comp in enumerate(components) for cl in comp}
        root_node_counter = {}
        for c in self.roots:
            comp_idx = comp_map[c]
            count = root_node_counter.get(comp_idx, 0)
            root_label = f"RootNode_{comp_idx}_{count}"
            root_node_counter[comp_idx] = count + 1
            G_traj.add_node(root_label, type="root_node")
            G_traj.add_edge(root_label, node_for_cluster[c], label=str(c))

        for c in clusters:
            p = parent.get(c, None)
            if p is not None:
                G_traj.add_edge(node_for_cluster[p], node_for_cluster[c], label=str(c))

        self.G_traj = G_traj
        self.branch_probabilities = {(u, v): 1.0 for u, v in G_traj.edges()}

        max_level = max(levels.values()) if levels else 1
        for node in G_traj.nodes():
            cluster = node.split('_')[-1] if G_traj.nodes[node]['type'] != "root_node" else None
            lvl = levels.get(cluster, 0) if cluster else 0
            G_traj.nodes[node]['t'] = lvl / max_level if max_level > 0 else 0.0

        self.node_to_index = {node: i for i, node in enumerate(G_traj.nodes())}
        self.edge_list = [(self.node_to_index[u], self.node_to_index[v]) for u, v in G_traj.edges()]

    def _initialize_transition_probabilities(self):
        for u in self.G_traj.nodes():
            t_u = self.G_traj.nodes[u].get('t', 0.0)
            children = list(self.G_traj.successors(u))
            if not children:
                continue
            weights = {v: 1.0 for v in children}
            Z_u = (1 - t_u) + sum(weights.values())
            self.normalizing_constants[u] = Z_u
            for v in children:
                self.transition_probabilities[(u, v)] = weights[v] / Z_u
                self.G_traj.edges[u, v]['A'] = self.transition_probabilities[(u, v)]

    def _compute_edge_segments(self):
        inv_map = {v: k for k, v in self.node_to_index.items()}
        self.edge_segments = []
        for u_idx, v_idx in self.edge_list:
            u_name = inv_map[u_idx]
            v_name = inv_map[v_idx]
            t_u = self.G_traj.nodes[u_name].get('t', 0.0)
            t_v = self.G_traj.nodes[v_name].get('t', 1.0)
            self.edge_segments.append((min(t_u, t_v), max(t_u, t_v)))

    def assign_cells_to_paths(self, random_state=0):
        rng = np.random.default_rng(random_state)
        cluster_to_edge = {data['label']: (src, dst) for src, dst, data in self.G_traj.edges(data=True) if 'label' in data}

        assignments = pd.DataFrame(index=self.adata.obs_names)
        assignments['edge'] = self.adata.obs[self.cluster_key].astype(str).map(cluster_to_edge)
        assignments['latent_time'] = rng.random(size=self.adata.n_obs)
        return assignments

    def initialize_emission_parameters(self, cell_assignment):
        X = np.asarray(self.adata.X)
        cell_to_index = {cell: idx for idx, cell in enumerate(self.adata.obs_names)}
        node_expr = {node: [] for node in self.G_traj.nodes()}
    
        # Aggregate expression for each node
        for edge in self.G_traj.edges():
            df = cell_assignment[cell_assignment['edge'] == edge]
            if df.empty:
                continue
            for cell in df.index:
                expr = X[cell_to_index[cell]]
                node_expr[edge[0]].append(expr)
                node_expr[edge[1]].append(expr)
    
        # Compute mean expression per node
        self.node_emission = {
            node: np.mean(expr_list, axis=0) if expr_list else np.zeros(X.shape[1])
            for node, expr_list in node_expr.items()
        }
    
        # Initialize edge-specific variance (r²), dropout (pi), and K
        emission_params = {}
        for edge in self.G_traj.edges():
            df = cell_assignment[cell_assignment['edge'] == edge]
            if df.empty:
                var_expr = np.ones(X.shape[1])
                pi_val = np.zeros(X.shape[1])
            else:
                expr_edge = np.array([X[cell_to_index[cell]] for cell in df.index])
                var_expr = np.var(expr_edge, axis=0)
                pi_val = np.mean(expr_edge == 0, axis=0)
            K = np.ones_like(var_expr)
            emission_params[edge] = {
                'K': K,
                'r2': var_expr,
                'pi': pi_val
            }
    
        self.emission_params = emission_params
        return emission_params


    def plot_cells_on_trajectory(self, cell_assignment, **kwargs):
        plot_cells_on_trajectory(self.G_traj, cell_assignment, self.adata, branch_probs=self.branch_probabilities, **kwargs)


def initialize_trajectory(adata, random_state=None, debug=False):
    traj_graph = TrajectoryGraph(adata, random_state=random_state or np.random.randint(0, 2**32 - 1))
    cell_assignment = traj_graph.assign_cells_to_paths(random_state=42)
    traj_graph.initialize_emission_parameters(cell_assignment)
    if debug:
        print("Trajectory graph nodes:", list(traj_graph.G_traj.nodes()))
        print("Trajectory graph edges:", list(traj_graph.G_traj.edges()))
        print("\nSample cell assignments:\n", cell_assignment.head())
    return traj_graph, cell_assignment