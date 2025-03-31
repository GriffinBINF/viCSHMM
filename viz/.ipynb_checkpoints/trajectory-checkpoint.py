import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx

from utils import cubic_bezier


def plot_cells_on_trajectory(
    G_traj, assignments, adata, color_key='leiden',
    curve_amount=0.8, node_size=500, cell_size=30,
    horizontal_thresh=0.01, edge_width=5, edge_color='lightgrey',
    title="Cells on Stochastic PAGA Trajectory", branch_probs=None,
    plot_transitions=False
):
    """
    Visualizes cells overlaid on a trajectory graph defined by PAGA edges.

    Args:
        G_traj (nx.DiGraph): Directed trajectory graph with node/edge attributes.
        assignments (pd.DataFrame): DataFrame with 'edge' and 'latent_time' columns per cell.
        adata (AnnData): AnnData object containing expression and obs/var metadata.
        color_key (str): obs or gene name to color cells by.
        curve_amount (float): How much to curve edges (0 = straight).
        node_size (int): Size of graph nodes.
        cell_size (int): Size of individual cells.
        edge_width (int): Width of edge lines.
        edge_color (str): Edge color.
        title (str): Plot title.
        branch_probs (dict): Optional, used to annotate branch probabilities.
        plot_transitions (bool): Whether to overlay transition matrix A on split edges.
    """
    pos = nx.nx_agraph.graphviz_layout(G_traj, prog="dot", args="-Grankdir=LR")
    plt.figure(figsize=(12, 7))

    def get_node_color(node):
        ntype = G_traj.nodes[node].get('type', '')
        mapping = {'root': 'red', 'split': 'orange', 'leaf': 'lightgreen'}
        for key in mapping:
            if key in ntype:
                return mapping[key]
        return 'lightblue'

    def format_node_label(node):
        ntype = G_traj.nodes[node].get('type', '')
        cluster = node.split('_')[-1]
        prefix = {'root': 'R', 'split': 'S', 'leaf': 'L'}.get(ntype.split('_')[-1], 'I')
        return f"{prefix}{cluster}"

    # Plot graph nodes
    for node, (x, y) in pos.items():
        plt.scatter(x, y, color=get_node_color(node), s=node_size, zorder=3)
        plt.text(x, y, format_node_label(node), fontsize=10, ha="center", va="center", zorder=4)

    # Draw edges with optional Bézier curves
    edge_labels_info = []
    for src, dst in G_traj.edges():
        P0, P1 = pos[src], pos[dst]
        vertical_diff = abs(P1[1] - P0[1])
        is_horizontal = vertical_diff < horizontal_thresh * abs(P1[0] - P0[0])

        if is_horizontal:
            plt.plot([P0[0], P1[0]], [P0[1], P1[1]], color=edge_color, lw=edge_width, alpha=0.8, zorder=2)
            mid_pos = ((P0[0] + P1[0]) / 2, (P0[1] + P1[1]) / 2)
        else:
            cp1 = (P0[0] + (P1[0]-P0[0])/3.0, P0[1] + curve_amount*(P1[1]-P0[1]))
            cp2 = (P0[0] + 2*(P1[0]-P0[0])/3.0, P1[1] - curve_amount*(P1[1]-P0[1]))
            bez_points = cubic_bezier(P0, cp1, cp2, P1, np.linspace(0, 1, 100))
            plt.plot(bez_points[:, 0], bez_points[:, 1], color=edge_color, lw=edge_width, alpha=0.8, zorder=2)
            mid_pos = cubic_bezier(P0, cp1, cp2, P1, np.array([0.5]))[0]

        edge_label = G_traj.edges[src, dst].get('label', '')
        if branch_probs:
            bp = branch_probs.get((src, dst), None)
            if bp is not None:
                edge_label += f"\nB: {bp:.2f}"
        if plot_transitions and 'split' in G_traj.nodes[src].get('type', ''):
            A_val = G_traj.edges[src, dst].get('A', None)
            if A_val is not None:
                edge_label += f"\nA: {A_val:.2f}"
        edge_labels_info.append((mid_pos, edge_label))

    # Determine cell coloring
    if color_key in adata.obs.columns:
        cell_color_series = adata.obs[color_key]
    elif color_key in adata.var_names:
        gene_index = adata.var_names.get_loc(color_key)
        values = adata.X[:, gene_index].toarray().ravel() if hasattr(adata.X, "toarray") else np.array(adata.X[:, gene_index]).ravel()
        cell_color_series = pd.Series(values, index=adata.obs_names)
    else:
        raise ValueError(f"color_key '{color_key}' not found in adata.obs or adata.var_names")

    if pd.api.types.is_numeric_dtype(cell_color_series):
        norm = mcolors.Normalize(vmin=cell_color_series.min(), vmax=cell_color_series.max())
        cmap = cm.viridis
        get_color = lambda val: cmap(norm(val))
        add_colorbar = True
    else:
        categories = cell_color_series.unique()
        colors_discrete = cm.tab10.colors if len(categories) <= 10 else cm.tab20.colors
        cat2color = {cat: colors_discrete[i % len(colors_discrete)] for i, cat in enumerate(sorted(categories))}
        get_color = lambda val: cat2color[val]
        add_colorbar = False

    # Overlay cells on edges
    cell_x, cell_y, cell_colors = [], [], []
    for cell_id, row in assignments.iterrows():
        edge = row['edge']
        latent_time = row['latent_time']
        if edge:
            P0, P1 = pos[edge[0]], pos[edge[1]]
            cp1 = (P0[0] + (P1[0] - P0[0]) / 3.0, P0[1] + curve_amount * (P1[1] - P0[1]))
            cp2 = (P0[0] + 2 * (P1[0] - P0[0]) / 3.0, P1[1] - curve_amount * (P1[1] - P0[1]))
            pt = cubic_bezier(P0, cp1, cp2, P1, np.array([latent_time]))[0]
            cell_x.append(pt[0])
            cell_y.append(pt[1])
        else:
            cell_x.append(0)
            cell_y.append(0)
        val = cell_color_series.loc[cell_id]
        val = val.iloc[0] if isinstance(val, pd.Series) else val
        cell_colors.append(get_color(val))

    plt.scatter(cell_x, cell_y, color=cell_colors, s=cell_size, edgecolor='black', linewidth=0.5, zorder=5)

    if add_colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label=color_key)
    else:
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=cat,
                              markerfacecolor=cat2color[cat], markersize=8)
                   for cat in sorted(cat2color)]
        plt.legend(handles=handles, title=color_key, bbox_to_anchor=(1.05, 1), loc='upper left')

    for mid_pos, edge_label in edge_labels_info:
        circle = plt.Circle(mid_pos, radius=node_size * 0.015, edgecolor=edge_color,
                            facecolor='white', zorder=7)
        plt.gca().add_patch(circle)
        plt.text(mid_pos[0], mid_pos[1], edge_label, fontsize=9, color="black", ha="center", va="center", zorder=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
