import torch
from models.posterior import TreeVariationalPosterior
from models.belief import BeliefPropagator
from models.emission import pack_emission_params
from utils.inference import initialize_beta_from_cell_assignment
from .loop import train_model


def initialize_training_components(traj_graph, cell_assignment, device=None, sharpness=0.01):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unpack emission params
    (
        edge_tuple_to_index,
        index_to_edge_name,
        name_to_index,
        g_node_init,
        K_init,
        sigma2_init,
        pi_init
    ) = pack_emission_params(traj_graph, device=device)

    # Mark non-learnable tensors
    g_node_init.requires_grad_(False)
    sigma2_init.requires_grad_(False)

    # Extract expression matrix
    X_raw = traj_graph.adata.X
    X = torch.tensor(X_raw.toarray() if hasattr(X_raw, 'toarray') else X_raw, dtype=torch.float32).to(device)

    # Setup variational posterior
    n_cells = X.shape[0]
    posterior = TreeVariationalPosterior(traj_graph, n_cells=n_cells, device=device)

    # Initialize posterior beta values using edge names
    initialize_beta_from_cell_assignment(
        posterior,
        cell_assignment,
        edge_tuple_to_index=edge_tuple_to_index,
        node_to_index=name_to_index,
        sharpness=sharpness
    )

    # Belief propagator setup
    belief_propagator = BeliefPropagator(traj_graph, posterior)

    return {
        "X": X,
        "traj_graph": traj_graph,
        "posterior": posterior,
        "belief_propagator": belief_propagator,
        "g_init": g_node_init,
        "K_init": K_init,
        "sigma2_init": sigma2_init,
        "pi_init": pi_init,
        "edge_tuple_to_index": edge_tuple_to_index,
        "index_to_edge_name": index_to_edge_name,
        "name_to_index": name_to_index,
    }
