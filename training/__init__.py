import torch
from models.posterior import TreeVariationalPosterior
from models.belief import BeliefPropagator
from models.emission import pack_emission_params
from utils.inference import initialize_beta_from_cell_assignment, initialize_edge_logits_from_assignment # Added logits init
from .loop import train_model


def initialize_training_components(traj_graph, cell_assignment, device=None, sharpness=0.01, logits_high=5.0, logits_low=0.0):
    """
    Initializes components needed for training.

    Args:
        traj_graph: Initialized TrajectoryGraph.
        cell_assignment: DataFrame with initial 'edge' and 'latent_time'.
        device: Target device.
        sharpness: Initial sharpness for Beta distribution from assignment.
        logits_high: Logit value for assigned edge during initialization.
        logits_low: Logit value for unassigned edges during initialization.

    Returns:
        dict: Dictionary containing initialized components ('X', 'traj_graph',
              'posterior', 'belief_propagator', 'g_init', 'K_init',
              'sigma2_init', 'pi_init', mappings).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unpack initial emission params (these are NOT Parameters yet)
    (
        edge_tuple_to_index,
        index_to_edge_name, # This maps index -> (u_name, v_name)
        name_to_index, # This maps node_name -> u_idx
        g_node_init,
        K_init,
        sigma2_init,
        pi_init # Can be None or tensor
    ) = pack_emission_params(traj_graph, device=device)

    # g and sigma2 should not require grad initially as they are updated differently
    g_node_init.requires_grad_(False)
    sigma2_init.requires_grad_(False)
    # K_init and pi_init might require grad later, handle in train_model

    # Extract expression matrix
    X_raw = traj_graph.adata.X
    X = torch.tensor(X_raw.toarray() if hasattr(X_raw, 'toarray') else X_raw, dtype=torch.float32).to(device)

    # Setup variational posterior
    n_cells = X.shape[0]
    posterior = TreeVariationalPosterior(traj_graph, n_cells=n_cells, device=device) # Pass traj_graph itself

    # Initialize posterior Beta distributions and edge logits based on initial assignment
    # Need edge_tuple_to_index which maps (u_idx, v_idx) -> edge_idx
    # Need node_to_index (name_to_index)
    initialize_beta_from_cell_assignment(
        posterior,
        cell_assignment,
        edge_tuple_to_index=edge_tuple_to_index,
        node_to_index=name_to_index,
        sharpness=sharpness
    )

    # Need edge_to_index from posterior which maps (u_idx, v_idx) -> edge_idx
    initialize_edge_logits_from_assignment(
        posterior=posterior,
        cell_assignment=cell_assignment,
        traj_graph=traj_graph, # Pass the graph object
        edge_to_index=posterior.edge_to_index, # Use the mapping within posterior
        high=logits_high,
        low=logits_low
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
        "pi_init": pi_init, # Pass potentially None pi_init
        "edge_tuple_to_index": edge_tuple_to_index,
        # Pass other mappings if needed outside, but train_model primarily uses edge_tuple_to_index
        # "index_to_edge_name": index_to_edge_name,
        # "name_to_index": name_to_index,
    }