import torch
from models.posterior import TreeVariationalPosterior
from models.belief import BeliefPropagator
from models.emission import pack_emission_params
from utils.inference import initialize_beta_from_cell_assignment
from .init import setup_training
from .loop import run_training_loop

def initialize_training_components(traj_graph, cell_assignment, device=None, sharpness=0.01):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edge_to_index, g_a, g_b, K, sigma2, pi = pack_emission_params(traj_graph, device=device)
    g_a = g_a.requires_grad_()
    g_b = g_b.requires_grad_()
    K = K.requires_grad_()
    log_sigma2 = torch.log(sigma2).requires_grad_()

    X = torch.tensor(traj_graph.adata.X.toarray() if hasattr(traj_graph.adata.X, 'toarray') else traj_graph.adata.X,
                     dtype=torch.float32).to(device)
    
    n_cells = X.shape[0]
    posterior = TreeVariationalPosterior(traj_graph, n_cells=n_cells, device=device)
    posterior.path_logits = torch.nn.Parameter(torch.randn(n_cells, posterior.n_paths, device=device))
    
    initialize_beta_from_cell_assignment(posterior, cell_assignment, sharpness=sharpness)
    belief_propagator = BeliefPropagator(traj_graph, posterior)

    return {
        "X": X,
        "posterior": posterior,
        "belief_propagator": belief_propagator,
        "g_a": g_a,
        "g_b": g_b,
        "K": K,
        "log_sigma2": log_sigma2,
        "pi": pi,
        "edge_to_index": edge_to_index,
    }
