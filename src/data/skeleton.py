import torch
import numpy as np

def get_mouse_skeleton_adj(num_nodes=7, strategy='normalized'):
    """
    Returns the adjacency matrix for the 7-node unified mouse skeleton.
    Defined in src/data/transforms.py BodyPartMapping:
    ['nose', 'ear_left', 'ear_right', 'neck', 'side_left', 'side_right', 'tail_base']
    
    Indices:
    0: nose
    1: ear_left
    2: ear_right
    3: neck
    4: side_left
    5: side_right
    6: tail_base
    """
    if num_nodes != 7:
        # Fallback for identity if node count doesn't match standard skeleton
        return torch.eye(num_nodes)

    # Define physical connections (undirected edges)
    edges = [
        # Head Cluster
        (0, 1), (0, 2), # Nose connects to Ears
        (0, 3),         # Nose connects to Neck
        (1, 3), (2, 3), # Ears connect to Neck
        
        # Body/Spine
        (3, 4), (3, 5), # Neck connects to Sides (features often include hips/lateral)
        (3, 6),         # Neck connects to Tail Base (Spine)
        
        # Posterior
        (4, 6), (5, 6), # Sides connect to Tail Base
    ]

    # Initialize adjacency matrix
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    # Fill edges (undirected)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    # Add self-loops
    np.fill_diagonal(A, 1)

    # Normalization strategies
    if strategy == 'uniform':
        # Just 0s and 1s
        pass
    elif strategy == 'normalized':
        # D^-0.5 * A * D^-0.5
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
        D_mat_inv_sqrt = np.diag(D_inv_sqrt)
        A = np.dot(np.dot(D_mat_inv_sqrt, A), D_mat_inv_sqrt)
    
    return torch.FloatTensor(A)
