import torch
from torch.nn.functional import cosine_similarity


def initialize_3D_matrix(depth_vect, n_cells):
    Z = torch.log(torch.tensor(depth_vect, dtype=torch.float32))
    X = torch.rand(n_cells, requires_grad=True)
    Y = torch.rand(n_cells, requires_grad=True)
    return X, Y, Z


def loss_function(X, Y, Z, input_distance_matrix):
    # Compute 3D distances
    distances_3D = torch.sqrt((X[:, None] - X[None, :]) ** 2 + (Y[:, None] - Y[None, :]) ** 2 + (Z[:, None] - Z[None, :]) ** 2)
    # Compute cosine similarity between 3D distances and input distances
    cosine_sim = cosine_similarity(distances_3D.flatten(), input_distance_matrix.flatten(), dim=0)
    # Return negative cosine similarity as loss
    return -cosine_sim


def optimize_3D_matrix(depth_vect, input_distance_matrix, epochs=1000, lr=0.01):
    n_cells = len(depth_vect)
    X, Y, Z = initialize_3D_matrix(depth_vect, n_cells)
    optimizer = torch.optim.Adam([X, Y], lr=lr)

    for epoch in range(epochs):
        loss = loss_function(X, Y, Z, input_distance_matrix)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Return the optimized 3D matrix
    return torch.stack([X, Y, Z], dim=-1)


