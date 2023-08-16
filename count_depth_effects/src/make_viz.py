import numpy as np
import matplotlib.pyplot as plt
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


def plot_3D_matrix(matrix_3D, colorization):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract X, Y, Z coordinates
    X, Y, Z = matrix_3D[:, 0].numpy(
    ), matrix_3D[:, 1].numpy(), matrix_3D[:, 2].numpy()

    # Check if colorization is categorical or continuous
    if all(isinstance(item, str) for item in colorization):
        # Categorical labels
        unique_labels = list(set(colorization))
        color_map = plt.get_cmap('tab10', len(unique_labels))
        colors = [color_map(unique_labels.index(label))
                  for label in colorization]
    else:
        # Continuous values (normalized to 0-1)
        color_map = plt.get_cmap('inferno')
        normalized_values = (colorization - np.min(colorization)) / \
            (np.max(colorization) - np.min(colorization))
        colors = [color_map(value) for value in normalized_values]

    # Plot the points
    ax.scatter(X, Y, Z, c=colors, cmap='inferno')

    # Add labels and title if needed
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Visualization')

    # Show the plot
    plt.show()
