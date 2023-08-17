import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import cosine_similarity


def min_max_scaling(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    scaled_tensor = (tensor - min_val) / (max_val - min_val)
    return scaled_tensor


def initialize_3D_matrix(depth_vect, n_cells):
    Z = min_max_scaling(torch.log(torch.tensor(depth_vect, dtype=torch.float32)))
    X = torch.rand(n_cells, requires_grad=True)
    Y = torch.rand(n_cells, requires_grad=True)
    return X, Y, Z


def loss_function(X, Y, Z, input_distance_matrix, eps=1e-8):
    # Compute 3D distances
    distances_3D = torch.sqrt((X[:, None] - X[None, :]) ** 2 + (Y[:, None] - Y[None, :]) ** 2 + (Z[:, None] - Z[None, :]) ** 2 + eps)
    # Compute cosine similarity between 3D distances and input distances
    cosine_sim = cosine_similarity(distances_3D.flatten()+eps,
                                   input_distance_matrix.flatten()+eps, dim=0)
    ## Return negative cosine similarity as loss
    return -cosine_sim
    

def optimize_3D_matrix(depth_vect, input_distance_matrix, epochs=1000, lr=0.01, X=None, Y=None):
    n_cells = len(depth_vect)
    #X, Y, Z = initialize_3D_matrix(depth_vect, n_cells)
    Z = min_max_scaling(torch.log(torch.tensor(depth_vect, dtype=torch.float32)))
    if X is None:
        X = torch.rand(n_cells, requires_grad=True)
    else:
        X = torch.tensor(X, requires_grad=True)
    if Y is None:
        Y = torch.rand(n_cells, requires_grad=True)
    else:
        Y = torch.tensor(Y, requires_grad=True)
    
    optimizer = torch.optim.Adam([X, Y], lr=lr)

    for epoch in range(epochs):
        loss = loss_function(X, Y, Z, input_distance_matrix)
        if epoch % 100 ==0:
            print("\t\t\t",loss)
        #print("Gradient of X:", X.grad)
        #print("Gradient of Y:", Y.grad)
        loss.backward()
        #print("Gradient of X:", X.grad)
        #print("Gradient of Y:", Y.grad)
        #torch.nn.utils.clip_grad_value_([X, Y], clip_value=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Return the optimized 3D matrix
    return torch.stack([X, Y, Z], dim=-1), loss.item()


def plot_3D_matrix(matrix_3D, colorization, title="3D projection", out_file=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract X, Y, Z coordinates
    X, Y, Z = matrix_3D[:, 0].detach().numpy(
    ), matrix_3D[:, 1].detach().numpy(
    ), matrix_3D[:, 2].detach().numpy()

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
    scatter = ax.scatter(X, Y, Z, c=colors, cmap='inferno')

    # Add labels and title if needed
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)

    # Function to update the angle of the plot
    def update(frame):
        ax.view_init(elev=10, azim=frame)
        return scatter,

    # Create animation
    anim = FuncAnimation(fig, update, frames=np.arange(
        0, 360, 1), interval=100, blit=True)

    # Save or show the plot
    if out_file:
        anim.save(out_file, writer='imagemagick')
    else:
        plt.show()


