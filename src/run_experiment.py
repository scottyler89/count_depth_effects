from .run_sim import *
from .norm_methods import *
from make_viz import optimize_3D_matrix
from sklearn.metrics.pairwise import euclidean_distances as euc
import numpy as np
import torch

def run_experiment(norm_methods):
    # Generate the data matrix as per the original code
    X, Xdepth, depth_vect = generate_dataset()

    # Benchmark different normalization methods
    results = {}
    for norm_name, norm_method in norm_methods.items():
        normalized_X = norm_method(X)  # Assuming X is the original data matrix

        # Compute the input distance matrix
        input_distance_matrix = euc(normalized_X)

        # Optimize the 3D matrix using the gradient descent method
        optimized_3D_matrix = optimize_3D_matrix(depth_vect, input_distance_matrix)

        # Store the results
        results[norm_name] = optimized_3D_matrix

    # Here you can add code to analyze, visualize, or save the results
    return results

if __name__ == "__main__":
    np.random.seed(123456)
    torch.seed
    run_experiment(NORM)

