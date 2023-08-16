from count_depth_effects.src.run_sim import *
from count_depth_effects.src.norm_methods import *
from count_depth_effects.src.make_viz import optimize_3D_matrix, plot_3D_matrix
from sklearn.metrics.pairwise import euclidean_distances as euc
import numpy as np
import torch

def run_experiment(norm_methods):
    # Generate the data matrix as per the original code
    print("Generating the original data")
    X, Xdepth, depth_vect = generate_dataset()

    print("runing the normalizations & projections")
    # Benchmark different normalization methods
    results = {}
    for norm_name, norm_method in norm_methods.items():
        print("\t",norm_name)
        try:
            normalized_X = norm_method(X)  # Assuming X is the original data matrix
        except:
            normalized_X = norm_method(scipy.sparse.csc_matrix(X))

        print("\t\tgetting_distance")
        # Compute the input distance matrix
        input_distance = euc(normalized_X)

        print("\t\tgetting 3D coordinates")
        # Optimize the 3D matrix using the gradient descent method
        optimized_3D_matrix = optimize_3D_matrix(torch.tensor(depth_vect), torch.tensor(input_distance))
        plot_3D_matrix(optimized_3D_matrix, depth_vect)

        # Store the results
        results[norm_name] = optimized_3D_matrix

    # Here you can add code to analyze, visualize, or save the results
    return results

if __name__ == "__main__":
    np.random.seed(123456)
    torch.manual_seed(123456)
    results = run_experiment(NORM)

