from .run_sim import *
from .norm_methods import *
from make_viz import optimize_3D_matrix
from sklearn.metrics.pairwise import euclidean_distances as euc
import numpy as np

def run_experiment(norm_method_names, norm_method_funcs):
    # Generate the data matrix as per the original code
    # TODO...

    # Create a list of normalization methods
    normalization_methods = [depth_norm, tenk_norm]

    # Benchmark different normalization methods
    results = {}
    for norm_method in normalization_methods:
        normalized_X = norm_method(X)  # Assuming X is the original data matrix

        # Compute the input distance matrix
        input_distance_matrix = euc(normalized_X)

        # Optimize the 3D matrix using the gradient descent method
        optimized_3D_matrix = optimize_3D_matrix(depth_vect, input_distance_matrix)

        # Store the results
        results[norm_method.__name__] = optimized_3D_matrix

    # Here you can add code to analyze, visualize, or save the results
    return results

if __name__ == "__main__":
    run_experiment()

