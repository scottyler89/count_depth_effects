from count_depth_effects.src.run_sim import *
from count_depth_effects.src.norm_methods import *
from count_depth_effects.src.make_viz import optimize_3D_matrix, plot_3D_matrix, standardize
from sklearn.metrics.pairwise import euclidean_distances as euc
from scipy.stats import pearsonr
from torch.nn.functional import cosine_similarity
import pandas as pd
import numpy as np
import torch
import scipy
import os


def run_experiment(norm_methods, iters=5):
    results = {}
    for it in range(iters):
        # Generate the data matrix as per the original code
        print("Generating the original data")
        X, Xdepth, depth_vect = generate_dataset()
        
        ## first get the ground truth distances
        norm_name = "ground_truth"
        ground_truth_ditances = torch.tensor(euc(X))
        optimized_3D_matrix, temp_loss = optimize_3D_matrix(
            torch.tensor(depth_vect), ground_truth_ditances)
        results["ground_truth"]=optimized_3D_matrix
        plot_3D_matrix(optimized_3D_matrix, np.log(depth_vect),
                    title=f"{norm_name}\ncosine dist: {temp_loss}",
                    out_file=os.path.join("assets", "3D", norm_name)+".gif")

        print("runing the normalizations & projections")
        # Benchmark different normalization methods
        for norm_name, norm_method in norm_methods.items():
            meth_res = {"iteration":it,"method":norm_name}
            print("\t",norm_name)
            try:
                # Assuming X is the original data matrix
                normalized_X = norm_method(Xdepth)
            except:
                normalized_X = norm_method(scipy.sparse.csc_matrix(Xdepth))

            print("\t\tgetting_distance")
            # Compute the input distance matrix
            input_distance = euc(normalized_X)
            #r, p = pearsonr(standardize(ground_truth_ditances.flatten()), standardize(input_distance.flatten()))
            #print(f"\t\t\tr: {r:.2f} p: {p:.2e}")
            cos_sim = cosine_similarity(ground_truth_ditances.flatten(),
                            torch.tensor(input_distance.flatten()), dim=0)
            meth_res["cosine_similarity"]=cos_sim
            print(f"\t\t\tcosine similarity:{cos_sim:.4f}")

            if it == 0:
                print("\t\tgetting 3D coordinates")
                # Optimize the 3D matrix using the gradient descent method
                optimized_3D_matrix, temp_loss = optimize_3D_matrix(torch.tensor(depth_vect), 
                                                        torch.tensor(input_distance),
                                                        X=results["ground_truth"][:,0].clone().detach().numpy(),# Initialize to the ground truth distance X/Y
                                                        Y=results["ground_truth"][:,1].clone().detach().numpy()
                                                        )
                plot_3D_matrix(optimized_3D_matrix,
                            np.log(depth_vect),
                            title=f"{norm_name}\ncosine sim (fit): {-temp_loss}\ncosine sim (GT dists):{cos_sim}",
                            out_file=os.path.join("assets", "3D", norm_name)+".gif"
                            )
                meth_res["3D_coords"]=optimized_3D_matrix.clone().detach().numpy()
            else:
                meth_res["3D_coords"]=None

            # Store the results
            results[norm_name] = meth_res

    # Here you can add code to analyze, visualize, or save the results
    return results

if __name__ == "__main__":
    np.random.seed(123456)
    torch.manual_seed(123456)
    results = run_experiment(NORM)

