import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from copy import deepcopy
from count_split.count_split import multi_split
from pyminer_norm.downsample import new_rewrite_get_transcript_vect as ds
from pyminer_norm.downsample import downsample_mat
from sklearn.metrics.pairwise import euclidean_distances as euc
from anticor_features.anticor_stats import no_p_spear
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from dclustval.cluster import do_cluster_validation
from anticor_features.anticor_features import get_anti_cor_genes
## first make some dummy single cell data
# I'm making absolutely no claim that this is a "good" simulation of 
# single cell data, just a quick whip-up of something that looks somewhat similar
np.random.seed(123456)
n_genes = 5000
n_group_specific_genes = 500
n_groups = 2
# make sure we have enough genes for this sim
assert n_group_specific_genes*n_groups < n_genes
cells_per_group = 500
n_cells = int(n_groups*cells_per_group)
# We'll make two clusters, but first we'll make a "base" transcriptome
base_transcriptome_lambdas = np.random.negative_binomial(.1,.01,size=n_genes)
group_transcriptome_lambdas = []
min_lambda = 0.1
group_vect = []
## make some cell-type specific gene choices
cell_type_genes = []
for g in range(1,n_groups+1):
    cell_type_genes.append(np.arange((g-1)*n_group_specific_genes,(g)*n_group_specific_genes))

## make the main transcript lambdas for each group
for g in range(n_groups):
    temp_lamb=deepcopy(base_transcriptome_lambdas)
    temp_lamb[cell_type_genes[g]]=np.random.negative_binomial(.1,.01,size=len(cell_type_genes[g]))
    ## make sure we're at noise level for other group's genes
    for g2 in range(n_groups):
        if g!=g2:
            temp_lamb[cell_type_genes[g2]]=min_lambda
    # add low level non-specific noise for zeros
    temp_lamb[temp_lamb<min_lambda]=min_lambda
    group_transcriptome_lambdas.append(temp_lamb)
    # also add the group membership vector
    group_vect += [g for i in range(cells_per_group)]


# here cells are in rows, genes are in columns
X = np.zeros((n_cells,n_genes))
for cell_idx in range(n_cells):
    temp_group = group_vect[cell_idx]
    temp_transcript_vect = np.zeros((n_genes))
    for t in range(n_genes):
        # Add a poisson sample for each gene's lambda +noise for their given group
        temp_transcript_vect[t]=np.random.poisson(max(min_lambda,group_transcriptome_lambdas[temp_group][t]))
    X[cell_idx,:]=deepcopy(temp_transcript_vect)


# If you want to check out how insane the depth effect is, even with nice distributions
# Set this to True and see how badly over-clustered things can appear without depth
# normalization!
def depth_norm(in_mat,target=None):
    # Normalizes the row
    loading = np.sum(in_mat,axis=1)
    if target is None:
        # Default to counts/avg total counts
        target = np.mean(loading)
    loading_factors = loading/target
    return(in_mat / loading_factors[:,None])


def tenk_norm(in_mat):
    return(depth_norm(in_mat, target=10000))


checkout_depth = True
if checkout_depth:
    depth_vect = np.random.lognormal(np.log(2500),1, size=n_cells)
    Xdepth=deepcopy(X)
    for cell_idx in range(n_cells):
        Xdepth[cell_idx,:]=ds(X[cell_idx,:],int(depth_vect[cell_idx]))
    Xdepth = Xdepth.T
else:
    # For the "main event" we'll account for depth through downsampling though
    # We'll downsample to 2500 counts per cell
    Xdepth = downsample_mat(X.T,2500).todense()


