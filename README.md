

## Results
Here are the simulated cells in similarity space, as deteremined by a gradient descent algorithm, optimizing X and Y to take the the distances in the original ambient space, and recapitulate those distances in this 3D space (as determined by minimizing cosine distance between cell-cell distances in the 3D projection and the distances in ambient space), while locking the Z-axis to be equal to the min-max normalized log(depth).

In this specific case, because we know the ground truth is two blobs, we aren't very worried about "cutting" links in the topology, as would be necessary where the meaningful dimensionality is greater than two.

Note that they annotated cosine similarity maxes out at 1 & the closer the value is to 1, the better the fit was in recapitulating the distances in the ambient space (after normalization). Then we also show the cosine similarity to the ground-truth ambient distances. These are also colorized by the log depth for an added visual aid, even though this is also captured on the Z axis.


Ground truth: (before log-normal Poisson sampling process)

![Ground Truth](assets/3D/ground_truth.gif)


Raw Counts: (After log-normal Poisson sampling process)

![raw counts](assets/3D/raw.gif)


Log transformed:

![log](assets/3D/log.gif)


Relative Log Expression (RLE): 

![rle](assets/3D/rle.gif)

Pearson Residuals:
![pear](assets/3D/pear.gif)

PF:

![pf](assets/3D/pf.gif)


PF-log:

![pf_log](assets/3D/pf_log.gif)


PF-log-PF:

![pf_log_pf](assets/3D/pf_log_pf.gif)


Sqrt:

![sqrt](assets/3D/sqrt.gif)


Log(Counts per Million):

![cpm_log](assets/3D/cpm_log.gif)


Log(Counts per 10k):

![cp10k_log](assets/3D/cp10k_log.gif)



## Methods
### Install 
`python3 -m pip install -e .`

### Use
Then Just run the experiment:
`python3 count_depth_effects/src/run_experiment.py`

### Normalization methods
The normalization methods implemented here were from here, so if you cite this work, pleas also cite that:
-  Depth normalization for single-cell genomics count data
A. Sina Booeshaghi, Ingileif B. Hallgrímsdóttir, Ángel Gálvez-Merchán, Lior Pachter
bioRxiv 2022.05.06.490859; doi: https://doi.org/10.1101/2022.05.06.490859 

Relative Log Expression was also used, as implemented here:
- https://github.com/scottyler89/st_rle
But that repository was just a pythonic implementation of the original in DESeq, here:
- Anders S, Huber W. Differential expression analysis for sequence count data. Genome Biol. 2010;11(10):R106. doi: 10.1186/gb-2010-11-10-r106. Epub 2010 Oct 27. PMID: 20979621; PMCID: PMC3218662.

### Detailed Methods
First, we generate the 
