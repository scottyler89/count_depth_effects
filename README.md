

## Results
Ground truth: (before log-normal Poisson sampling process)
![Ground Truth](assets/3D/ground_truth.gif)
Raw Counts: (After log-normal Poisson sampling process)
![raw counts](assets/3D/raw.gif)
Log transformed:
![log](assets/3D/.gif)
Relative Log Expression (RLE): 
![rle](assets/3D/rle.gif)
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

### Detailed Methods
First, we generate the 
