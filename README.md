

## Results
Ground truth: (before log-normal Poisson sampling process)
![Ground Truth](assets/ground_truth.gif)
Raw Counts: (After log-normal Poisson sampling process)
![raw counts](assets/raw.gif)
Log transformed:
![log](assets/.gif)
Relative Log Expression (RLE): 
![rle](assets/rle.gif)
PF:
![pf](assets/pf.gif)
PF-log:
![pf_log](assets/pf_log.gif)
PF-log-PF:
![pf_log_pf](assets/pf_log_pf.gif)
Sqrt:
![sqrt](assets/sqrt.gif)
Log(Counts per Million):
![cpm_log](assets/cpm_log.gif)
Log(Counts per 10k):
![cp10k_log](assets/cp10k_log.gif)



## Methods
### Install 
`python3 -m pip install -e .`

### Use
Then Just run the experiment:
`python3 count_depth_effects/src/run_experiment.py`

### Detailed Methods
First, we generate the 
