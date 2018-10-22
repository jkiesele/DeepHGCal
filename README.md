DeepHGCal 
==================

Based on the DeepJetCore framework (https://github.com/DL4Jets/DeepJetCore) [CMS-AN-17-126] for HGCal reconstruction purposes.

The framework consists of two parts:
1) HGCal ntuple converter (Dependencies: root5.3/6)
2) DNN training/evaluation (Dependencies: DeepJetCore and all therein).
   
The DeepJetCore framework and the DeepHGCal framework should be checked out to the same parent directory.
Before usage, always set up the environment by sourcing XXX_env.sh


## Usage

The experiments are usually conducted in three steps:
1. Training
2. Testing (dumping of inference result somewhere on disk)
3. Plotting and anlysis

### Training

```
python bin/train/train_file.py path/to/config.ini config_name
```


### Testing

```
python bin/train/train_file.py path/to/config.ini config_name --test True
```


### Plotting and analysis

```
python bin/plot/plot_file.py path/to/config.ini config_name
```

For clustering, the plot_file can be `plot_inference_clustering.py`

It will plot the resolution histogram as well as output mean and variance of resolution on stdout.