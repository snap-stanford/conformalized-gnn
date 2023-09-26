# Conformalized Graph Neural Networks

This repository hosts the code base for

```
Uncertainty Quantification over Graph with Conformalized Graph Neural Networks
Kexin Huang, Ying Jin, Emmanuel Candès, Jure Leskovec
NeurIPS 2023, Spotlight
```

### Abstract

Graph Neural Networks (GNNs) are powerful machine learning prediction models on graph-structured data. However, GNNs lack rigorous uncertainty estimates, limiting their reliable deployment in settings where the cost of errors is significant. We propose conformalized GNN (CF-GNN), extending conformal prediction (CP) to graph-based models for guaranteed uncertainty estimates. Given an entity in the graph, CF-GNN produces a prediction set/interval that provably contains the true label with pre-defined coverage probability (e.g.~90%). We establish a permutation invariance condition that enables the validity of CP on graph data and provide an exact characterization of the test-time coverage. Besides valid coverage, it is crucial to reduce the prediction set size/interval length for practical use. We observe a key connection between non-conformity scores and network structures, which motivates us to develop a topology-aware output correction model that learns to update the prediction and produces more efficient prediction sets/intervals. Extensive experiments show that CF-GNN achieves any pre-defined target marginal coverage while significantly reducing the prediction set/interval size by up to 74% over the baselines. It also empirically achieves satisfactory conditional coverage over various raw and network features. 


<p align="center"><img src="./fig/logo.png" alt="logo" width="800px" /></p>


## Installation

Install Torch and PyG following [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and then do

```bash
pip install -r requirements.txt
```

## Run


### Datasets

**Classification datasets are supported in PyG. For regression datasets, download from [this link](https://drive.google.com/file/d/1qHqR4JYc9fMVppOj1K9x89OPh4AtjYQ1/view?usp=sharing) and put the folder under this repository.**

Here are the list of datasets for classification tasks: `Cora_ML_CF`, `DBLP_CF`, `CiteSeer_CF`, `PubMed_CF`, `Amazon-Computers`, `Amazon-Photo`, `Coauthor-CS`, `Coauthor-Physics`

Here are the list of datasets for regression tasks: `Anaheim`, `ChicagoSketch`, `county_education_2012`, `county_election_2016`, `county_income_2012`, `county_unemployment_2012`, `twitch_PTBR`


### Pre-trained GNN base models

**To reproduce the paper result, please use the fixed pre-trained GNN base models from [this link](https://drive.google.com/file/d/1-z17AWIkDJ7LoI9OG_qQfbCZ9BqDxGbx/view?usp=sharing).** This makes sure the gain is from the conformal adjustment instead of the noise in the base model training. After downloading and unzipping this link, please put the `model` folder under this repository.

If you wish to re-train GNN base model, simply remove the base model folder in this repository and the model will train again.

### Key Arguments

`--model`: base GNN model, select from 'GAT', 'GCN', 'GraphSAGE', 'SGC'
`--dataset`: dataset name, select from 'Cora_ML_CF', 'CiteSeer_CF', 'DBLP_CF', 'PubMed_CF', 'Amazon-Computers', 'Amazon-Photo', 'Coauthor-CS', 'Coauthor-Physics', 'Anaheim', 'ChicagoSketch', 'county_education_2012', 'county_election_2016', 'county_income_2012', 'county_unemployment_2012', 'twitch_PTBR'
`--device`: cuda device
`--alpha`: pre-specified miscoverage rate, default is 0.1
`--optimal`: use optimal hyperparameter set
`--hyperopt`: conduct a sweep of hyperparameter optimization
`--num_runs`: number of runs, default is 10
`--wandb`: turn on weight and bias tracking
`--verbose`: verbose mode, print out log (incl. training loss)
`--optimize_conformal_score`: for classification only, options: aps and raps
`--not_save_res`: default is saving the result to the pred folder, by adding this flag, you choose to NOT save the result
`--epochs`: number of epochs for conformal correction


### Training CF-GNN

```bash
python train.py --model GCN \
                --dataset Cora_ML_CF \
                --device cuda \
                --alpha 0.1\
                --optimal \
                --num_runs 1
```

### Training baseline models

For classification datasets `Cora_ML_CF`, `DBLP_CF`, `CiteSeer_CF`, `PubMed_CF`, `Amazon-Computers`, `Amazon-Photo`, `Coauthor-CS`, `Coauthor-Physics`:

All baselines are calibration methods, choose `calibrator` from `TS` `VS` `ETS` `CaGCN` `GATS`.

```bash
python train.py --model GCN \
                --dataset Cora_ML_CF \
                --device cuda \
                --alpha 0.05 \
                --conf_correct_model Calibrate \
                --calibrator TS
```

For regression datasets `Anaheim`, `ChicagoSketch`, `county_education_2012`, `county_election_2016`, `county_income_2012`, `county_unemployment_2012`, `twitch_PTBR`:

To use `mcDropout`:

```bash
python train.py --model GCN \
                --dataset Anaheim \
                --device cuda \
                --alpha 0.05 \
                --conf_correct_model mcdropout_std
```

To use `BayesianNN`:

```bash
python train.py --model GCN \
                --dataset Anaheim \
                --device cuda \
                --alpha 0.05 \
                --bnn
```

To use `QuantileRegression`:

```bash
python train.py --model GCN \
                --dataset Anaheim \
                --device cuda \
                --alpha 0.05 \
                --conf_correct_model QR
```

### Launching a hyper-parameter search for CF-GNN

```bash
python train.py --model GCN \
                --dataset Cora_ML_CF \
                --device cuda \
                --alpha 0.1 \          
                --hyperopt
```

### Adjusting pre-specified coverage 1-alpha

This is the script for Fig 5(1).

```bash
for data in Anaheim Cora_ML_CF
do
for alpha in 0.05 0.1 0.15 0.2 0.25 0.3 
do
python train.py --model GCN --dataset $data --device cuda --optimal --alpha $alpha
done
done
```

### Adjusting holdout calibration set fraction

This is the script for Fig 5(2).

```bash
for data in Anaheim Cora_ML_CF
do
for calib_frac in 0.1 0.3 0.7 0.9
do
python train.py --model GCN --dataset $data --device cuda --optimal --calib_fraction $calib_frac
done
done
```


## Citation

```
@article{huang2023conformalized_gnn,
  title={Uncertainty quantification over graph with conformalized graph neural networks},
  author={Huang, Kexin and Jin, Ying and Candes, Emmanuel and Leskovec, Jure},
  journal={NeurIPS},
  year={2023}
}
```
