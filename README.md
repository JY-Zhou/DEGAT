
# DEGAT
DEGAT : Diffusion-Enhanced Graph Attention Network for Cancer Type Classfication

## Overview
This repository contains code necessary to run DEGAT model.DEGAT is an end-to-end model mainly based on graph attention networks (GAT). Protein expression information and proteins interaction information are utilized to predict the type or the subtype of cancer. KEGAT is tested on real-world clinical database [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) and outperforms several machine learning and deep learning methods in most effectiveness measures.

## Folder Specification

### data(Including the data set used in the experiment)
RPPA_data_of_BRCA.csv: This file is the patient's protein expression data, this file could be downloaded from TCGA database.

clinical_data_of_BRCA.csv: This file are the patient's clinical records, which contain the patient's diagnosis, stage of cancer and subtype of BRCA, etc.this file could be downloaded from TCGA database.

ppi_network_of_BRCA: This file describe the protein-protein interaction. this file could be downloaded from STRING database.（https://string-db.org/）



### codes
layers.py: This file describes a graph neural network.

models.py: This file contains the overall network architecture of KDGN, including clinical records, domain knowledge processing and confidence generation networks.

new.py : This file defines the graph transformer neural network architecture.

train_KDGN.py : This file trains the KDGN model.

util.py: This file contains some defined functions.

## Requirements
- pandas: 1.3.0
- tensorflow: 2.13.0
- scikit-learn: 0.24.2
- numpy: 1.21.1
- Python 3.7
- matplotlib 3.3.0
