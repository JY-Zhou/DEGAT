
# DEGAT
DEGAT : Diffusion-Enhanced Graph Attention Network for Cancer Type Classfication

## Overview
This repository contains code necessary to run DEGAT model.DEGAT is an end-to-end model mainly based on graph attention network (GAT). Protein expression information and proteins interaction information are utilized to predict the type or the subtype of cancer. DEGAT is tested on real-world cancer database "The Cancer Genome Atlas Program"(TCGA)https://www.cancer.gov/ccg/research/genome-sequencing/tcga and outperforms several machine learning and deep learning methods in most of the metrics.

## Folder Specification

### data(Including the datasets used in the experiment)
RPPA_data_of_BRCA.csv: This file is the patient's protein expression data, this file could be downloaded from TCGA database.

clinical_data_of_BRCA.csv: This file are the patient's clinical records, which contain the patient's diagnosis, stage of cancer and subtype of BRCA, etc.this file could be downloaded from TCGA database.

ppi_network_of_BRCA.csv: This file describe the protein-protein interaction of BRCA. this file could be downloaded from STRING database.https://string-db.org

RPPA_data_of_pancancer.csv: This file is the patient's protein expression data, this file could be downloaded from TCGA database.

ppi_network_of_pancancer.csv: This file describe the protein-protein interaction of pancancer. this file could be downloaded from STRING database.https://string-db.org

### codes
gat.py: This file describes a graph attention network.

models.py: This file contains the overall network architecture of DEGAT, including data processing, diffusion module and graph classifier.


sparsification_curve.py : This file shows the effect of different parameters on sparsification.

## Requirements
- pandas: 1.3.0
- tensorflow: 2.13.0
- scikit-learn: 0.24.2
- numpy: 1.21.1
- Python 3.7
- matplotlib 3.3.0
- seaborn 0.12
- scipy 1.19.2
