
# DEGAT
DEGAT : Diffusion-Enhanced Graph Attention Network for Cancer Type Classfication

## Overview
This repository contains code necessary to run DEGAT model.DEGAT is an end-to-end model mainly based on graph attention network (GAT). Protein expression information and protein-protein interaction information are utilized to predict the type or the subtype of cancer. DEGAT is tested on real-world cancer database "The Cancer Genome Atlas Program"(TCGA)https://www.cancer.gov/ccg/research/genome-sequencing/tcga and outperforms several machine learning and deep learning methods in most of the metrics.

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

best_model.h5:This file is the best-scored model we trained.

best_model.py:You can use this file to run best_model.h5.

sparsification_curve.py : This file shows the effect of different parameters on sparsification.

requirements.txt:This file is the required environment to run these codes.

## Requirements
- pandas: 2.0.1
- tensorflow: 2.7.0
- Keras 2.7.0
- scikit-learn: 1.2.2
- numpy: 1.24.3
- Python 3.9.13
- matplotlib 3.7.1
- seaborn 0.12.2
- scipy 1.10.1
```bash
pip install -r requirements.txt
