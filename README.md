# A Mixture of Experts Approach to Predicting Protein-Molecule Binding Affinity
# Overview
The goal of this project is to use the BELKA dataset to develop an MoE model for predicting protein/molecule bonding, an essential aspect of drug discovery.

# Data
- Source: BELKA dataset provided by Leash Biosciences.
- Description: Contains molecular structures in SMILES format and binary binding classifications for three protein targets. This is stored in a training and testing parquet file which will be loaded into respective SQL databases
- DBMS: SQLite

# Tech Stack
## Core Technologies
- Python: Primary programming language for all scripts and workflows.
- PyTorch: Framework for building and training deep learning models.
- Polars: Fast DataFrame library for efficient data preprocessing.
- Apache Spark: Distributed data processing for large-scale dataset handling.
- SQLite: Lightweight relational database for storing and querying metadata.

## Key Libraries and Tools
- RDKit: For molecular data preprocessing, visualization, and feature extraction from SMILES.
- Hugging Face Transformers: For leveraging pretrained models like ChemBERTa for SMILES-based representations.
- Scikit-learn: For traditional machine learning techniques and evaluation metrics.
- Optuna: For hyperparameter tuning to optimize model performance.
- Matplotlib/Seaborn: For data visualization and analysis.

## Potential Pre-Trained Models
- ChemBERTa: A pretrained transformer model for molecular SMILES strings, useful for feature representation.
- DeepChem: A library with pretrained models and utilities for chemistry and drug discovery.

# Project Structure
```
project-root/
│
├── data/                  # Storing source data    
│   ├── train.parquet
│   └── test.parquet
│
├── notebooks/             # Jupyter notebooks for exploration and prototyping
│   ├── EDA.ipynb  
│   ├── feature_engineering.ipynb
│   └── model_training.ipynb
│
├── scripts/               # Core scripts
│   ├── data_processing/   # Scripts for data preprocessing
│   ├── models/            # Model definition and training scripts
│   ├── utils/             # Utility functions
│   └── main.py            # Main script for running the pipeline
│
├── results/               # Model outputs, logs, and evaluation results
│
├── requirements.txt       
├── README.md           
└── LICENSE    
```

# Notes
I will be working on a research paper (primarily focused on results) and a technical write up (extremely in-depth write up of methods) in parallel with this project. The links to each of these can be found below:
Research Paper: https://docs.google.com/document/d/15ld0bWlFiwg-XrWRe_PTkrl9rGPPTUjsKDLNHat_Mxc/edit?tab=t.0

Technical Writeup: https://docs.google.com/document/d/1ZfZ5g1Lj9UebW897y2bz3oIcPuDZVjvgFPP7W6i_YSg/edit?tab=t.0