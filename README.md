# Feature Selection and Classification Pipeline

## Overview

This code implements a feature selection and classification pipeline for analyzing brain connectivity data. It supports various machine learning models and provides comprehensive metrics for model evaluation.

## Features

- **Feature Selection**: Based on the significance matrix to extract the most relevant features.
- **Model Support**: Supports logistic regression, SVM (Support Vector Machine), and Random Forest classifiers.
- **Cross-Validation**: Offers both k-fold cross-validation and leave-one-out cross-validation (LOOCV).
- **Metrics Calculation**: Computes accuracy, sensitivity, specificity, positive predictive value (PPV), negative predictive value (NPV), and area under the ROC curve (AUC).
- **Visualization**: Generates ROC curves and feature size performance maps.
- **Permutation_Test**: For permutation test.
  
  ## Dependencies
  
  To run this code, you need the following Python libraries:
- `numpy`
- `scipy`
- `sklearn`
- `matplotlib`
- `statsmodels`
- `argparse`
  You can install these dependencies using pip:
  ```bash
  pip install numpy scipy scikit-learn matplotlib statsmodels argparse

#### Required Arguments

- `--data_path`: Path to the feature file(s). Multiple paths can be provided.
- `--co_path`: Path to the significance matrix file(s). Multiple paths can be provided.

#### Optional Arguments

- `--F`: Number of feature types (default: 1).
- `--output_path`: Output directory path (default: "../output").
- `--o`: Task selection (`chinese` or `speed`).
- `--c`: Classification type (0 for binary classification; 1 for three categories; 2 for binary classification with intermediate values removed).
- `--s`: Feature size (number of features to select-). `--model`: Model choice (`svm`, `log`, or `rf`) (default: "svm").
- `--cv`: Cross-validation method (cv=0 for LOOCV; default: 10).

#### Example Usage



`python main.py  --data_path reading+work/allmatrix.mat --o chinese --c 2 --s 66 --co_path "chinese_R.mat"`

## Repository Files
| File | Description |
|------|-------------|
| [`DTI.py`](DTI.py) | Python implementation of classification algorithm  
| [`Readme.md`](Readme.md) | Readme
| [`allmatrix.mat`](allmatrix.mat) | Complete dataset containing all subject data 
| [`chinese_R.mat`](chinese_R.mat) | Significant matrix for Chinese language processing 
| [`speed_R.mat`](speed_R.mat) | Significant matrix for reading speed metrics 
| [`zreadBav.mat`](zreadBav.mat) | Z-score normalized behavioral data were used as ground truth labels for classification validation
| [`readBav.mat`](readBav.mat) | raw behavioral data
