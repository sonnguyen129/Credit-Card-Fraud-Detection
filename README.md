# Credit-Card-Fraud-Detection

![Python](https://img.shields.io/badge/Python-3.8.10-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-LightGBM-orange)

## Introduction
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

## Problem Statement
With the provided information, build a model to predict whether this customer will commit fraud when using a credit card or not.

## Description
This data set is [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

### Exploratory Data Analysis
* Exploratory Data Analysis is the first step of understanding your data and acquiring domain knowledge. 

### Data Preprocessing
* Log transform ```amout``` feature
* Handeling Data Imbalance: I used **SMOTE** method for balancing the dataset. 
* **Robust scaling** all data

### Features Selection:
* On using **Correlation** method, I don't need to drop any features.

### Model Training
* Widely used ML models: Logistic Regression, Decision Tree, Random Forest, LightGBM, Catboost, XGBoost, AdaBoost
* SOTA model: [TabNet](https://arxiv.org/pdf/1908.07442.pdf)
* Hyperparameter Tuning **LightGBM** with ```Optuna```

### Explainable
* Used **SHAP** based on test prediction.

## Installation
