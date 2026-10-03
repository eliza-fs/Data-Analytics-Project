# Bank Customer Churn Analysis

Group project for Data Analytics, BINUS University (4-person team).

## Overview

Retaining existing bank customers is far cheaper than acquiring new ones, yet churn often goes unnoticed until the account is already closed. This project analyzes bank customer data (~10K customers, ~20% churn) to uncover what drives churn and to build a model that flags at-risk customers early.

## My Role

- **Exploratory Data Analysis:** analyzed churn patterns across demographic and financial features
- **Random Forest model:** built and tuned the model, including hyperparameter tuning (RandomizedSearchCV), cross-validation, robustness checks, and feature importance analysis

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook

## Methodology

1. **EDA:** 8 visualizations covering churn distribution, activity status, age, country, balance, tenure, and number of products
2. **Preprocessing:** one-hot encoding, feature scaling (for Logistic Regression), stratified 80/20 train-test split
3. **Modeling:** Logistic Regression (baseline) vs Random Forest, with class weighting to handle class imbalance (~80% stay / ~20% churn)
4. **Tuning and validation:** RandomizedSearchCV (50 candidates, 5-fold), 5-fold stratified cross-validation, and threshold selection using Youden's J statistic

## Results

| Metric | Logistic Regression | Random Forest (tuned) |
|---|---|---|
| F1-Score | 0.5062 | **0.6364** |
| ROC-AUC | 0.7748 | **0.8602** |
| False Positives | 436 | **222** |
| False Negatives | 122 | **114** |

Random Forest cross-validation: F1 0.6209 ± 0.0144, ROC-AUC 0.8615 ± 0.0063.

### Key insights

- **Inactive members** churn far more than active ones.
- **Age 40-50** is the most at-risk segment, and age is the top Random Forest feature (importance 0.30).
- **Germany** has the highest churn rate (32.4%, about double France and Spain at ~16%).
- **Customers with 3-4 products** almost all churn, which suggests aggressive cross-selling backfires.
- **High-balance customers** are more likely to leave, so the bank risks losing its most valuable customers.

### Limitation

The tuned Random Forest shows mild overfitting (train F1 0.74 vs test F1 0.62), while Logistic Regression generalizes more stably at lower performance.

## Dataset

`Dataset (Cleaned).csv` contains 9,996 customer records and 12 columns (credit score, country, gender, age, tenure, balance, number of products, credit card, active member status, estimated salary, and churn label).

## Report

The full report (in Indonesian) is available in [`Data-Analytics-Report.pdf`](Data-Analytics-Report.pdf/).
