# Predicting Customer Churn Using Supervised Machine Learning

## Overview

This project investigates whether customer churn can be accurately predicted using supervised machine learning models applied to structured tabular data. The objective is to compare multiple classification models and select a final model based on performance metrics appropriate for imbalanced datasets.

The analysis forms part of a postgraduate assignment on regression and classification using tabular data.

---

## Research Question

To what extent can customer churn be predicted using supervised machine learning techniques applied to customer-level tabular data?

---

## Dataset

Source: Telco Customer Churn Dataset (Kaggle)

- Total observations: 7,043 customers  
- Target variable: `Churn` (1 = churned, 0 = retained)  
- Features include:
  - Tenure (months with company)
  - Contract type
  - Monthly charges
  - Total charges
  - Internet service type
  - Additional service features (e.g., OnlineSecurity, TechSupport)

The dataset contains both numerical and categorical variables and exhibits class imbalance (~26% churn rate).

---

## Methodology

### Data Preparation
- Converted `TotalCharges` to numeric format
- Removed missing values
- Dropped non-predictive identifier (`customerID`)
- Applied one-hot encoding to categorical variables
- Used stratified train-test split (80/20)

### Models Implemented

1. Logistic Regression (baseline linear model)
2. Random Forest (ensemble model)
3. Tuned Random Forest (GridSearchCV optimisation)

### Evaluation Metrics

Given class imbalance, model evaluation focused on:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Recall and ROC-AUC were prioritised due to the business cost of false negatives (missed churners).

---

## Key Findings

- Tenure is the strongest predictor of churn.
- Long-term contracts significantly reduce churn likelihood.
- Pricing variables (MonthlyCharges, TotalCharges) influence attrition.
- The tuned Random Forest achieved the best balance between recall and ROC-AUC.

---

## Final Model

The Tuned Random Forest was selected as the final model because it:

- Achieved the highest ROC-AUC
- Improved recall compared to the untuned model
- Provided a balanced trade-off between precision and recall
- Captured non-linear relationships within the data

---

## Reproducibility

### 1. Clone the Repository

