# Credit Risk Prediction using Decision Tree

## Overview

This project implements a machine learning classification model to predict
credit default risk using applicant financial and credit-related information.

The model uses a Decision Tree Classifier to classify applicants into
different credit risk categories based on historical data.

## Dataset

The project uses the Credit Risk Dataset available on Kaggle:

https://www.kaggle.com/datasets/laotse/credit-risk-dataset

## Features

The model uses features including:

- Person income
- Employment length
- Loan amount
- Loan percent of income
- Credit history length
- Home ownership
- Loan intent
- Previous default history

## Data Preprocessing

The following preprocessing steps were performed:

- Handled missing values using median imputation
- Removed selected features
- Applied one-hot encoding to categorical variables
- Used stratified train-test splitting

## Machine Learning Model

**Decision Tree Classifier**

```python
DecisionTreeClassifier(
    max_depth=15,
    random_state=42
)
