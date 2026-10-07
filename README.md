# Credit Risk Prediction using Decision Tree

## Overview

This project implements a machine learning classification model for predicting credit default risk using applicant financial and credit-related information.

The project follows an end-to-end machine learning workflow, including data preprocessing, feature engineering, model experimentation, evaluation, and prediction on new applicant data.

## Dataset

The project uses the **Credit Risk Dataset** from Kaggle:

https://www.kaggle.com/datasets/laotse/credit-risk-dataset

The dataset contains information related to applicants' income, employment history, loan characteristics, credit history, home ownership, loan purpose, and previous default history.

## Project Workflow

The project follows these main steps:

1. Load and inspect the dataset
2. Handle missing values
3. Select relevant features
4. Encode categorical variables using One-Hot Encoding
5. Split the dataset into training and testing sets
6. Experiment with different Decision Tree depths
7. Evaluate model performance using multiple classification metrics
8. Select the best-performing configuration
9. Generate predictions for new applicant data

## Data Preprocessing

The dataset was prepared before model training through several preprocessing steps.

### Missing Values

Missing values in `person_emp_length` were handled using median imputation:

```python
df["person_emp_length"] = df["person_emp_length"].fillna(
    df["person_emp_length"].median()
)
```

### Feature Selection

The following features were excluded from the final model:

* `person_age`
* `loan_grade`
* `loan_int_rate`

### Categorical Encoding

Categorical features were converted into numerical representations using One-Hot Encoding:

* `person_home_ownership`
* `loan_intent`
* `cb_person_default_on_file`

## Model Development and Hyperparameter Experimentation

Instead of selecting a Decision Tree depth arbitrarily, multiple `max_depth` values were evaluated to understand their effect on model performance.

The model was tested with different configurations, including:

```text
2, 3, 4, 6, 8, 10, 12, 15, 20, None
```

For each configuration, the following evaluation metrics were calculated:

* Accuracy
* Precision
* Recall
* F1-Score

After comparing the results, the configuration with the strongest overall performance was selected for the final model.

The final model uses:

```python
DecisionTreeClassifier(
    max_depth=15,
    random_state=42
)
```

This approach helped avoid choosing the model configuration based on assumption alone and provided a more systematic basis for model selection.

## Model Evaluation

The final model is evaluated using:

* **Accuracy** — Overall proportion of correct predictions
* **Precision** — Proportion of predicted defaults that were actually defaults
* **Recall** — Proportion of actual defaults correctly identified
* **F1-Score** — Harmonic mean of precision and recall
* **Confusion Matrix** — Detailed breakdown of correct and incorrect classifications

## Prediction on New Data

The project also includes an interactive prediction component.

Users can provide applicant information such as:

* Income
* Employment length
* Loan amount
* Loan-to-income percentage
* Credit history length
* Home ownership
* Loan intent
* Previous default history

The trained model then generates a predicted credit-risk class.

### Target Classes

```text
0 → Non-default
1 → Default
```

Therefore, the prediction is interpreted as:

```text
0 → Lower default risk
1 → Higher default risk
```

> Note: The model provides a prediction based on patterns learned from historical data and should not be interpreted as a guaranteed financial decision.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Decision Tree Classifier

## Project Structure

```text
credit-risk-prediction/
│
├── credit_risk_prediction.py
├── requirements.txt
└── README.md

## Usage

Run the Python script:

```bash
python credit_risk_prediction.py
```

The program will request applicant information through the terminal and generate a credit-risk prediction.

## Future Improvements

Potential improvements include:

* Hyperparameter tuning using GridSearchCV
* Cross-validation
* Comparison with Random Forest and Gradient Boosting models
* Feature importance analysis
* Probability-based risk scoring
* Deployment through Flask or FastAPI
* Development of a web-based user interface

## Author

**Umer Hayat**

Machine Learning | Python | Data Science
