# Credit Scoring Model

A machine learning project to predict an individual's creditworthiness using classification algorithms.

## Project Overview

This project implements a credit scoring model that predicts whether an individual is creditworthy (1) or not (0) based on financial data. The model uses various classification algorithms and evaluates their performance using multiple metrics.

## Features

- **Data Preprocessing**: Loading and cleaning financial data
- **Feature Engineering**: Creating meaningful features from raw financial data
- **Multiple Models**: Implementation of three classification algorithms:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- **Comprehensive Evaluation**: Performance metrics including:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Score
- **Visualization**: ROC curves and feature importance plots

## Dataset

The dataset includes the following features:
- `age`: Age of the individual
- `income`: Annual income
- `debt`: Total debt amount
- `payment_history`: Payment history score (0-1)
- `credit_utilization`: Credit utilization ratio
- `loan_amount`: Loan amount requested
- `employment_length`: Years of employment
- `credit_score`: Credit score
- `creditworthy`: Target variable (1 = creditworthy, 0 = not creditworthy)

## Feature Engineering

The following engineered features are created:
- `debt_to_income_ratio`: Debt divided by income
- `loan_to_income_ratio`: Loan amount divided by income
- `payment_consistency`: Payment history multiplied by employment length
- `credit_utilization_score`: (1 - credit_utilization) multiplied by credit score

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to train and evaluate the models:

```bash
python credit_scoring.py
```

The script will:
1. Load and preprocess the data
2. Perform feature engineering
3. Train three classification models
4. Evaluate model performance
5. Generate visualizations (ROC curves and feature importance plots)
6. Save model comparison results to CSV

## Model Performance

The script outputs comprehensive evaluation metrics for each model and saves a comparison table to `model_comparison.csv`.

## Output Files

- `roc_curves.png`: ROC curves for all models
- `feature_importance_decision_tree.png`: Feature importance for Decision Tree
- `feature_importance_random_forest.png`: Feature importance for Random Forest
- `model_comparison.csv`: Comparison of model performance metrics

## Dependencies

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## License

This project is for educational purposes.
