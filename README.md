# Salary Prediction Project

## Project Overview
This project predicts employee salaries using Machine Learning. The objective is to understand the complete ML workflow including data preprocessing, Exploratory Data Analysis (EDA), feature engineering, model training, evaluation, and saving the trained model.

## Dataset
Dataset: AI and Data Science Job Salaries 2026 Dataset

- Total Records: 5000
- Total Features: 27
- Target Variable: salary_usd

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Project Workflow

1. Load Dataset
2. Data Inspection
3. Data Cleaning
4. Exploratory Data Analysis
5. Categorical Feature Encoding
6. Train-Test Split (80%-20%)
7. Linear Regression Model Training
8. Salary Prediction
9. Model Evaluation
10. Model Saving

## Data Cleaning

Performed:
- Missing value checking
- Duplicate value checking
- Data type inspection

Result:
- No missing values found
- No duplicate records found

## Exploratory Data Analysis

Created graphs:

- Salary Distribution Histogram
- Experience vs Salary Scatter Plot
- Salary Box Plot
- Feature Correlation Heatmap
- Top Paying Jobs Analysis

## Machine Learning Model

Algorithm Used:

Linear Regression

Train-Test Split:

- Training Data: 80%
- Testing Data: 20%

## Model Evaluation

Metrics Used:

- MAE
- MSE
- RMSE
- R² Score

Model Performance:

R² Score: 0.8062

## Saved Model

Trained model saved at:

models/salary_prediction_model.pkl

## Project Structure
