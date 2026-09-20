# 🚗 Used Car Price Prediction

A machine learning project for predicting used car selling prices using regression models, feature engineering, exploratory data analysis, and hyperparameter tuning.

## 📌 Overview

This project builds an end-to-end regression pipeline to estimate the selling price of used cars.

The workflow includes:

- Exploratory Data Analysis (EDA)
- Data cleaning and validation
- Feature engineering
- Categorical encoding
- Feature scaling
- Multiple regression models
- Model comparison
- XGBoost hyperparameter tuning
- Model evaluation
- Saving trained model and preprocessing artifacts

## 📊 Dataset

The dataset contains **4,340 used-car records** with 9 original features.

### Features

- `name` — Car model/name
- `year` — Manufacturing year
- `selling_price` — Target selling price in lakhs
- `present_price` — Current/new price in lakhs
- `kms_driven` — Kilometres driven
- `fuel_type` — Petrol, Diesel, or CNG
- `seller_type` — Individual or Dealer
- `transmission` — Manual or Automatic
- `owner` — Number of previous owners

There were no missing values or duplicate rows in the dataset.

## ⚙️ Feature Engineering

Additional features were created to capture vehicle age and usage patterns:

- `brand` — Extracted from the car name
- `car_age` — Vehicle age relative to 2021
- `kms_per_year` — Average kilometres driven per year
- `is_automatic` — Automatic transmission indicator
- `is_first_owner` — First-owner indicator
- `is_dealer` — Dealer seller indicator

Categorical features were encoded and numerical features were standardized before model training.

## 🤖 Models

The following regression models were evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Evaluation metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## 📈 Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Gradient Boosting | 0.560 | 1.013 | 0.9902 |
| XGBoost | 0.598 | 1.155 | 0.9872 |
| Random Forest | 0.615 | 1.200 | 0.9862 |
| Decision Tree | 0.805 | 1.407 | 0.9810 |
| Lasso | 2.505 | 4.062 | 0.8419 |
| Linear Regression | 2.544 | 4.067 | 0.8416 |
| Ridge | 2.544 | 4.067 | 0.8416 |

## 🔧 XGBoost Hyperparameter Tuning

XGBoost was further optimized using 5-fold cross-validation with GridSearchCV.

### Best Parameters

```text
learning_rate = 0.05
max_depth     = 3
n_estimators  = 300
subsample     = 0.8
