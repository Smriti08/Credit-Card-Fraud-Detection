# Credit Card Fraud Detection
Overview

This project aims to detect fraudulent credit card transactions using machine learning techniques. It leverages a dataset of anonymized transactions and applies various classification models to differentiate between genuine and fraudulent transactions.

Dataset

The dataset consists of transactions made by European credit cardholders in September 2013.

It contains 284,807 transactions, out of which 492 are frauds (~0.172% of all transactions).

Features are transformed using Principal Component Analysis (PCA), except for Time and Amount.

Target Variable: Class (0 = Normal, 1 = Fraudulent)

Technologies Used

Python

Pandas, NumPy

Scikit-learn

Imbalanced-learn (SMOTE)

Seaborn, Matplotlib

Flask/Streamlit (for deployment)

Joblib (for model serialization)

Steps Involved

1. Data Preprocessing

Load the dataset using Pandas.

Remove duplicate transactions.

Normalize the Amount column using StandardScaler.

Drop the Time column (not needed for prediction).

2. Exploratory Data Analysis

Check for missing values.

Visualize class distribution using sns.countplot().

Identify imbalance in the dataset (highly skewed towards non-fraudulent transactions).

3. Handling Imbalance

SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the dataset by oversampling fraudulent transactions.

4. Model Training & Evaluation

We trained multiple models:

Logistic Regression

Decision Tree Classifier

For each model, we calculated:

Accuracy

Precision

Recall

F1 Score

5. Model Deployment

Trained a DecisionTreeClassifier and saved the model using joblib.

Implemented a prediction function that classifies transactions as "Normal Transaction" or "Fraud Transaction".

To deploy the model, we used Flask/Streamlit (not included here but recommended for real-world applications).

