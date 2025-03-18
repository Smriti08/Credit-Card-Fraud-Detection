# Credit Card Fraud Detection

## Overview

This project aims to detect fraudulent credit card transactions using machine learning techniques. It leverages a dataset of anonymized transactions and applies various classification models to differentiate between genuine and fraudulent transactions.

## Dataset

- The dataset consists of transactions made by European credit cardholders in September 2013.
- It contains **284,807** transactions, out of which **492** are frauds (\~0.172% of all transactions).
- Features are transformed using **Principal Component Analysis (PCA)**, except for `Time` and `Amount`.
- **Target Variable**: `Class` (0 = Normal, 1 = Fraudulent)

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Seaborn, Matplotlib
- Flask/Streamlit (for deployment)
- Joblib (for model serialization)

## Steps Involved

### 1. Data Preprocessing

- Load the dataset using Pandas.
- Remove duplicate transactions.
- Normalize the `Amount` column using `StandardScaler`.
- Drop the `Time` column (not needed for prediction).

### 2. Exploratory Data Analysis

- Check for missing values.
- Visualize class distribution using `sns.countplot()`.
- Identify imbalance in the dataset (highly skewed towards non-fraudulent transactions).

### 3. Handling Imbalance

- **SMOTE (Synthetic Minority Over-sampling Technique)** is used to balance the dataset by oversampling fraudulent transactions.

### 4. Model Training & Evaluation

We trained multiple models:

- **Logistic Regression**
- **Decision Tree Classifier**

For each model, we calculated:

- Accuracy
- Precision
- Recall
- F1 Score

### 5. Model Deployment

- Trained a **DecisionTreeClassifier** and saved the model using `joblib`.
- Implemented a prediction function that classifies transactions as "Normal Transaction" or "Fraud Transaction".
- To deploy the model, we used **Flask/Streamlit** (not included here but recommended for real-world applications).

## Model Training Code

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib

# Splitting Data
X = data.drop('Class', axis=1)
y = data['Class']
X_res, y_res = SMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train models
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}

for name, clf in classifiers.items():
    print(f"\n=========={name}===========")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")

# Save model
joblib.dump(classifiers['Decision Tree Classifier'], "credit_card_model.pkl")
```

## Results

After balancing the dataset with **SMOTE** and training models, we achieved:

- **Logistic Regression**: High precision but lower recall
- **Decision Tree Classifier**: Better recall, suitable for fraud detection

## How This Helps in Fraud Detection

- Identifies fraudulent transactions in real-time.
- Reduces financial losses for banks and customers.
- Enhances security by detecting unusual spending patterns.
- Can be integrated into banking systems for automatic fraud prevention.

## Next Steps

- Deploy the model using **Flask** or **Streamlit**.
- Use **Random Forest Classifier** or **XGBoost** for better accuracy.
- Implement real-time detection with a web interface.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python fraud_detection.py
   ```

---

### 📌 **Note:**

This dataset is anonymized and lacks real transaction details. In a production environment, consider integrating with actual financial transaction data and adding more advanced anomaly detection techniques.







