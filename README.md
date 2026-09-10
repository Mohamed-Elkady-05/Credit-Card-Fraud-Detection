# Fraud Classification: SMOTE vs. Class-Weighting Benchmark

## Overview
Fraudulent transactions account for less than **0.2%** of all records in this dataset. Standard accuracy metrics fail in this domain—a naive classifier predicting "not fraud" on every transaction achieves 99.8% accuracy while letting 100% of fraud slip through.

This project systematically benchmarks strategies for handling extreme class imbalance in financial transaction datasets, focusing on **Precision-Recall AUC (PR-AUC)**, **Cost-Sensitive Learning**, and **Decision Threshold Tuning**.

---

## Work Division & Pipeline Architecture

| Phase | Person A (Pipeline & Resampling Specialist) | Person B (Cost-Sensitive & Metrics Specialist) |
| :--- | :--- | :--- |
| **1. Setup & Preprocessing** | Stratified Train/Test splits; feature scaling using `RobustScaler`. | Exploratory Data Analysis (EDA); target distribution & PCA correlation inspection. |
| **2. Modeling Strategy** | **Oversampling:** SMOTE / ADASYN implementation with standard classifiers. | **Algorithmic Penalization:** `class_weight='balanced'` with Random Forest & XGBoost. |
| **3. Metric Optimization** | **Threshold Tuning:** Sweeping decision cutoffs ($0.1$ to $0.9$) to maximize Recall. | **Evaluation Pipeline:** PR-AUC calculation & Cost-Matrix (FN vs. FP loss) modeling. |
| **4. Synthesis** | Documenting SMOTE computational overhead and synthetic boundary artifacts. | Authoring final comparative trade-off analysis and production takeaways. |

---

## Tech Stack & Dependencies
* **Language:** Python 3.10+
* **Libraries:** `scikit-learn`, `imbalanced-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## Key Experimental Focus
1. **SMOTE Oversampling vs. Class-Weight Adjustment:** Comparing synthetic sample generation against cost-sensitive loss penalization on Recall and Precision-Recall curves.
2. **Precision-Recall AUC over ROC-AUC:** Using PR-AUC as the primary metric to prevent true negative dominance from skewing evaluation.
3. **Threshold Optimization:** Shifting decision cutoffs below `0.50` to minimize business-critical False Negatives (uncaught fraud).

---

## Benchmark Results

| Model / Approach | Precision | Recall | PR-AUC | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (Unadjusted Random Forest) | -- | -- | -- | -- |
| SMOTE + Random Forest | -- | -- | -- | -- |
| Class-Weighted Random Forest | -- | -- | -- | -- |
| Class-Weighted XGBoost | -- | -- | -- | -- |
| **Threshold-Tuned Model (Final)** | **--** | **--** | **--** | **--** |

---

## Dataset
* **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Prevalence:** 284,807 transactions with 492 fraud cases (0.172% positive class)

---

## Quickstart & Setup

```bash
# Clone the repository
git clone [https://github.com/Mohamed-Elkady-05/Credit-Card-Fraud-Detection.git](https://github.com/Mohamed-Elkady-05/Credit-Card-Fraud-Detection.git)
cd Credit-Card-Fraud-Detection

# Install dependencies
pip install -r requirements.txt
