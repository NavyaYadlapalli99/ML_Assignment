# Machine Learning Assignment 2 – Classification Models

## 1. Problem Statement

The objective is to implement and compare multiple classification algorithms on one public classification dataset and deploy an interactive Streamlit application that allows users to upload test data, select a model, and view evaluation results.

## 2. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The dataset contains **569 instances and 30 numerical features**, satisfying the assignment requirement of at least 500 instances and 12 features.

Target mapping:
- `0` = malignant
- `1` = benign

## 3. GitHub Repository Link

<https://github.com/NavyaYadlapalli99/ML_Assignment.git>

## 4. Models Used and Evaluation

The assignment document lists five models:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest Ensemble

The same stratified held-out test set is used to evaluate all models.

### Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |\n|
Decision Tree | 0.9035 | 0.9373 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |\n|
KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |\n|
Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |\n|
Random Forest | 0.9561 | 0.9944 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |\n
### Observations

| ML Model | Observation |
|---|---|
| Logistic Regression | Strong linear baseline with balanced performance and excellent AUC; feature scaling helps optimization. |\n| 
Decision Tree | Easy to interpret and captures nonlinear relationships, but a single tree is less stable than an ensemble. |\n| 
KNN | Distance-based model that benefits from scaling; it gives strong recall on this test split. |\n| 
Naive Bayes | Fast probabilistic baseline; its feature-independence assumption can limit performance when features are correlated. |\n| 
Random Forest | An ensemble of decision trees that is robust to nonlinear patterns and gives strong overall performance. |\n
### Overall Winner

Based on the **F1 score** on this held-out test set, the overall winner is **Logistic Regression**. The complete metric table should be considered when discussing model performance.

## 5. Project Structure

```text
ML_Assignment_2/
│── app.py
│── train_models.py
│── requirements.txt
│── README.md
│── test_data.csv
│── metrics.csv
│── metadata.json
│── SUBMISSION_REPORT.md
│── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── random_forest.joblib
```

## 7. Streamlit Features

The application provides:
- CSV test-data upload
- Model selection dropdown
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Confusion matrix
- Classification report
- Prediction table with class-1 probabilities

