# Assignment 2 Submission Report

## Student Details
- Name: ______________________________
- Student ID: _________________________
- Course: M.Tech (AIML/DSE)
- Subject: Machine Learning
- Assignment: Assignment - 2

## 1. GitHub Repository
<PASTE-GITHUB-LINK>

## 2. Live Streamlit Application
<PASTE-STREAMLIT-LINK>

## 3. Dataset

**Breast Cancer Wisconsin (Diagnostic)** — UCI Machine Learning Repository.

- Instances: 569
- Features: 30
- Classification: Binary
- Target: 0 = malignant, 1 = benign

## 4. Model Comparison

| Model               |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.9825 | 0.9954 |      0.9861 |   0.9861 | 0.9861 | 0.9623 |
| Decision Tree       |     0.9035 | 0.9373 |      0.9420 |   0.9028 | 0.9220 | 0.7969 |
| KNN                 |     0.9737 | 0.9884 |      0.9600 |   1.0000 | 0.9796 | 0.9442 |
| Naive Bayes         |     0.9386 | 0.9878 |      0.9452 |   0.9583 | 0.9517 | 0.8676 |
| Random Forest       |     0.9561 | 0.9944 |      0.9589 |   0.9722 | 0.9655 | 0.9054 |

## 5. Observations

- **Logistic Regression:** Strong linear baseline with balanced performance and excellent AUC; feature scaling helps optimization.
- **Decision Tree:** Easy to interpret and captures nonlinear relationships, but a single tree is less stable than an ensemble.
- **KNN:** Distance-based model that benefits from scaling; it gives strong recall on this test split.
- **Naive Bayes:** Fast probabilistic baseline; its feature-independence assumption can limit performance when features are correlated.
- **Random Forest:** An ensemble of decision trees that is robust to nonlinear patterns and gives strong overall performance.

**Overall winner by F1 score:** Logistic Regression

## 6. BITS Virtual Lab Screenshot

Insert the required screenshot here.

## 7. README Content

Include the complete README.md content in the final PDF as required by the assignment.
