
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

TARGET = "target"

st.set_page_config(
    page_title="ML Classification Model Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ML Classification Model Explorer")
st.write(
    "Compare five classification models on the UCI Breast Cancer Wisconsin "
    "(Diagnostic) test dataset."
)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_default_test_data():
    return pd.read_csv(os.path.join(BASE_DIR, "test_data.csv"))

# Sidebar
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader(
    "Upload test data (CSV)",
    type=["csv"],
    help="Use the supplied test_data.csv. It must contain the target column."
)

model_name = st.sidebar.selectbox("Select classification model", list(MODEL_FILES.keys()))

if uploaded is not None:
    df = pd.read_csv(uploaded)
    source_label = "Uploaded CSV"
else:
    df = load_default_test_data()
    source_label = "Bundled test_data.csv"

if TARGET not in df.columns:
    st.error("The uploaded CSV must contain a 'target' column for evaluation.")
    st.stop()

feature_cols = [c for c in df.columns if c != TARGET]
model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = load_model(model_path)

# Validate features
expected_features = list(getattr(model, "feature_names_in_", []))
if not expected_features:
    # Pipelines expose feature_names_in_ when fitted with a DataFrame.
    expected_features = feature_cols

missing = [c for c in expected_features if c not in df.columns]
extra = [c for c in feature_cols if c not in expected_features]

if missing:
    st.error("Missing required feature columns: " + ", ".join(missing))
    st.stop()

X = df[expected_features]
y = df[TARGET]

if not pd.api.types.is_numeric_dtype(y):
    st.error("The target column must contain numeric class labels 0 and 1.")
    st.stop()

y = y.astype(int)

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

accuracy = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_prob)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
mcc = matthews_corrcoef(y, y_pred)

st.caption(f"Data source: {source_label} | Rows evaluated: {len(df)}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{accuracy:.4f}")
c2.metric("AUC", f"{auc:.4f}")
c3.metric("Precision", f"{precision:.4f}")
c4.metric("Recall", f"{recall:.4f}")
c5.metric("F1 Score", f"{f1:.4f}")
c6.metric("MCC", f"{mcc:.4f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )
    st.dataframe(cm_df, use_container_width=True)

with right:
    st.subheader("Classification Report")
    report = classification_report(
        y, y_pred,
        target_names=["Malignant (0)", "Benign (1)"],
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

st.subheader("Predictions")
display_df = df.copy()
display_df["Predicted Class"] = y_pred
display_df["Prediction Probability (Class 1)"] = np.round(y_prob, 4)
st.dataframe(display_df.head(50), use_container_width=True)

st.info(
    "Target mapping: 0 = malignant, 1 = benign. "
    "For evaluation, keep the target column in the test CSV."
)
