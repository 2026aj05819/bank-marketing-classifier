"""Term deposit prediction - Streamlit app for ML Assignment 2.

Loads the five trained pipelines from model/ and evaluates them on a test CSV.
The models are only used for prediction here; all training is done in
bank_marketing_models.ipynb.
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(APP_DIR, "model")
TEST_FILE = os.path.join(APP_DIR, "test_data.csv")

st.set_page_config(page_title="Term Deposit Predictor", layout="centered")


@st.cache_resource
def load_pipelines():
    with open(os.path.join(MODEL_FOLDER, "columns.json")) as f:
        meta = json.load(f)
    pipelines = {}
    for model_name, file_name in meta["model_files"].items():
        pipelines[model_name] = joblib.load(os.path.join(MODEL_FOLDER, file_name))
    return meta, pipelines


def yes_score(model, X):
    # the SVM was saved without probability output, so its AUC uses decision_function
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(model, X, y_true, y_pred):
    if y_true.nunique() == 2:
        auc = roc_auc_score(y_true, yes_score(model, X))
    else:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


meta, pipelines = load_pipelines()
feature_cols = meta["features"]
target_col = meta["target"]

st.title("Term Deposit Predictor")
st.markdown(
    "A Portuguese bank called its clients to sell a term deposit. Five classifiers "
    "were trained on the UCI **Bank Marketing** data to predict whether a client "
    "subscribes (`y` = yes / no). Load test data below to see how each saved model "
    "performs on it."
)

# ---- 1. test data ----
st.header("1. Test data")
source = st.radio(
    "Where should the test data come from?",
    ["Upload a CSV file", "Use test_data.csv from this repository"],
    horizontal=True,
)
with open(TEST_FILE, "rb") as f:
    st.download_button("Download test_data.csv", f.read(),
                       file_name="test_data.csv", mime="text/csv")

if source == "Upload a CSV file":
    uploaded_file = st.file_uploader(
        f"CSV with the {len(feature_cols)} input columns and the target column `{target_col}`",
        type="csv")
    if uploaded_file is None:
        st.info("Upload a file, or pick the test_data.csv option above.")
        st.stop()
    # sep=None lets pandas detect a comma or a semicolon
    test_df = pd.read_csv(uploaded_file, sep=None, engine="python")
else:
    test_df = pd.read_csv(TEST_FILE)

missing_cols = [c for c in feature_cols + [target_col] if c not in test_df.columns]
if missing_cols:
    st.error("The file is missing these columns: " + ", ".join(missing_cols))
    st.stop()

answer = test_df[target_col].astype(str).str.strip().str.lower()
y_true = answer.map({"yes": 1, "no": 0, "1": 1, "0": 0})
if y_true.isna().any():
    st.error(f"Column `{target_col}` must contain only yes/no (or 1/0). "
             f"{int(y_true.isna().sum())} rows have something else.")
    st.stop()
y_true = y_true.astype(int)
X_test = test_df[feature_cols]

st.success(f"{len(test_df)} clients loaded. {int(y_true.sum())} of them subscribed "
           f"({y_true.mean():.1%}).")
with st.expander("Preview the first 20 rows"):
    st.dataframe(test_df.head(20))

# ---- 2. model ----
st.header("2. Choose a model")
chosen = st.selectbox("Model", list(pipelines))
model = pipelines[chosen]
y_pred = model.predict(X_test)
scores = compute_metrics(model, X_test, y_true, y_pred)

# ---- 3. metrics ----
st.header("3. Evaluation metrics")
boxes = st.columns(3) + st.columns(3)
for box, (metric_name, value) in zip(boxes, scores.items()):
    box.metric(metric_name, f"{value:.4f}")

# ---- 4. confusion matrix and report ----
st.header("4. Confusion matrix and classification report")
col_cm, col_report = st.columns(2)
with col_cm:
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1], display_labels=["no", "yes"],
        cmap="Greens", colorbar=False, ax=ax)
    ax.set_title(chosen)
    st.pyplot(fig)
    plt.close(fig)
with col_report:
    report = classification_report(y_true, y_pred, labels=[0, 1],
                                   target_names=["no", "yes"], digits=4,
                                   zero_division=0)
    st.text(report)

# ---- 5. all models ----
st.header("5. All five models on this data")
all_rows = {name: compute_metrics(m, X_test, y_true, m.predict(X_test))
            for name, m in pipelines.items()}
comparison = pd.DataFrame(all_rows).T.round(4)
comparison.index.name = "ML Model Name"
st.dataframe(comparison)
st.bar_chart(comparison[["F1", "MCC"]])
st.write(f"Best F1 on this data: **{comparison['F1'].idxmax()}**, "
         f"best MCC: **{comparison['MCC'].idxmax()}**.")

st.caption("The models are loaded from the model/ folder and only used to predict. "
           "Nothing is retrained on the uploaded data.")
