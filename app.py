
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

# Page title
st.title("Breast Cancer Classification - Model Comparison")
st.write("Upload test data and compare 5 ML classification models.")

# Model names and their saved files
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}

# Feature 1: CSV upload
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Separate features and target
    X_test = df.drop("target", axis=1)
    y_test = df["target"]

    # Load scaler and scale the data
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    X_test_scaled = scaler.transform(X_test)

    # Feature 2: Model selection dropdown
    choice = st.selectbox("Select a model", list(model_files.keys()))

    # Load the chosen model
    with open("model/" + model_files[choice], "rb") as f:
        model = pickle.load(f)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Feature 3: Show metrics
    st.subheader("Evaluation Metrics for " + choice)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }
    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    metrics_df["Value"] = metrics_df["Value"].round(4)
    st.table(metrics_df)

    # Feature 4: Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4))
else:
    st.info("Please upload the test_data.csv file to begin.")
