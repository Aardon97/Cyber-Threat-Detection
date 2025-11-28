from utils import db_connect
engine = db_connect()

import os
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# -------------------------------
# Load the trained model
# -------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rf_model.pkl")
rf_model = joblib.load(MODEL_PATH)

# Debug: show the classes
st.write("Model classes:", rf_model.classes_)

# Map class indices to attack names (only 3 classes in your model)
class_labels = {
    0: "Benign",
    1: "DDoS",
    2: "PortScan"
}

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("Cyber Threat Detection App")
st.write("This app predicts whether network traffic is malicious or benign using a trained Random Forest model.")

# Option 1: Upload CSV
st.subheader("Upload a CSV file")
uploaded_file = st.file_uploader("Upload a CSV file with features", type=["csv"])
threshold_csv = st.slider("Select probability threshold for CSV predictions", 0.0, 1.0, 0.5, 0.05)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", data.head())

    # Get predictions and probabilities
    probabilities = rf_model.predict_proba(data)
    predictions = rf_model.predict(data)

    # Apply threshold logic row by row
    adjusted_labels = []
    for i, row in enumerate(probabilities):
        max_idx = row.argmax()
        max_prob = row[max_idx]
        if max_prob >= threshold_csv:
            adjusted_labels.append(class_labels[max_idx])
        else:
            adjusted_labels.append("Uncertain")

    # Build a clean results table with rounding
    results_df = pd.DataFrame(probabilities, columns=[class_labels[c] for c in rf_model.classes_])
    results_df = results_df.round(3)
    results_df.insert(0, "Predicted Class", adjusted_labels)

    # Highlight only numeric columns
    styled_df = results_df.style.highlight_max(
        subset=[class_labels[c] for c in rf_model.classes_], axis=1, color="lightgreen"
    )

    st.write("Prediction results:", styled_df)
    st.caption("Note: Green highlight = most likely class per row. Predictions below threshold are marked 'Uncertain'.")

# Option 2: Manual Input
st.subheader("Manual Input")
st.write("Enter feature values manually for a single prediction:")

feature_names = ['Source_PortScan', 'Flow Packets/s_log', 'Flow Duration_log', 'Source_Wednesday', 'Flow Bytes/s_log']
inputs = {}

for feat in feature_names:
    if "Source" in feat:
        inputs[feat] = st.number_input(f"{feat}", value=0, step=1)
    else:
        inputs[feat] = st.number_input(f"{feat}", value=0.0)

if st.button("Predict from manual input"):
    input_df = pd.DataFrame([inputs], columns=feature_names)
    prediction = rf_model.predict(input_df)[0]
    proba = rf_model.predict_proba(input_df)[0]

    st.write("Prediction (class label):", class_labels[prediction])

    proba_df = pd.DataFrame({
        "Class": [class_labels[c] for c in rf_model.classes_],
        "Probability": [round(p, 3) for p in proba]
    })
    st.write("Prediction probabilities:", proba_df)

    fig, ax = plt.subplots()
    ax.bar([class_labels[c] for c in rf_model.classes_], proba)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probability Distribution")
    st.pyplot(fig)

# Option 2b: Threshold Tuning (Manual Input)
st.subheader("Threshold Tuning (Manual Input)")
threshold = st.slider("Select probability threshold", 0.0, 1.0, 0.5, 0.05)

if st.button("Predict with threshold"):
    input_df = pd.DataFrame([inputs], columns=feature_names)
    proba = rf_model.predict_proba(input_df)[0]
    max_idx = proba.argmax()
    max_class = class_labels[max_idx]
    max_prob = proba[max_idx]

    if max_prob >= threshold:
        st.write(f"Prediction: {max_class} (probability {round(max_prob,3)})")
    else:
        st.write(f"Prediction: Below threshold ({round(max_prob,3)}). Classified as 'Uncertain'.")

    proba_df = pd.DataFrame({
        "Class": [class_labels[c] for c in rf_model.classes_],
        "Probability": [round(p, 3) for p in proba]
    })
    st.write("Prediction probabilities:", proba_df)

    fig, ax = plt.subplots()
    ax.bar([class_labels[c] for c in rf_model.classes_], proba)
    ax.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probability Distribution with Threshold")
    ax.legend()
    st.pyplot(fig)

# Option 3: Confusion Matrix + Metrics
st.subheader("Confusion Matrix (Test Set)")

import numpy as np

# Load test data
X_test = pd.read_csv("data/X_test.csv")   # replace with your actual test set file
y_test = pd.read_csv("data/y_test.csv")   # replace with your actual labels file

# Clean invalid values before prediction
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(0)

# Run predictions
y_pred = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[class_labels[c] for c in rf_model.classes_],
            yticklabels=[class_labels[c] for c in rf_model.classes_])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
st.caption("Note: Rows = actual classes, Columns = predicted classes.")

# Show performance metrics
st.write("### Model Performance Metrics")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")


