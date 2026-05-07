import streamlit as st
import pandas as pd

# 1. Define the logic function first
def suggest_model(df, target_col):
    y = df[target_col]
    
    # Check if the target is categorical or numerical
    if y.dtype == "object" or y.nunique() < 10:  # Adding nunique as a safety check
        task = "Classification"
        models = ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Decision Tree Classifier"]
    else:
        task = "Regression"
        models = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
     
    return task, models

# 2. UI Layout
st.set_page_config(page_title="ML Suggester", page_icon="🤖")
st.title("🤖 Simple ML Model Suggester")

file = st.file_uploader("Upload CSV file", type=['csv'])

if file is not None:
    # Use latin-1 encoding as a fallback if utf-8 fails
    try:
        df = pd.read_csv(file)
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding='latin-1')

    # Data Preview Section
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Shape Info
    rows, cols = df.shape
    st.info(f"**Dataset Dimensions:** {rows} Rows and {cols} Columns")

    # Model Suggestion Logic
    target = st.selectbox("Select Target Column (Y)", df.columns)

    if st.button("Suggest Model"):
        task, models = suggest_model(df, target)

        st.divider()
        st.subheader(f"Detected Task: {task}")
        
        st.write("### Recommended Models:")
        for m in models:
            st.success(m)