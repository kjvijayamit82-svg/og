import streamlit as st
import pandas as pd

# Simple suggestion function
def suggest_model(df, target):
    y = df[target]

    # Detect task
    if y.dtype == "object":
        task = "Classification"
        models = ["Logistic Regression", "KNN", "Decision Tree"]
    else:
        task = "Regression"
        models = ["Linear Regression", "Decision Tree", "Random Forest"]

    return task, models


# UI
st.title("🤖 Simple ML Model Suggester")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Suggest Model"):
        task, models = suggest_model(df, target)

        st.write(f"### Task: {task}")

        st.write("### Suggested Models:")
        for m in models:
            st.write(f"- {m}")