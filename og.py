import streamlit as st
import pandas as pd

# Title
st.title("🤖 Algorithm Suggestion Bot")

# Upload
file = st.file_uploader("📂 Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head())

    # Target selection
    target = st.selectbox("🎯 Select Target Column", df.columns)

    if st.button("🤖 Suggest Algorithm"):

        y = df[target]

        # Logic
        unique_values = y.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(y)

        st.write("### 🧠 Bot Analysis")

        if is_numeric and unique_values > 10:
            st.success("🔹 Problem Type: Regression")

            st.write("### ✅ Suggested Algorithms:")
            st.write("- Linear Regression")
            st.write("- Random Forest Regressor")
            st.write("- Gradient Boosting")

        elif unique_values <= 10:
            st.success("🔹 Problem Type: Classification")

            st.write("### ✅ Suggested Algorithms:")
            st.write("- Logistic Regression")
            st.write("- Decision Tree")
            st.write("- Random Forest")
            st.write("- SVM")

        else:
            st.success("🔹 Problem Type: Clustering")

            st.write("### ✅ Suggested Algorithms:")
            st.write("- K-Means")
            st.write("- DBSCAN")

        # Extra insights
        st.write("### 📊 Dataset Info")
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])
        st.write("Unique Target Values:", unique_values)