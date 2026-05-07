import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Title
st.title("🤖 Smart Algorithm Selector + Evaluation")
st.write("HELLO WELCOME TO SMART ALGORITHM SUGGEST")
# Upload file
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write("### 📄 Dataset Preview")
    st.dataframe(df.head())

    # Select target column
    target = st.selectbox("🎯 Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical features
    X = pd.get_dummies(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("🚀 Train Model"):

        # Detect problem type
        if (
            y.dtype == "object"
            or y.dtype.name == "category"
            or y.nunique() <= 10
        ):
            task = "classification"
        else:
            task = "regression"

        st.info(f"🔍 Detected Problem Type: {task}")

        # ================= CLASSIFICATION =================
        if task == "classification":

            y_train = y_train.astype(str)
            y_test = y_test.astype(str)

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # Accuracy
            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Accuracy: {acc:.2f}")

            # Confusion Matrix (Table)
            cm = confusion_matrix(y_test, y_pred)
            st.write("### 📊 Confusion Matrix (Table)")
            st.dataframe(cm)
            
            # Confusion Matrix (Chart)
            st.write("### 📊 Confusion Matrix (Chart)")
            fig, ax = plt.subplots()
            ax.imshow(cm)

            # Add labels
            labels = sorted(y.astype(str).unique())
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

            # Add numbers inside cells
            for i in range(len(cm)):
                for j in range(len(cm[0])):
                    ax.text(j, i, cm[i, j], ha="center", va="center")

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            st.pyplot(fig)

            # Classification Report
            st.write("### 📄 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

        # ================= REGRESSION =================
        else:
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            st.success(f"📈 R² Score: {r2:.2f}")
            st.warning(f"📉 Mean Squared Error: {mse:.2f}")