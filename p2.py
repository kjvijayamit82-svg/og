import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns

st.title("🤖 Smart Algorithm Suggestion System")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X)

    # Detect problem
    if y.nunique() <= 10:
        problem = "Classification"
    else:
        problem = "Regression"

    st.write(f"Detected Problem: {problem}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier()
        }

        best_score = 0
        best_preds = None
        best_name = ""

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            st.write(f"{name} Accuracy: {acc:.2f}")

            if acc > best_score:
                best_score = acc
                best_name = name
                best_preds = preds

        st.success(f"Best Model: {best_name} (Accuracy: {best_score:.2f})")

        # Confusion Matrix
        cm = confusion_matrix(y_test, best_preds)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor()
        }

        best_score = float("inf")
        best_name = ""

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            st.write(f"{name} MSE: {mse:.2f}")

            if mse < best_score:
                best_score = mse
                best_name = name

        st.success(f"Best Model: {best_name} (Lowest MSE: {best_score:.2f})")