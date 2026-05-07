import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns

st.title("Simple ML Dashboard")


file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    # Remove missing values (important!)
    df = df.dropna()

    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X)

    st.write("Classes in target:", y.nunique())

    if y.nunique() <= 10:
        problem = "Classification"
    else:
        problem = "Regression"

    st.write("Problem Type:", problem)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem == "Classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier()
        }

        best_score = -1
        best_preds = None
        best_name = ""

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            st.write(name, "Accuracy:", round(acc, 2))

            # FIX: ensures best_preds is always set
            if best_preds is None or acc > best_score:
                best_score = acc
                best_preds = preds
                best_name = name

        st.write("Best Model:", best_name)

        # Confusion Matrix (will NOT crash now)
        cm = confusion_matrix(y_test, best_preds)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
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
            st.write(name, "MSE:", round(mse, 2))

            if mse < best_score:
                best_score = mse
                best_name = name

        st.write("Best Model:", best_name)