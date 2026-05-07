import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


st.title("🤖 Smart Algorithm Selector + Evaluation")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical features
    X = pd.get_dummies(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("🚀 Train Model"):

        # ✅ Robust task detection
        if (
            y.dtype == "object"
            or y.dtype.name == "category"
            or y.nunique() <= 10
        ):
            task = "classification"
        else:
            task = "regression"

        st.write("### 🔍 Detected Problem Type:", task)

        # ================= CLASSIFICATION =================
        if task == "classification":
            
            # 🚨 FIX: Force the target to be strings so sklearn treats floats as discrete categories
            y_train = y_train.astype(str)
            y_test = y_test.astype(str)

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            st.write("### ✅ Accuracy:", acc)
            st.write("### 📊 Confusion Matrix")
            st.write(cm)

        # ================= REGRESSION =================
        else:
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            st.write("### 📈 R² Score:", r2)
            st.write("### 📉 MSE:", mse)