import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# -----------------------------------
# TITLE
# -----------------------------------
st.title("🤖 Simple AutoML App")

# -----------------------------------
# FILE UPLOAD
# -----------------------------------
file = st.file_uploader("Upload CSV File", type=["csv"])

# -----------------------------------
# IF FILE UPLOADED
# -----------------------------------
if file:

    # Read dataset
    data = pd.read_csv(file)

    # Show dataset
    st.subheader("📊 Dataset")
    st.write(data.head())

    # Select target column
    target = st.selectbox(
        "Select Target Column",
        data.columns
    )

    # Run button
    if st.button("Run AutoML"):

        # -----------------------------------
        # INPUT AND OUTPUT
        # -----------------------------------
        X = data.drop(columns=[target])
        y = data[target]

        # Convert text columns to numbers
        X = pd.get_dummies(X)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        # -----------------------------------
        # MODELS
        # -----------------------------------
        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Logistic Regression": LogisticRegression()
        }

        best_accuracy = 0
        best_model_name = ""

        # -----------------------------------
        # TRAIN & TEST MODELS
        # -----------------------------------
        for name, model in models.items():

            # Train model
            model.fit(X_train, y_train)

            # Prediction
            y_pred = model.predict(X_test)

            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)

            st.write(f"{name} Accuracy: {accuracy:.2f}")

            # Find best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name

        # -----------------------------------
        # SHOW BEST MODEL
        # -----------------------------------
        st.success(f"🏆 Best Model: {best_model_name}")
        st.success(f"🎯 Accuracy: {best_accuracy:.2f}")