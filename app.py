import streamlit as st
import pandas as pd
import numpy as np

# --- 🧠 ALGORITHM SUGGESTION ENGINE ---
def suggest_algorithms(df, target_col):
    y = df[target_col]
    n_samples, n_features = df.shape
    
    # 1. Determine Task Type (Using our robust fix from earlier!)
    is_classification = False
    if y.dtype == 'object' or y.dtype.name == 'category':
        is_classification = True
    elif pd.api.types.is_numeric_dtype(y) and y.nunique() <= 20:
        # If it's numeric but has 20 or fewer unique values, assume it's classification (like ratings 1-5, or 0/1)
        is_classification = True
        
    task = "Classification" if is_classification else "Regression"

    # 2. Analyze Size & Dimensionality
    suggestions = []
    reasoning = []

    # High Dimensionality Check (Curse of Dimensionality)
    if n_features > n_samples:
        if is_classification:
            suggestions = ["Linear SVC", "Naive Bayes"]
        else:
            suggestions = ["Lasso Regression", "Ridge Regression"]
        reasoning.append("Your dataset has more features (columns) than samples (rows). Penalized linear models work best to prevent overfitting.")
        return task, suggestions, reasoning

    # Size-based Routing
    if n_samples < 10000:
        # Small Dataset
        reasoning.append(f"Small dataset ({n_samples} rows). Simpler models are recommended to avoid overfitting.")
        if is_classification:
            suggestions = ["Logistic Regression", "Support Vector Machine (RBF Kernel)", "Random Forest (Shallow)"]
        else:
            suggestions = ["Ridge Regression", "Support Vector Regression (SVR)", "Random Forest Regressor"]
            
    elif n_samples < 100000:
        # Medium Dataset
        reasoning.append(f"Medium dataset ({n_samples} rows). Tree-based ensemble models usually provide the best accuracy out-of-the-box.")
        if is_classification:
            suggestions = ["Random Forest Classifier", "XGBoost Classifier", "LightGBM"]
        else:
            suggestions = ["Random Forest Regressor", "XGBoost Regressor", "LightGBM"]
            
    else:
        # Large Dataset
        reasoning.append(f"Large dataset ({n_samples} rows). Highly scalable models or Neural Networks are required for performance.")
        if is_classification:
            suggestions = ["LightGBM Classifier", "Stochastic Gradient Descent (SGD)", "Neural Networks (PyTorch/TensorFlow)"]
        else:
            suggestions = ["LightGBM Regressor", "SGD Regressor", "Neural Networks (PyTorch/TensorFlow)"]

    return task, suggestions, reasoning

# --- 🎨 STREAMLIT UI ---
st.set_page_config(page_title="AI Model Suggester", page_icon="🤖")

st.title("🤖 Intelligent Algorithm Suggester")
st.markdown("Upload a CSV dataset, pick your target variable, and let the app suggest the best machine learning algorithms for your specific data.")

# File Uploader
file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)
    
    st.subheader("1. Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]:,} rows and {df.shape[1]} columns")
    st.dataframe(df.head(3))
    
    st.subheader("2. Target Selection")
    target = st.selectbox("Which column are you trying to predict?", df.columns)
    
    if st.button("🔍 Analyze & Suggest"):
        with st.spinner("Analyzing data characteristics..."):
            
            # Call our engine
            task_type, recommended_models, reasons = suggest_algorithms(df, target)
            
            # Display Results
            st.divider()
            st.subheader(f"🎯 Detected Task: {task_type}")
            
            # Show Reasoning
            for reason in reasons:
                st.info(f"**Why?** {reason}")
            
            # Show Models
            st.write("### 🏆 Top Recommended Algorithms:")
            for i, model in enumerate(recommended_models):
                if i == 0:
                    st.success(f"**🥇 1st Choice:** {model}")
                elif i == 1:
                    st.warning(f"**🥈 2nd Choice:** {model}")
                else:
                    st.info(f"**🥉 3rd Choice:** {model}")