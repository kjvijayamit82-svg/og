
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# -------------------------------
# Utility Function 
# (Moved outside the main logic & fixed indentation)
# -------------------------------
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

# -------------------------------
# 1. Load Dataset
# -------------------------------
file_path = input("Enter CSV file path: ")
df = pd.read_csv(file_path)

print("\nDataset Preview:")
print(df.head())

# -------------------------------
# 2. Select Target Column
# -------------------------------
target = input("\nEnter target column name: ")

X = df.drop(columns=[target])
y = df[target]

# Convert categorical → numeric
X = pd.get_dummies(X)

# -------------------------------
# 3. Detect Problem Type
# -------------------------------
if y.nunique() <= 10:
    problem = "Classification"
else:
    problem = "Regression"

print(f"\nDetected Problem Type: {problem}")

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Model Training & Comparison
# -------------------------------
if problem == "Classification":
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier()
    }

    best_score = 0
    best_model = None
    best_preds = None

    print("\nModel Performance:")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.2f}")

        if acc > best_score:
            best_score = acc
            best_model = name
            best_preds = preds

    print(f"\nBest Model: {best_model} (Accuracy: {best_score:.2f})")

    # Confusion Matrix
    cm = confusion_matrix(y_test, best_preds)
    print("\nConfusion Matrix:")
    print(cm)

else:
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor()
    }

    best_score = float("inf")
    best_model = None

    print("\nModel Performance:")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        print(f"{name} MSE: {mse:.2f}")

        if mse < best_score:
            best_score = mse
            best_model = name

    print(f"\nBest Model: {best_model} (Lowest MSE: {best_score:.2f})")