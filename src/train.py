"""
train.py

SageMaker training script for Airbnb price prediction.
Uses numeric features only and log_price as target.
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # SageMaker standard paths
    data_dir = "/opt/ml/input/data/train"
    model_dir = "/opt/ml/model"

    print("Training data directory contents:", os.listdir(data_dir))

    # Load dataset
    df = pd.read_csv(os.path.join(data_dir, "airbnb.csv"), low_memory=False)

    print("Columns:", df.columns.tolist())

    # Correct numeric features present in the dataset
    feature_cols = [
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "review_scores_rating"
    ]

    target_col = "log_price"

    # Select features and target
    X = df[feature_cols]
    y = df[target_col]

    # Handle missing values (IMPORTANT)
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))


if __name__ == "__main__":
    main()
