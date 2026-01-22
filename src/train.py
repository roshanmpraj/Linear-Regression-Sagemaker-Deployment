"""
train.py

This script trains a Linear Regression model using scikit-learn.
It is executed inside an AWS SageMaker Training Job.

Input:
- Training data from /opt/ml/input/data/train

Output:
- Trained model saved to /opt/ml/model
"""

import argparse
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # SageMaker passes these paths automatically
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(os.path.join(args.data_path, "airbnb.csv"))

    # SAME features as your notebook
    features = [
        "accommodates",
        "bedrooms",
        "bathrooms",
        "number_of_reviews"
    ]

    X = df[features]
    y = df["price"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

    # Save trained model (SageMaker uploads this to S3 automatically)
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "linear_regression_model.joblib"))


if __name__ == "__main__":
    main()
