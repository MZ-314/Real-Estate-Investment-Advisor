# =============================================================================
# 03 - ML MODELS WITH MLFLOW TRACKING
# Real Estate Investment Advisor Project
# =============================================================================

import os
import mlflow
import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# =============================================================================
# CONFIG
# =============================================================================

MLFLOW_EXPERIMENT = "real_estate_investment_experiment"

OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Correct processed file name
PROCESSED_DATA = "processed_real_estate_data.csv"


# =============================================================================
# LOAD & ENCODE DATA
# =============================================================================

def load_dataset():
    df = pd.read_csv(PROCESSED_DATA)

    # Validate target columns
    if "Good_Investment" not in df.columns:
        raise ValueError("❌ Dataset missing Good_Investment target")
    if "Future_Price_5Y" not in df.columns:
        raise ValueError("❌ Dataset missing Future_Price_5Y target")

    # Target variables
    y_clf = df["Good_Investment"]
    y_reg = df["Future_Price_5Y"]

    # Drop targets from feature matrix
    X = df.drop(["Good_Investment", "Future_Price_5Y"], axis=1)

    # Convert ALL categoricals to numeric
    X = pd.get_dummies(X, drop_first=True)

    # Ensure everything is numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    return X, y_clf, y_reg


# =============================================================================
# CLASSIFICATION TRAINING
# =============================================================================

def train_classification(X, y, experiment=MLFLOW_EXPERIMENT):

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="classification_parent_run"):

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Models to test
        models = {
            "logistic_regression": LogisticRegression(max_iter=300),
            "random_forest": RandomForestClassifier(n_estimators=150, random_state=42)
        }

        best_model = None
        best_name = None
        best_f1 = -1
        best_scaler = None

        for name, model in models.items():

            with mlflow.start_run(run_name=f"classifier_{name}", nested=True):

                mlflow.log_param("classifier_name", name)

                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                probs = model.predict_proba(X_test_scaled)[:, 1]

                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, zero_division=0)
                rec = recall_score(y_test, preds, zero_division=0)
                f1 = f1_score(y_test, preds, zero_division=0)
                roc = roc_auc_score(y_test, probs)

                # Log metrics
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc)

                print(f"{name.upper()} → acc={acc:.4f} | f1={f1:.4f} | roc_auc={roc:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_name = name
                    best_scaler = scaler

        # Save best model
        joblib.dump(best_model, f"{OUTPUT_DIR}/best_classifier_{best_name}.pkl")
        joblib.dump(best_scaler, f"{OUTPUT_DIR}/scaler_classifier_{best_name}.pkl")

        print(f"\n🏆 Best Classification Model: {best_name.upper()} (F1 = {best_f1:.4f})")

        return best_model, best_scaler


# =============================================================================
# REGRESSION TRAINING
# =============================================================================

def train_regression(X, y, experiment=MLFLOW_EXPERIMENT):

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="regression_parent_run"):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(
                n_estimators=150, random_state=42
            )
        }

        best_model = None
        best_name = None
        best_rmse = 1e18
        best_scaler = None

        for name, model in models.items():

            with mlflow.start_run(run_name=f"regressor_{name}", nested=True):

                mlflow.log_param("regressor_name", name)

                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)

                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                print(f"{name.upper()} → rmse={rmse:.4f} | mae={mae:.4f} | r2={r2:.4f}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_name = name
                    best_scaler = scaler

        joblib.dump(best_model, f"{OUTPUT_DIR}/best_regressor_{best_name}.pkl")
        joblib.dump(best_scaler, f"{OUTPUT_DIR}/scaler_regressor_{best_name}.pkl")

        print(f"\n🏆 Best Regression Model: {best_name.upper()} (RMSE = {best_rmse:.4f})")

        return best_model, best_scaler


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n==============================================")
    print(" TRAINING ML MODELS WITH MLFLOW")
    print("==============================================")

    X, y_clf, y_reg = load_dataset()

    print("\n--- Training CLASSIFICATION Models ---")
    train_classification(X, y_clf)

    print("\n--- Training REGRESSION Models ---")
    train_regression(X, y_reg)

    print("\n🎉 Training Completed Successfully!")
