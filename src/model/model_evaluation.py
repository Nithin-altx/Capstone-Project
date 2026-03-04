import os
import json
import pickle
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import dagshub


# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "nithinemmadishetti25"
repo_name = "Capstone-Project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


# -------------------------------------------------------
# MLflow + DagsHub Configuration (Local Setup)
# -------------------------------------------------------

# MLFLOW_TRACKING_URI = "https://dagshub.com/nithinemmadishetti25/Capstone-Project.mlflow"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# dagshub.init(
#     repo_owner="nithinemmadishetti25",
#     repo_name="Capstone-Project",
#     mlflow=True
# )


# -------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------------------------------------
# Utility Functions
# -------------------------------------------------------

def load_model(file_path: str):
    """Load the trained model from a pickle file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at {file_path}")

    with open(file_path, "rb") as file:
        model = pickle.load(file)

    logging.info(f"Model loaded from {file_path}")
    return model


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")

    df = pd.read_csv(file_path)
    logging.info(f"Data loaded from {file_path}")
    return df


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model performance."""
    y_pred = clf.predict(X_test)

    # Some models may not support predict_proba
    if hasattr(clf, "predict_proba"):
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "auc": auc
    }

    logging.info("Model evaluation completed")
    return metrics


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save metrics to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(metrics, file, indent=4)

    logging.info(f"Metrics saved to {file_path}")


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save run ID and model path to JSON."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    model_info = {
        "run_id": run_id,
        "model_path": model_path
    }

    with open(file_path, "w") as file:
        json.dump(model_info, file, indent=4)

    logging.info(f"Model info saved to {file_path}")


# -------------------------------------------------------
# Main Execution
# -------------------------------------------------------

def main():
    mlflow.set_experiment("my-dvc-pipeline")

    try:
        with mlflow.start_run() as run:

            # Load model and data
            clf = load_model("./models/model.pkl")
            test_data = load_data("./data/processed/test_bow.csv")

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # Evaluate
            metrics = evaluate_model(clf, X_test, y_test)

            # Save locally
            save_metrics(metrics, "reports/metrics.json")
            save_model_info(run.info.run_id, "model", "reports/experiment_info.json")

            # Log metrics to MLflow
            for name, value in metrics.items():
                if value is not None:
                    mlflow.log_metric(name, value)

            # Log parameters if available
            if hasattr(clf, "get_params"):
                for param, value in clf.get_params().items():
                    mlflow.log_param(param, value)

            # Log artifacts
            mlflow.sklearn.log_model(
                sk_model=clf,
                # name="model",
                artifact_path="model", 
                serialization_format="pickle",
                # registered_model_name="my_model"  
            )
            from mlflow.tracking import MlflowClient

            client = MlflowClient()

# Get latest version properly (safe method)
            latest_versions = client.search_model_versions("name='my_model'")
            latest_version = max([int(mv.version) for mv in latest_versions])

# Assign alias (NEW METHOD)
            client.set_registered_model_alias(
                      name="my_model",
                      alias="staging",
                      version=str(latest_version)
                      )
            mlflow.log_artifact("reports/metrics.json")
            mlflow.log_artifact("reports/experiment_info.json")

            logging.info("MLflow run completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()