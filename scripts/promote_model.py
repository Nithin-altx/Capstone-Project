import os
import mlflow
from mlflow.tracking import MlflowClient

def promote_model():

    # DagsHub credentials
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "nithinemmadishetti25"
    repo_name = "Capstone-Project"

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    client = MlflowClient()
    model_name = "my_model"

    # Get all model versions
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise Exception("No model versions found")

    # Get latest version
    latest_version = max(int(v.version) for v in versions)

    # Assign alias "champion" (production)
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=latest_version
    )

    print(f"Model version {latest_version} promoted to alias 'champion'")

if __name__ == "__main__":
    promote_model()