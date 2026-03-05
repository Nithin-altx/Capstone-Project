import os
import mlflow
from mlflow.tracking import MlflowClient

def promote_model():

    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "nithinemmadishetti25"
    repo_name = "Capstone-Project"

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    client = MlflowClient()
    model_name = "my_model"

    # get all versions
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise Exception("No model versions found")

    # get latest version
    latest_version = max(versions, key=lambda v: int(v.version)).version

    # archive current production
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])

    for v in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=v.version,
            stage="Archived"
        )

    # promote latest model
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production"
    )

    print(f"Model version {latest_version} promoted to Production")


if __name__ == "__main__":
    promote_model()