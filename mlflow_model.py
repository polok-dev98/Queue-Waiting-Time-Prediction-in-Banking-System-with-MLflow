from mlflow.tracking import MlflowClient
import mlflow
# Set up the MLflow tracking URI
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
models = client.search_registered_models()
for model in models:
    print(f"Model name: {model.name}, latest version: {model.latest_versions[-1].version}")

