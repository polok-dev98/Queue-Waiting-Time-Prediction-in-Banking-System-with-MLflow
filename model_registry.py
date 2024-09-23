from mlflow.tracking import MlflowClient
import mlflow
from mlflow.entities import ViewType

# Set up the MLflow tracking URI
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# Step 1: Search for the current production model
model_name = "Best-model-regressor"

try:
    # Retrieve the latest production model version
    latest_versions = client.get_latest_versions(name=model_name, stages=["Production"])

    if latest_versions:
        production_model_version = latest_versions[0]
        production_run_id = production_model_version.run_id

        # Get the accuracy of the current production model
        production_run = client.get_run(production_run_id)
        production_accuracy = production_run.data.metrics.get('accuracy', 0)
        print(f"Current production model accuracy: {production_accuracy:.4f}")
    else:
        # If there's no production model, any new model will be considered
        production_accuracy = 0
        print("No production model found. Any new model will be considered for deployment.")

except Exception as e:
    print(f"Error retrieving production model: {e}")
    production_accuracy = 0  # Proceed with no production model as fallback


# Step 2: Get the best new model from the latest experiment run
try:
    runs = client.search_runs(
        experiment_ids='1',  # Replace with the appropriate experiment ID
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=2,
        order_by=["metrics.accuracy DESC"]  # Order by accuracy in descending order
    )

    if len(runs) < 1:
        print("No runs found for comparison.")
    else:
        # Get the best new run
        best_run = runs[0]
        best_run_accuracy = best_run.data.metrics.get('accuracy', 0)
        print(f"Best new model accuracy: {best_run_accuracy:.4f}")

        # Step 3: Compare the new model's accuracy with the production model's accuracy
        if best_run_accuracy > production_accuracy:
            print("New model has better accuracy. Deploying...")
            run_id = best_run.info.run_id

            try:
                # Construct the model URI
                model_uri = f"runs:/{run_id}/model"

                # Register the new model
                model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
                print(f"Successfully registered model '{model_name}' from run {run_id}.")

                # Step 4: Transition the new model to the production stage
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage='Production'
                )
                print(f"Model version {model_version.version} moved to Production stage.")

            except Exception as e:
                print(f"Error during model registration or production transition: {e}")

        else:
            print("New model did not outperform the current production model. Keeping the current production model.")

except Exception as e:
    print(f"Error searching for new model runs: {e}")
