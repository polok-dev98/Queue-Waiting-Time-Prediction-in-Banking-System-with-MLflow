# from mlflow.tracking import MlflowClient
# import mlflow
# import json
# from mlflow.entities import ViewType

# # Set up the MLflow tracking URI
# MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# runs = client.search_runs(
#     experiment_ids='1',
#     # filter_string="metrics.RMSE > 7",
#     run_view_type=ViewType.ACTIVE_ONLY,
#     max_results=2,
#     order_by=["metrics.RMSE ASC"]
# )

# for run in runs:
#     print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['RMSE']:.4f}")

# # # Retrieve the best run and previous run
# best_run = runs[0]
# previous_run = runs[1]

# # Compare RMSE metrics to determine if the new model is better
# if best_run.data.metrics['RMSE'] < previous_run.data.metrics['RMSE']:
#     print("New model is better. Deploying...")
#     run_id = best_run.info.run_id
#     print(run_id)
#     model_uri = f"runs:/{run_id}/model"
#     mlflow.register_model(model_uri=model_uri, name="QPro-model-regressor")

# else:
#     print("New model did not outperform the previous model. Keeping the current production model.")
#     run_id = previous_run.info.run_id
#     print(run_id)
#     model_uri = f"runs:/{run_id}/model"
#     mlflow.register_model(model_uri=model_uri, name="QPro-model-regressor")


# model_name = "QPro-model-regressor"
# latest_versions = client.get_latest_versions(name=model_name)

# for version in latest_versions:
#     print(f"version: {version.version}, stage: {version.current_stage}")



from mlflow.tracking import MlflowClient
import mlflow
from mlflow.entities import ViewType

# Set up the MLflow tracking URI
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Search for runs
runs = client.search_runs(
    experiment_ids='1',
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=2,
    order_by=["metrics.RMSE ASC"]
)


# Check if we have enough runs
if len(runs) < 2:
    # print(runs)
    print("Not enough runs found for comparison.")
    run_id = runs[0].info.run_id
    model_name = "QPro-model-regressor"
    # Construct the model URI
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Successfully registered model '{model_name}' from run {run_id}.")
    
    # Update the model stage to 'Production'
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage='Production'
    )
    print(f"Model version {model_version.version} moved to Production stage.")

else:
    # Print run details
    for run in runs:
        print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['RMSE']:.4f}")

    # Retrieve the best run and previous run
    best_run = runs[0]
    previous_run = runs[1]

    # Compare RMSE metrics to determine if the new model is better
    if best_run.data.metrics['RMSE'] < previous_run.data.metrics['RMSE']:
        print("New model is better. Deploying...")
        run_id = best_run.info.run_id
    else:
        print("New model did not outperform the previous model. Keeping the current production model.")
        run_id = previous_run.info.run_id

    # Construct the model URI
    model_uri = f"runs:/{run_id}/model"

    # Register the model
    try:
        model_name = "QPro-model-regressor"
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"Successfully registered model '{model_name}' from run {run_id}.")
        
        # Update the model stage to 'Production'
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='Production'
        )
        print(f"Model version {model_version.version} moved to Production stage.")
    except Exception as e:
        print(f"Error: {e}")
