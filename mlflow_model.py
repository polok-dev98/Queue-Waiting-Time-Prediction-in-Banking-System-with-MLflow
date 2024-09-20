# # # # # import mlflow

# # # # # mlflow.set_tracking_uri("sqlite:///mlflow.db")
# # # # # from mlflow import search_runs

# # # # # # Search all runs in the experiment
# # # # # runs = search_runs(experiment_ids=['1'])

# # # # # # Display the RMSE and associated run information
# # # # # for index, row in runs.iterrows():
# # # # #     print(f"Run ID: {row['run_id']}, RMSE: {row['metrics.RMSE']}")


# # # from mlflow.tracking import MlflowClient


# # # MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
# # # client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# # # print(client.search_experiments())

# # # from mlflow.entities import ViewType

# # # runs = client.search_runs(
# # #     experiment_ids='1',
# # #     # filter_string="metrics.RMSE > 7",
# # #     run_view_type=ViewType.ACTIVE_ONLY,
# # #     max_results=5,
# # #     order_by=["metrics.RMSE ASC"]
# # # )

# # # for run in runs:
# # #     print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['RMSE']:.4f}")


# # # # # # Interacting with the Model Registry
# # # # # import mlflow

# # # # # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# # # # # run_id = "3bb02a8b0df846b196971f71ed598a8c"
# # # # # model_uri = f"runs:/{run_id}/model"
# # # # # mlflow.register_model(model_uri=model_uri, name="nyc-taxi-regressor")

# # # # # model_name = "nyc-taxi-regressor"
# # # # # latest_versions = client.get_latest_versions(name=model_name)

# # # # # for version in latest_versions:
# # # # #     print(f"version: {version.version}, stage: {version.current_stage}")


# # # # from mlflow.tracking import MlflowClient
# # # # import mlflow
# # # # import json

# # # # # Set up the MLflow tracking URI
# # # # MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
# # # # client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# # # # # Retrieve the latest runs sorted by RMSE (ascending order)
# # # # latest_runs = client.search_runs(experiment_ids=['1'], order_by=["metrics.RMSE ASC"], max_results=2)

# # # # # Retrieve the best run and previous run
# # # # best_run = latest_runs[0]
# # # # previous_run = latest_runs[1]

# # # # # Compare RMSE metrics to determine if the new model is better
# # # # if best_run.data.metrics['RMSE'] < previous_run.data.metrics['RMSE']:
# # # #     print("New model is better. Deploying...")
    
# # # #     log_model_history = best_run.data.tags['mlflow.log-model.history']
# # # #     # Convert the string to a Python list
# # # #     history_list = json.loads(log_model_history)
# # # #     # Extract the run_id
# # # #     run_id = history_list[0]['run_id']
# # # #     print(run_id)
# # # #     # run_id = "b8904012c84343b5bf8ee72aa8f0f402"
# # # #     model_uri = f"runs:/{run_id}/model"
# # # #     mlflow.register_model(model_uri=model_uri, name="nyc-taxi-regressor")

# # # # else:
# # # #     print("New model did not outperform the previous model. Keeping the current production model.")
# # # #     log_model_history = previous_run.data.tags['mlflow.log-model.history']
# # # #     # Convert the string to a Python list
# # # #     history_list = json.loads(log_model_history)
# # # #     # Extract the run_id
# # # #     run_id = history_list[0]['run_id']
# # # #     print(run_id)
# # # #     # run_id = "b8904012c84343b5bf8ee72aa8f0f402"
# # # #     model_uri = f"runs:/{run_id}/model"
# # # #     mlflow.register_model(model_uri=model_uri, name="nyc-taxi-regressor")

# # import sqlite3
# # import pandas as pd

# # # Connect to the SQLite database
# # conn = sqlite3.connect('mlflow.db')

# # # Load data from a table into a Pandas DataFrame
# # experiments_df = pd.read_sql_query("SELECT * FROM experiments", conn)
# # runs_df = pd.read_sql_query("SELECT * FROM runs", conn)

# # # Display the DataFrame
# # print(experiments_df)
# # print("\n\n")
# # print(runs_df)

# # # Close the connection
# # conn.close()


# from mlflow.tracking import MlflowClient
# import mlflow
# from mlflow.entities import ViewType

# # Set up the MLflow tracking URI
# MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# # Search for runs
# runs = client.search_runs(
#     experiment_ids=['1'],  # Changed to list
#     run_view_type=ViewType.ACTIVE_ONLY,
#     max_results=2,
#     order_by=["metrics.RMSE ASC"]
# )

# # Check if we have enough runs
# if len(runs) < 2:
#     print("Not enough runs found for comparison.")
#     run_id = runs[0].info.run_id
#     model_name = "QPro-model-regressor"
#     # Construct the model URI
#     model_uri = f"runs:/{run_id}/model"
#     model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
#     print(f"Successfully registered model '{model_name}' from run {run_id}.")
    
#     # Update the model stage to 'Production'
#     client.transition_model_version_stage(
#         name=model_name,
#         version=model_version.version,
#         stage='Production'
#     )
#     print(f"Model version {model_version.version} moved to Production stage.")

# else:
#     # Print run details
#     for run in runs:
#         print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['RMSE']:.4f}")

#     # Retrieve the best run and previous run
#     best_run = runs[0]
#     previous_run = runs[1]

#     # Compare RMSE metrics to determine if the new model is better
#     if best_run.data.metrics['RMSE'] < previous_run.data.metrics['RMSE']:
#         print("New model is better. Deploying...")
#         run_id = best_run.info.run_id
#     else:
#         print("New model did not outperform the previous model. Keeping the current production model.")
#         run_id = previous_run.info.run_id

#     # Construct the model URI
#     model_uri = f"runs:/{run_id}/model"

#     # Register the model
#     try:
#         model_name = "QPro-model-regressor"
#         model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
#         print(f"Successfully registered model '{model_name}' from run {run_id}.")
        
#         # Update the model stage to 'Production'
#         client.transition_model_version_stage(
#             name=model_name,
#             version=model_version.version,
#             stage='Production'
#         )
#         print(f"Model version {model_version.version} moved to Production stage.")

#         # Move all other models to 'Staging'
#         all_versions = client.get_latest_versions(name=model_name)
#         print("\n================",all_versions)
#         if len(all_versions) > 1:
#             for version in all_versions:
#                 if version.version != model_version.version:
#                     client.transition_model_version_stage(
#                         name=model_name,
#                         version=version.version,
#                         stage='Staging'
#                     )
#                     print(f"Model version {version.version} moved to Staging stage.")
#         else:
#             print("No other model versions available to move to Staging.")

#     except Exception as e:
#         print(f"Error: {e}")


from mlflow.tracking import MlflowClient
import mlflow
# Set up the MLflow tracking URI
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
models = client.search_registered_models()
for model in models:
    print(f"Model name: {model.name}, latest version: {model.latest_versions[-1].version}")

