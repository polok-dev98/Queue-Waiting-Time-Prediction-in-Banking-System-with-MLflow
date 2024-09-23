from flask import Flask, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# Set up the MLflow tracking URI
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Get all registered models
registered_models = client.search_registered_models()

# Find the model in the "Production" stage
production_model_name = None
for registered_model in registered_models:
    for latest_version in registered_model.latest_versions:
        if latest_version.current_stage == "Production":
            production_model_name = registered_model.name
            break
    if production_model_name:
        break

print(f"The model in Production stage is: {production_model_name}")


app = Flask(__name__)

model = mlflow.pyfunc.load_model(f"models:/{production_model_name}/Production")
print(model)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Convert the JSON data to a Pandas DataFrame
        df = pd.DataFrame(data['data'], columns=data['columns'])

        # Perform prediction
        predictions = model.predict(df)
        
        # Return predictions as JSON with a custom key
        response = {"Waiting Time": predictions.tolist()}
        return jsonify(response)
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5006)


