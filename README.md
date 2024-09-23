# Project Title: **Bank Customer Queue Waiting Time Prediction**

## Overview
The project aims to predict the waiting time for bank customers based on the length of the service queue. By utilizing machine learning models such as (XGBoost, RandomForest, KNN), the system helps customers estimate how long they will need to wait before receiving service, making the process more transparent and improving customer satisfaction.

The project uses **MLflow** to manage and track machine learning models, compares different models' performance based on accuracy, and deploys the best model to production. A **Flask API** is used to serve the production model for real-time waiting time predictions. </br> </br>

![22](https://github.com/user-attachments/assets/7de94864-253a-4d45-9c6c-34c06532d421)

</br>

## Project Structure

```bash
Project
│
├── app.py                   # Flask application for serving model predictions
├── model_registry.py         # MLflow registry script for managing models in production
├── mlflow_model.py           # Script to display registered models and versions
├── QPro_Test_code.ipynb      # Notebook for training and experimenting with models
├── mlflow.db                 # SQLite database for MLflow tracking
├── requirements.txt          # Project dependencies
├── .gitignore                # Git ignored files
├── LICENSE                   # License file            
└── README.md                 # Project documentation
```

## Features

#### Queue Waiting Time Prediction:
- Predicts how long a bank customer will need to wait based on the queue length.

#### Model Tracking and Experiment Management using MLflow:
- Register and track models, monitoring performance metrics like accuracy.
- Automatically deploy the best-performing model to production.

#### REST API for Predictions:
- A `/predict` endpoint allows for real-time predictions based on input data (queue length, customer service parameters, etc.).

#### MLflow Integration:
- Track metrics such as accuracy during model training.
- Register and transition models between stages (e.g., "Production").
- SQLite-based backend (`mlflow.db`) for tracking experiments and models.

#### Automated Model Deployment:
- Automatically compares newly trained models with the production model and deploys the better model.

#### Production Model API:
- A RESTful API allows clients (such as web applications or mobile apps) to request predicted waiting times.</br>

## Setup

#### 1. Clone the repository:
   ```bash
   https://github.com/polok-dev98/Queue-Waiting-Time-Prediction-in-Banking-System-with-MLflow.git
   cd Queue-Waiting-Time-Prediction-in-Banking-System-with-MLflow
   ```


#### 2. Install the required Python dependencies:

  ```bash
  pip install -r requirements.txt
  ```

#### 3. Ensure that MLflow is set up with a local SQLite database:
The database file mlflow.db is included in the repository and tracks all model runs and metrics.
</br>

## Usage

### Running the Flask Application

1. Start the Flask server to serve the production model:
   ```bash
   python app.py
   ```

The API will be available at `http://0.0.0.0:5000/`.

Send a POST request to the `/predict` endpoint to get predictions:

```bash
curl -X POST http://localhost:5006/predict -H "Content-Type: application/json" -d '{
  "data": [[5.1, 3.5, 1.4, 0.2]],  # Replace with appropriate features
  "columns": ["queue_length", "average_service_time", "other_features"]  # Replace with feature names
}'
```

## Traing the Models
- To train the ML models such as XGBoost, RandomForest, KNN etc, check the notebook `(QPro_Test_code.ipynb)`. 
- This notebook contains how to load the dataset, preprocess the data, feature enginnering steps, training and log the models artifatcts on MLFlow tracking server.</br>

## Managing Models

#### Tracking and Managing Models

To track and manage models, use the `model_registry.py` script:

```bash
python model_registry.py
```

This script compares the accuracy of newly trained models with the current production model and updates the production stage accordingly.

#### Listing Registered Models
Use `mlflow_model.py` to list all registered models:

```bash
python mlflow_model.py
```

## Requirements

The following Python libraries are required:

- `pandas`
- `scikit-learn`
- `matplotlib`
- `shap`
- `lime`
- `plotly`
- `seaborn`
- `mlflow==2.16.0`
- `pyenv-win==1.2.1`
- `xgboost`
- `flask`

You can install these dependencies by running:

```bash
pip install -r requirements.txt
```

## License
This project is licensed under the `MIT License` - see the LICENSE file for details.


