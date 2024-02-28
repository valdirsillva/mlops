import os 
import json 
import mlflow 
import uvicorn 
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI(title="Fetal Health API",
              openapi_tags=[
                  { "name":"Health", "description":"Get api health" },
                  { "name":"Prediction", "description":"Model prediction" }
              ])

# Função p/ carregamento do Modelo
def load_model():
    print('reading model...')
    
    MLFLOW_TRACKING_URI = 'https://dagshub.com/valdirsillva/mlops-ead.mlflow'
    MLFLOW_TRACKING_USERNAME = 'valdirsillva'
    MLFLOW_TRACKING_PASSWORD = 'c0183e495142fcfc23532726e3c91712f4bcaeef'
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    print('setting  mlflow....')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print('creating client...')
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    print('getting registered model...')
    registered_model = client.get_registered_model('fetah_health')
    print('read model...')
    run_id = registered_model.latest_versions[-1].run_id
    logged_model = f'runs:/{run_id}/model'

    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(loaded_model)
    return loaded_model


@app.get(path='/', tags=['Health'])
def api_health():
    return { "status":"healthy" }

@app.post(path='/predict', tags=['Prediction'])
def predict():
    loaded_model = load_model()
    
    accelerations = 0
    fetal_movement = 0
    uterine_contractions = 0
    severe_decelerations = 0

    received_data = np.array([
        accelerations,
        fetal_movement,
        uterine_contractions,
        severe_decelerations,
    ]).reshape(1, -1)

    print(loaded_model.predict(received_data))
    return { "prediction": 0 }


