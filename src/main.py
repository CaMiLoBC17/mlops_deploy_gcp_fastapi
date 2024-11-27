from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import FileResponse
# Librerías que necesita mi Pipeline de Machine Learning
import numpy as np
import pandas as pd
import joblib
import os
import subprocess
from configurations import config
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer
)
from input.preprocessors import LogTransformation, Scaler

app = FastAPI()

def prediccion(pipeline_ml, data):
    data = data.drop(['Date', 'Close'], axis=1)
    data = data[config.FEATURES]

    pred = pipeline_ml.predict(data)

    original_scale_pred = np.exp(pred)

    return pred, original_scale_pred, data

ruta_actual = os.getcwd()

@app.get("/")
def home():
    return "La API está en línea"

@app.get("/ver_ruta_actual")
def ver_ruta_actual():
    return {"message":f"La ruta actual es {ruta_actual}"}

@app.get("/ruta_actual1")
def ruta_actual_1():
    ruta = subprocess.check_output(["pwd"], text = True).strip("")
    archivos = subprocess.check_output(["ls"], text = True)
    return {"Ruta":f"{ruta.splitlines()}", "Archivos":f"{archivos.splitlines()}"}

@app.get("/ruta_actual2")
def ruta_actual_2():
    archivos = subprocess.check_output(["ls", "src"], text = True).strip("")
    return {"Archivos":f"{archivos.splitlines()}"}

@app.post("/recibir_data_&_predecir")
def publicar(file: UploadFile = File(...)):
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")
    
    try:
        # Guardar el archivo CSV subido temporalmente
        file_location = f"{ruta_actual}/{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(file.file.read())

        # Leer el archivo CSV
        df_de_los_datos_subidos = pd.read_csv(file_location)

        # Cargar el pipeline de producción desde la ruta correcta
        # NOTA: Asegúrate de que esta ruta es donde estás copiando el archivo en el Dockerfile
        ruta_modelo = os.path.join(ruta_actual, "src/full_ml_pipeline.joblib")  # Actualización de la ruta
        pipeline_de_produccion = joblib.load(ruta_modelo)

        # Hacer la predicción
        predicciones, predicciones_sin_escalar, datos_test_procesados = prediccion(pipeline_de_produccion, df_de_los_datos_subidos)

        # Concatenar los datos procesados y las predicciones
        df_concatenado = pd.concat([datos_test_procesados, pd.Series(predicciones, name="Predicciones"), pd.Series(predicciones_sin_escalar, name="Predicciones_Sin_Escalar")], axis=1)

        # Guardar el archivo de salida
        output_file = f"{ruta_actual}/salida_datos_y_predicciones.csv"
        df_concatenado.to_csv(output_file, index=False)

        # Devolver el archivo resultante
        return FileResponse(output_file, media_type="application/octet-stream", filename="salida_datos_y_predicciones.csv")
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío o tiene un formato incorrecto.")
    except joblib.externals.loky.process_executor.TerminatedWorkerError:
        raise HTTPException(status_code=500, detail="Error al cargar el modelo de predicción.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")