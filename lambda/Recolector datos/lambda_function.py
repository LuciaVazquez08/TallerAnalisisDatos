import requests
import pandas as pd
import os
import json
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
import boto3 


BUCKET_NAME = 'proyecto-clima-datos'
ARCHIVO_MAESTRO = 'estaciones_zonas.csv'

def descargar_una_obs(s_id):
    """Descarga la observación más reciente de una estación meteorológica.
 
    Realiza hasta 2 intentos ante fallos de red o respuestas no exitosas.
    En el primer intento fallido espera 0,5 s antes de reintentar.
    Si la estación devuelve 404 se abandona inmediatamente (no existe o
    no tiene datos recientes).
 
    Args:
        s_id (str): Identificador de la estación según la API de la NOAA
            (p. ej. ``"KNYC"``).
 
    Returns:
        tuple[str, dict | None]: Par ``(station_id, datos_json)`` donde
        ``datos_json`` es el objeto JSON de la respuesta o ``None`` si
        no fue posible obtener datos.
 
    Example:
        >>> station_id, obs = descargar_una_obs("KNYC")
        >>> if obs:
        ...     print(obs["properties"]["temperature"])
    """
    url_obs = f"https://api.weather.gov/stations/{s_id}/observations/latest"
    for intento in range(2):
        try:
            res = requests.get(url_obs, timeout=10)
            if res.status_code == 200:
                return s_id, res.json()
            if res.status_code == 404:
                break
        except:
            if intento == 0: time.sleep(0.5)
    return s_id, None

def descargar_un_pronostico(args):

    s_id, url_fcst = args
    for intento in range(2):
        try:
            res = requests.get(url_fcst, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if 'properties' in data and 'periods' in data['properties']:
                    data['properties']['periods'] = data['properties']['periods'][:1]
                    return s_id, data
                return s_id, data 
            if res.status_code == 404:
                break
        except:
            if intento == 0: time.sleep(0.5)
    return s_id, None


def lambda_handler(event, context):
    """Punto de entrada de la función AWS Lambda.
 
    Orquesta la descarga de observaciones y pronósticos, y los persiste
    en S3 con la siguiente estructura de claves::
 
        observations/{YYYY-MM-DD}/batch_observations_{YYYYMMDDHHhs}.json
        forecasts/{YYYY-MM-DD}/batch_forecasts_{YYYYMMDDHHhs}.json
 
    Args:
        event (dict): Payload del evento que disparó la función (puede
            ser un evento de EventBridge/CloudWatch, un test manual, etc.).
            No se utiliza internamente, pero es requerido por el contrato
            de Lambda.
        context: Objeto de contexto de Lambda que provee metadata de
            ejecución (tiempo restante, nombre de la función, etc.).
            No se utiliza internamente.
 
    Returns:
        dict: Resultado de la ejecución con las claves:
 
            - ``status`` (str): ``"success"`` o ``"error"``.
            - ``duration_sec`` (float): Tiempo total de ejecución en
              segundos (solo en éxito).
            - ``obs_count`` (int): Número de estaciones con observación
              descargada (solo en éxito).
            - ``zones_count`` (int): Número de zonas con pronóstico
              descargado (solo en éxito).
            - ``message`` (str): Descripción del error (solo en error).
 
    """
    start_time = time.time()
    s3 = boto3.client('s3')
    
    ahora = datetime.now()
    fecha_hoy = ahora.strftime("%Y-%m-%d")
    timestamp_id = ahora.strftime("%Y%m%d_%Hhs")

    try:
        df = pd.read_csv(ARCHIVO_MAESTRO)
    except Exception as e:
        return {"status": "error", "message": f"No se encontró el CSV: {str(e)}"}

    lote_observaciones = {}
    lote_pronosticos = {}

    # Descarga paralela de observaciones
    with ThreadPoolExecutor(max_workers=20) as executor:
        res_obs = list(executor.map(descargar_una_obs, df['station_id']))

    for s_id, data in res_obs:
        if data:
            lote_observaciones[s_id] = data

    # Descarga paralela de pronósticos 
    tareas_fcst = list(zip(df['station_id'], df['forecast_hourly_url']))
    with ThreadPoolExecutor(max_workers=20) as executor:
        res_fcst = list(executor.map(descargar_un_pronostico, tareas_fcst))

    for s_id, data in res_fcst:
        if data:
            lote_pronosticos[s_id] = data

    try:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=f"observations/{fecha_hoy}/batch_observations_{timestamp_id}.json",
            Body=json.dumps(lote_observaciones)
        )
        
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=f"forecasts/{fecha_hoy}/batch_forecasts_{timestamp_id}.json",
            Body=json.dumps(lote_pronosticos)
        )
    except Exception as e:
        return {"status": "error", "message": f"Error al subir a S3: {str(e)}"}

    duracion = time.time() - start_time
    
    return {
        "status": "success",
        "duration_sec": round(duracion, 2),
        "obs_count": len(lote_observaciones),
        "forecasts_count": len(lote_pronosticos)
    }