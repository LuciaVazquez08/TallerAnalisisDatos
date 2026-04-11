from datetime import datetime
import boto3
import requests
import json

s3 = boto3.client("s3")

BUCKET = "datos-clima-prueba"


def nombre_archivo(base):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    return f"{base}_{timestamp}.json"


def guardar_en_s3(data, key):

    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(data)
    )


def consumir_y_guardar(url, archivo_base):

    response = requests.get(url)

    data = response.json()

    key = nombre_archivo(archivo_base)

    guardar_en_s3(data, key)


def obtener_endpoints_coordenadas(lat, lon):

    url = f"https://api.weather.gov/points/{lat},{lon}"

    response = requests.get(url)

    data = response.json()

    props = data["properties"]

    return {
        "stations": props["observationStations"],
        "forecast": props["forecast"],
        "forecast_hourly": props["forecastHourly"]
    }


def obtener_pronosticos(lat, lon):

    url = f"https://api.weather.gov/points/{lat},{lon}"

    response = requests.get(url)

    data = response.json()

    props = data["properties"]

    return {
        "forecast_periods": props["forecast"],
        "forecast_hourly": props["forecastHourly"]
    }


def obtener_datos_estacion(station_id):

    base_url = (
        f"https://api.weather.gov/stations/{station_id}"
    )

    return {
        "current": f"{base_url}/observations/latest",
        "history": f"{base_url}/observations"
    }


def lambda_handler(event, context):

    lat = 47.60
    lon = -122.33

    station_id = "KSEA"

    endpoints = obtener_endpoints_coordenadas(lat, lon)

    consumir_y_guardar(
        endpoints["stations"],
        "stations"
    )

    pronosticos = obtener_pronosticos(lat, lon)

    consumir_y_guardar(
        pronosticos["forecast_periods"],
        "forecast_periods"
    )

    consumir_y_guardar(
        pronosticos["forecast_hourly"],
        "forecast_hourly"
    )

    estacion = obtener_datos_estacion(
        station_id
    )

    consumir_y_guardar(
        estacion["current"],
        "station_current"
    )

    consumir_y_guardar(
        estacion["history"],
        "station_history"
    )

    return {"statusCode": 200}