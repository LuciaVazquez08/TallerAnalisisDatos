import boto3
import pandas as pd
import json
import io
import os
from datetime import datetime, timedelta, timezone

# Configuración
BUCKET_NAME = "proyecto-clima-datos"
s3 = boto3.client("s3")

def load_maestro():
    """Carga el CSV que mapea estaciones a zonas desde S3."""
    obj = s3.get_object(Bucket=BUCKET_NAME, Key="maestro/estaciones_zonas.csv")
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def list_s3_files_recent(prefix, hours=26):
    """Lista de solo archivos modificados en el tiempo definido."""
    files = []
    threshold = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                if obj["Key"].endswith(".json") and obj["LastModified"] >= threshold:
                    files.append(obj["Key"])
    return sorted(files)

def read_json_from_s3(key):
    """Lee un archivo JSON desde S3 y lo devuelve como diccionario."""
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def clean_wind_speed(val):
    """
    Limpieza del string de velocidad del viento de la NWS y lo convierte de mph a km/h.
    La NWS usa mph, mientras que Open-Meteo (observaciones) usa km/h.
    """
    if val is None:
        return None

    # Si es un rango (ej: '5 to 10 mph'), tomamos el valor superior o el promedio?
    # El notebook original solo quitaba ' mph', así que intentamos extraer el primer número
    # o el valor simple si no es rango.
    try:
        if isinstance(val, str):
            s = val.replace(" mph", "").strip()
            if " to " in s:
                # Caso rango: promediamos para ser más representativos
                parts = s.split(" to ")
                val_num = (float(parts[0]) + float(parts[1])) / 2
            else:
                val_num = float(s)
        else:
            val_num = float(val)

        # Conversión mph -> km/h (1 mph = 1.60934 km/h)
        return val_num * 1.60934
    except (ValueError, TypeError):
        return None

def process_observations(obs_json):
    """Aplana el JSON de observaciones a un DataFrame con variables extendidas."""
    data = []
    for s_id, obs in obs_json.items():
        if obs and "properties" in obs:
            p = obs["properties"]
            data.append({
                "station_id": s_id,
                "obs_timestamp": pd.to_datetime(p.get("timestamp"), utc=True),
                "temp_real": p.get("temperature", {}).get("value"),
                "dew_real": p.get("dewpoint", {}).get("value"),
                "hum_real": p.get("relativeHumidity", {}).get("value"),
                "wind_speed_real": p.get("windSpeed", {}).get("value"),
                "wind_gust_real": p.get("windGust", {}).get("value"),
                "wind_dir_real": p.get("windDirection", {}).get("value"),
                "press_real": p.get("barometricPressure", {}).get("value"),
                "sea_level_press_real": p.get("seaLevelPressure", {}).get("value"),
                "precip_1h_real": p.get("precipitationLastHour", {}).get("value"),
                "visibility_real": p.get("visibility", {}).get("value"),
            })
    return pd.DataFrame(data)

def process_forecasts(forecast_json, df_maestro):
    """Aplana el JSON de pronósticos a un DataFrame con variables extendidas.
    Soporta tanto el nuevo formato (key = station_id) como el viejo (key = zona_id).
    """
    data = []
    valid_stations = set(df_maestro["station_id"].unique())
    for key, fcst in forecast_json.items():
        if fcst and "properties" in fcst:
            periods = fcst["properties"].get("periods", [])
            for p in periods:
                raw_temp = p.get("temperature")
                unit = p.get("temperatureUnit", "F")
                
                # Conversión de Fahrenheit a Celsius si es necesario
                temp_c = (raw_temp - 32) * 5 / 9 if unit == "F" and raw_temp is not None else raw_temp
                
                entry_base = {
                    "fcst_start": pd.to_datetime(p.get("startTime"), utc=True),
                    "fcst_end": pd.to_datetime(p.get("endTime"), utc=True),
                    "is_daytime_fcst": p.get("isDaytime"),
                    "temp_fcst": temp_c,
                    "dew_fcst": p.get("dewpoint", {}).get("value") if isinstance(p.get("dewpoint"), dict) else None,
                    "hum_fcst": p.get("relativeHumidity", {}).get("value") if isinstance(p.get("relativeHumidity"), dict) else p.get("relativeHumidity"),
                    "wind_speed_fcst": clean_wind_speed(p.get("windSpeed")),
                    "wind_dir_fcst": p.get("windDirection"),
                    "precip_prob_fcst": p.get("probabilityOfPrecipitation", {}).get("value"),
                    "short_fcst": p.get("shortForecast"),
                }
                
                if key in valid_stations:
                    entry = entry_base.copy()
                    entry["station_id"] = key
                    data.append(entry)
                elif key.startswith("ZONA_"):
                    stations_in_zone = df_maestro[df_maestro["zona_id"] == key]["station_id"].tolist()
                    for s_id in stations_in_zone:
                        entry = entry_base.copy()
                        entry["station_id"] = s_id
                        data.append(entry)
    return pd.DataFrame(data)

def bronze_to_silver():
    df_maestro = load_maestro()
    
    # Carga dataset historico si existe
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key="silver/clima_unificado.parquet")
        df_silver_old = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception:
        df_silver_old = pd.DataFrame()

    # --- Cargando Observaciones ---
    print("--- Cargando Observaciones ---")
    all_obs = []
    for f in list_s3_files_recent("observations/", hours=26):
        try:
            all_obs.append(process_observations(read_json_from_s3(f)))
        except Exception as e:
            print(f"Error en {f}: {e}")

    df_obs = pd.concat(all_obs).drop_duplicates()
    df_obs = pd.merge(
        df_obs,
        df_maestro[["station_id", "zona_id", "lat_estacion", "lon_estacion", "estado"]],
        on="station_id", how="left",
    )

    # --- Cargando Pronósticos ---
    print("--- Cargando Pronósticos ---")
    all_fcst = []
    for f in list_s3_files_recent("forecasts/", hours=26):
        try:
            all_fcst.append(process_forecasts(read_json_from_s3(f), df_maestro))
        except Exception as e:
            print(f"Error en {f}: {e}")

    df_fcst = pd.concat(all_fcst)
    df_fcst = df_fcst.drop_duplicates(subset=["station_id", "fcst_start"], keep="last")

    # --- Merge temporal ---
    print("--- Merge temporal ---")
    df_obs = df_obs.sort_values("obs_timestamp")
    df_fcst = df_fcst.sort_values("fcst_start")

    df_silver = pd.merge_asof(
        df_obs, df_fcst,
        left_on="obs_timestamp",
        right_on="fcst_start",
        by="station_id",
        direction="backward",
    )

    df_silver = df_silver[df_silver["obs_timestamp"] < df_silver["fcst_end"]]

    # --- FILTRO DE OUTLIERS TÉCNICOS (GLITCH DE PROVEEDOR) ---
    # Se detectó un error masivo en el API de la NWS (National Weather Service)
    # afectando el ciclo de pronóstico del 2026-04-23 20:00 UTC en el sureste de Wisconsin.
    # El API devolvió valores de temperatura de hasta 60°C (140°F) y humedad del 8%,
    # lo cual es físicamente imposible para la región y época.
    # Estos valores "basura" distorsionan el análisis estadístico y los modelos de ML.
    # Descartamos cualquier pronóstico de temperatura > 50°C.
    anomalous_count = (df_silver["temp_fcst"] > 50).sum()
    if anomalous_count > 0:
        print(
            f"Limpieza: Eliminando {anomalous_count} registros con errores técnicos de la NWS (>50°C)."
        )
        df_silver = df_silver[df_silver["temp_fcst"] <= 50]
    # ----------------------------------------------------------

    # Concatena con historico y elimina duplicados por clave natural
    print("--- Actualizando Dataset Silver ---")
    df_final = pd.concat([df_silver_old, df_silver]).drop_duplicates(
        subset=["station_id", "obs_timestamp"], keep="last"
    )

    # Guarda archivo unico en silver
    buffer = io.BytesIO()
    df_final.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key="silver/clima_unificado.parquet", Body=buffer)
    print(f"Dataset guardado con exito. Filas totales: {len(df_final)}")

if __name__ == "__main__":
    bronze_to_silver()