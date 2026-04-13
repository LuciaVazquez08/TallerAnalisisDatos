import boto3
import pandas as pd
import json
import os
from datetime import datetime
from tqdm import tqdm

### IMPORTANTE: configurar

# Configuración
BUCKET_NAME = "proyecto-clima-datos"  # Cambiar según corresponda
CSV_MAESTRO = "../lambda/Recolector datos/estaciones_zonas.csv"
OUTPUT_PARQUET = "clima_unificado.parquet"

s3 = boto3.client("s3")


def load_maestro():
    """Carga el CSV que mapea estaciones a zonas."""
    return pd.read_csv(CSV_MAESTRO)


def list_s3_files(prefix):
    """Lista todos los archivos JSON bajo un prefijo dado en S3."""
    files = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                if obj["Key"].endswith(".json"):
                    files.append(obj["Key"])
    return sorted(files)  # Ordenados por fecha/timestamp en el nombre


def read_json_from_s3(key):
    """Lee un archivo JSON desde S3 y lo devuelve como diccionario."""
    print(f"Leyendo: {key}")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def process_observations(obs_json):
    """Aplana el JSON de observaciones a un DataFrame con variables extendidas."""
    data = []
    for s_id, obs in obs_json.items():
        if obs and "properties" in obs:
            p = obs["properties"]
            data.append(
                {
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
                }
            )
    return pd.DataFrame(data)


def process_forecasts(forecast_json):
    """Aplana el JSON de pronósticos a un DataFrame con variables extendidas."""
    data = []
    for z_id, fcst in forecast_json.items():
        if fcst and "properties" in fcst:
            periods = fcst["properties"].get("periods", [])
            for p in periods:
                data.append(
                    {
                        "zona_id": z_id,
                        "fcst_start": pd.to_datetime(p.get("startTime"), utc=True),
                        "fcst_end": pd.to_datetime(p.get("endTime"), utc=True),
                        "is_daytime_fcst": p.get("isDaytime"),
                        "temp_fcst": p.get("temperature"),
                        "dew_fcst": p.get("dewpoint", {}).get("value")
                        if isinstance(p.get("dewpoint"), dict)
                        else None,
                        "hum_fcst": p.get("relativeHumidity", {}).get("value")
                        if isinstance(p.get("relativeHumidity"), dict)
                        else p.get("relativeHumidity"),
                        "wind_speed_fcst": p.get("windSpeed"),
                        "wind_dir_fcst": p.get("windDirection"),
                        "precip_prob_fcst": p.get("probabilityOfPrecipitation", {}).get("value"),
                        "short_fcst": p.get("shortForecast"),
                    }
                )
    return pd.DataFrame(data)



def unify():
    df_maestro = load_maestro()
    print(f"Conectando al bucket: {BUCKET_NAME}")

    obs_files = list_s3_files("observations/")
    fcst_files = list_s3_files("forecasts/")

    if not obs_files or not fcst_files:
        print("No se encontraron suficientes archivos en S3 para procesar.")
        return

    # 1. Cargar todas las Observaciones
    print("\n--- Cargando Observaciones ---")
    all_obs = []
    for f in tqdm(obs_files):
        try:
            raw = read_json_from_s3(f)
            all_obs.append(process_observations(raw))
        except Exception as e:
            print(f"Error en {f}: {e}")
    
    df_obs_total = pd.concat(all_obs).drop_duplicates()
    
    # Unir con maestro para obtener zona_id y metadata geográfica
    df_obs_total = pd.merge(
        df_obs_total, 
        df_maestro[['station_id', 'zona_id', 'lat_estacion', 'lon_estacion', 'estado']], 
        on='station_id', 
        how='left'
    )

    # 2. Cargar todos los Pronósticos
    print("\n--- Cargando Pronósticos ---")
    all_fcst = []
    for f in tqdm(fcst_files):
        try:
            raw = read_json_from_s3(f)
            all_fcst.append(process_forecasts(raw))
        except Exception as e:
            print(f"Error en {f}: {e}")
            
    df_fcst_total = pd.concat(all_fcst).drop_duplicates()

    # 3. Unificación Temporal (Merge Asof)
    # Para cada observación, buscamos el pronóstico que empezó ANTES o IGUAL al tiempo de obs
    print("\n--- Realizando alineación temporal ---")
    
    # Ordenar es requisito para merge_asof
    df_obs_total = df_obs_total.sort_values('obs_timestamp')
    df_fcst_total = df_fcst_total.sort_values('fcst_start')

    # El merge_asof une por tiempo cercano dentro de cada zona_id
    df_final = pd.merge_asof(
        df_obs_total,
        df_fcst_total,
        left_on='obs_timestamp',
        right_on='fcst_start',
        by='zona_id',
        direction='backward' # Busca el periodo vigente
    )

    # 4. Limpieza final y guardado
    # Filtramos para asegurar que la observación realmente caiga dentro del periodo del pronóstico
    df_final = df_final[df_final['obs_timestamp'] < df_final['fcst_end']]

    print(f"\nUnificación completada. Filas finales: {len(df_final)}")
    
    df_final.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Dataset guardado en: {OUTPUT_PARQUET}")


if __name__ == "__main__":
    unify()
