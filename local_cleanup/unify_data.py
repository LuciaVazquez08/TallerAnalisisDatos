import boto3
import pandas as pd
import json
import os
from datetime import datetime
from tqdm import tqdm

### IMPORTANTE:
# Antes de usar configurar las credenciales de AWS (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) en aws cli
# ejecutar desde root/local_cleanup

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


def clean_wind_speed(val):
    """
    Limpia el string de velocidad del viento de la NWS y lo convierte de mph a km/h.
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


def process_forecasts(forecast_json, df_maestro):
    """Aplana el JSON de pronósticos a un DataFrame con variables extendidas.
    Soporta tanto el nuevo formato (key = station_id) como el viejo (key = zona_id).
    """
    data = []
    # Set de estaciones válidas para identificar el tipo de key
    valid_stations = set(df_maestro["station_id"].unique())

    for key, fcst in forecast_json.items():
        if fcst and "properties" in fcst:
            periods = fcst["properties"].get("periods", [])
            for p in periods:
                raw_temp = p.get("temperature")
                unit = p.get("temperatureUnit", "F")

                # Conversión de Fahrenheit a Celsius si es necesario
                if unit == "F" and raw_temp is not None:
                    temp_c = (raw_temp - 32) * 5 / 9
                else:
                    temp_c = raw_temp

                entry_base = {
                    "fcst_start": pd.to_datetime(p.get("startTime"), utc=True),
                    "fcst_end": pd.to_datetime(p.get("endTime"), utc=True),
                    "is_daytime_fcst": p.get("isDaytime"),
                    "temp_fcst": temp_c,
                    "dew_fcst": (
                        p.get("dewpoint", {}).get("value")
                        if isinstance(p.get("dewpoint"), dict)
                        else None
                    ),
                    "hum_fcst": (
                        p.get("relativeHumidity", {}).get("value")
                        if isinstance(p.get("relativeHumidity"), dict)
                        else p.get("relativeHumidity")
                    ),
                    "wind_speed_fcst": clean_wind_speed(p.get("windSpeed")),
                    "wind_dir_fcst": p.get("windDirection"),
                    "precip_prob_fcst": p.get("probabilityOfPrecipitation", {}).get(
                        "value"
                    ),
                    "short_fcst": p.get("shortForecast"),
                }

                if key in valid_stations:
                    # Nuevo formato: El pronóstico es específico para esta estación
                    entry = entry_base.copy()
                    entry["station_id"] = key
                    data.append(entry)
                elif key.startswith("ZONA_"):
                    # Viejo formato: Mapear este pronóstico de zona a todas sus estaciones
                    stations_in_zone = df_maestro[df_maestro["zona_id"] == key][
                        "station_id"
                    ].tolist()
                    for s_id in stations_in_zone:
                        entry = entry_base.copy()
                        entry["station_id"] = s_id
                        data.append(entry)
                else:
                    # Formato desconocido, ignorar o loggear
                    pass

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
        df_maestro[["station_id", "zona_id", "lat_estacion", "lon_estacion", "estado"]],
        on="station_id",
        how="left",
    )

    # 2. Cargar todos los Pronósticos
    print("\n--- Cargando Pronósticos ---")
    all_fcst = []
    for f in tqdm(fcst_files):
        try:
            raw = read_json_from_s3(f)
            # Pasamos df_maestro para mapear zonas a estaciones si es necesario
            all_fcst.append(process_forecasts(raw, df_maestro))
        except Exception as e:
            print(f"Error en {f}: {e}")

    df_fcst_total = pd.concat(all_fcst)

    # Asegurar que para cada estación y cada inicio de periodo solo tenemos la versión más reciente
    # Como fcst_files viene ordenado por nombre (timestamp), el último es el más reciente
    df_fcst_total = df_fcst_total.drop_duplicates(
        subset=["station_id", "fcst_start"], keep="last"
    )

    # 3. Unificación Temporal (Merge Asof)
    # Para cada observación, buscamos el pronóstico que empezó ANTES o IGUAL al tiempo de obs
    print("\n--- Realizando alineación temporal ---")

    # Ordenar es requisito para merge_asof
    df_obs_total = df_obs_total.sort_values("obs_timestamp")
    df_fcst_total = df_fcst_total.sort_values("fcst_start")

    # El merge_asof une por tiempo cercano dentro de cada station_id
    df_final = pd.merge_asof(
        df_obs_total,
        df_fcst_total,
        left_on="obs_timestamp",
        right_on="fcst_start",
        by="station_id",
        direction="backward",  # Busca el periodo vigente
    )

    # 4. Limpieza final y guardado
    # Filtramos para asegurar que la observación realmente caiga dentro del periodo del pronóstico
    df_final = df_final[df_final["obs_timestamp"] < df_final["fcst_end"]]

    # --- FILTRO DE OUTLIERS TÉCNICOS (GLITCH DE PROVEEDOR) ---
    # Se detectó un error masivo en el API de la NWS (National Weather Service)
    # afectando el ciclo de pronóstico del 2026-04-23 20:00 UTC en el sureste de Wisconsin.
    # El API devolvió valores de temperatura de hasta 60°C (140°F) y humedad del 8%,
    # lo cual es físicamente imposible para la región y época.
    # Estos valores "basura" distorsionan el análisis estadístico y los modelos de ML.
    # Descartamos cualquier pronóstico de temperatura > 50°C.
    anomalous_count = (df_final["temp_fcst"] > 50).sum()
    if anomalous_count > 0:
        print(
            f"Limpieza: Eliminando {anomalous_count} registros con errores técnicos de la NWS (>50°C)."
        )
        df_final = df_final[df_final["temp_fcst"] <= 50]
    # ----------------------------------------------------------

    print(f"\nUnificación completada. Filas finales: {len(df_final)}")

    df_final.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Dataset guardado en: {OUTPUT_PARQUET}")


if __name__ == "__main__":
    unify()
