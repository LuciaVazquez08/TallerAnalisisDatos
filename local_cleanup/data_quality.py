import pandas as pd
import numpy as np
import os

# Configuración
INPUT_PARQUET = "clima_unificado.parquet"

def load_data():
    if not os.path.exists(INPUT_PARQUET):
        print(f"Error: No se encuentra el archivo {INPUT_PARQUET}. Ejecuta unify_data.py primero.")
        return None
    return pd.read_parquet(INPUT_PARQUET)

def run_quality_analysis(df):
    total_rows = len(df)
    print(f"--- Análisis de Calidad de Datos (Total Filas: {total_rows}) ---\n")

    # 1. INTEGRIDAD (Completitud)
    print("1. INTEGRIDAD (Valores No Nulos)")
    integrity = df.notnull().sum() / total_rows * 100
    print(integrity.round(2).to_string())
    print("-" * 30)

    # 2. UNICIDAD
    print("\n2. UNICIDAD")
    duplicates = df.duplicated(subset=['station_id', 'obs_timestamp']).sum()
    uniqueness = (1 - (duplicates / total_rows)) * 100
    print(f"Filas duplicadas (estación + tiempo): {duplicates}")
    print(f"Índice de Unicidad: {uniqueness:.2f}%")
    print("-" * 30)

    # 3. VALIDEZ (Reglas de Negocio)
    print("\n3. VALIDEZ (Fuera de Rango)")
    rules = {
        "temp_real": (-60, 60),      # Celsius
        "hum_real": (0, 100),       # Porcentaje
        "press_real": (80000, 110000), # Pascales (NOAA usa Pa)
        "wind_speed_real": (0, 250)  # km/h o m/s según unidad
    }
    
    for col, (min_val, max_val) in rules.items():
        if col in df.columns:
            invalid = df[(df[col] < min_val) | (df[col] > max_val)]
            pct_invalid = (len(invalid) / total_rows) * 100
            print(f"{col}: {len(invalid)} valores inválidos ({pct_invalid:.2f}%)")
    print("-" * 30)

    # 4. OPORTUNIDAD (Timeliness)
    print("\n4. OPORTUNIDAD")
    min_date = df['obs_timestamp'].min()
    max_date = df['obs_timestamp'].max()
    span = max_date - min_date
    print(f"Rango de datos: desde {min_date} hasta {max_date}")
    print(f"Ventana temporal total: {span}")
    print("-" * 30)

    # 5. PRECISIÓN Y CONSISTENCIA
    print("\n5. PRECISIÓN Y CONSISTENCIA (Real vs Pronóstico)")
    # Calculamos el error absoluto medio como métrica de consistencia
    if 'temp_real' in df.columns and 'temp_fcst' in df.columns:
        # Nota: NOAA forecast suele estar en Fahrenheit, real en Celsius. 
        # IMPORTANTE: Validar conversión antes de este paso en producción.
        # Por ahora solo comparamos si hay datos en ambos.
        aligned = df.dropna(subset=['temp_real', 'temp_fcst'])
        print(f"Filas con ambos datos (Real y Forecast): {len(aligned)} ({len(aligned)/total_rows*100:.2f}%)")
    print("-" * 30)

    # 6. IDONEIDAD
    print("\n6. IDONEIDAD")
    # ¿Están todas las estaciones representadas?
    unique_stations = df['station_id'].nunique()
    print(f"Estaciones únicas en el dataset: {unique_stations}")
    print(f"Zonas únicas cubiertas: {df['zona_id'].nunique()}")

if __name__ == "__main__":
    data = load_data()
    if data is not None:
        run_quality_analysis(data)
