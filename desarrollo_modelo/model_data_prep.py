import pandas as pd
import numpy as np
import os

def prepare_data(input_path="../local_cleanup/clima_unificado.parquet", output_path="clima_features.parquet"):
    # Si el script se corre desde la raíz, ajustamos el path
    if not os.path.exists(input_path) and os.path.exists("local_cleanup/clima_unificado.parquet"):
        input_path = "local_cleanup/clima_unificado.parquet"

    if not os.path.exists(input_path):
        print(f"Error: No se encuentra {input_path}")
        return

    df = pd.read_parquet(input_path)
    print(f"Cargados {len(df)} registros.")

    # 1. Temporal Features (Ciclo Diurno)
    df['obs_timestamp'] = pd.to_datetime(df['obs_timestamp'])
    df['hour'] = df['obs_timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # 2. Boolean Features
    df['is_daytime_fcst'] = df['is_daytime_fcst'].astype(int)

    # 3. Diferenciales Térmicos y de Humedad (per station)
    df = df.sort_values(['station_id', 'obs_timestamp'])
    df['temp_fcst_diff'] = df.groupby('station_id')['temp_fcst'].diff().fillna(0)
    df['hum_fcst_diff'] = df.groupby('station_id')['hum_fcst'].diff().fillna(0)

    # --- NUEVAS MEJORAS (Feature Engineering Avanzado) ---
    # 3.1 Lagged Observations (Persistencia del error reciente)
    # El error de la hora anterior es un predictor potente de la inercia del sesgo.
    df['temp_error_lag1'] = (df['temp_fcst'] - df['temp_real']).groupby(df['station_id']).shift(1)
    df['hum_error_lag1'] = (df['hum_fcst'] - df['hum_real']).groupby(df['station_id']).shift(1)
    
    # 3.2 Ventanas Móviles (Rolling stats del pronóstico)
    # Captura la tendencia de las últimas 3 horas
    df['temp_fcst_roll3_mean'] = df.groupby('station_id')['temp_fcst'].transform(lambda x: x.rolling(3).mean())
    df['temp_fcst_roll3_std'] = df.groupby('station_id')['temp_fcst'].transform(lambda x: x.rolling(3).std())

    # 4. Limpieza de NaNs en targets y features esenciales
    target_cols = ['temp_real', 'hum_real']
    df = df.dropna(subset=target_cols)
    
    # Features que usaremos (Lista extendida)
    feature_cols = [
        'temp_fcst', 'hum_fcst', 
        'hour_sin', 'hour_cos', 
        'is_daytime_fcst',
        'lat_estacion', 'lon_estacion',
        'temp_fcst_diff', 'hum_fcst_diff',
        'temp_error_lag1', 'hum_error_lag1',
        'temp_fcst_roll3_mean', 'temp_fcst_roll3_std'
    ]
    
    # Rellenar NaNs en features (ffill/bfill por estación es más seguro)
    for col in feature_cols:
        df[col] = df.groupby('station_id')[col].transform(lambda x: x.ffill().bfill())
    
    # Manejar el caso donde una estación entera no tenga datos previos para el lag1
    df[feature_cols] = df[feature_cols].fillna(0)

    print(f"Dataset final con features: {len(df)} registros.")
    df.to_parquet(output_path, index=False)
    print(f"Guardado en {output_path}")
    
    return df

if __name__ == "__main__":
    prepare_data()
