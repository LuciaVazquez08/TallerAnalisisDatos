import pandas as pd
import numpy as np
import os

def prepare_data(input_path="../local_cleanup/clima_unificado.parquet", output_path="clima_features.parquet"):
    """
    Carga el dataset unificado y realiza la ingeniería de variables (features) 
    necesaria para el modelo de corrección de sesgo.
    
    Args:
        input_path (str): Ruta al archivo parquet unificado (obs + fcst).
        output_path (str): Ruta donde se guardará el dataset procesado con features.
    """
    # Ajuste de ruta dinámico según el contexto de ejecución
    if not os.path.exists(input_path) and os.path.exists("local_cleanup/clima_unificado.parquet"):
        input_path = "local_cleanup/clima_unificado.parquet"

    if not os.path.exists(input_path):
        print(f"Error: No se encuentra {input_path}")
        return

    df = pd.read_parquet(input_path)
    print(f"Cargados {len(df)} registros originales.")

    # 1. Transformación de Variables Temporales (Ciclo Diurno)
    # Convertimos la hora en componentes seno/coseno para capturar su naturaleza cíclica
    # (ej. que las 23:00 y las 00:00 sean tratadas como horas cercanas).
    df['obs_timestamp'] = pd.to_datetime(df['obs_timestamp'])
    df['hour'] = df['obs_timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # 2. Variables Categóricas y Booleanas
    # Convertimos is_daytime_fcst a entero para el modelo
    df['is_daytime_fcst'] = df['is_daytime_fcst'].astype(int)

    # 3. Ingeniería de Variables Basada en Series de Tiempo (por Estación)
    # Es crucial ordenar por estación y tiempo antes de calcular diferenciales y lags
    df = df.sort_values(['station_id', 'obs_timestamp'])
    
    # 3.1 Diferencial del Pronóstico: Indica la tendencia que el modelo NWS espera
    df['temp_fcst_diff'] = df.groupby('station_id')['temp_fcst'].diff().fillna(0)
    df['hum_fcst_diff'] = df.groupby('station_id')['hum_fcst'].diff().fillna(0)

    # 3.2 Lags de Error (Persistencia): Captura si el modelo viene fallando en la hora anterior.
    # Esta variable es el predictor más potente del sistema.
    df['temp_error_lag1'] = (df['temp_fcst'] - df['temp_real']).groupby(df['station_id']).shift(1)
    df['hum_error_lag1'] = (df['hum_fcst'] - df['hum_real']).groupby(df['station_id']).shift(1)
    
    # 3.3 Estadísticas Móviles (Rolling): Captura la inercia térmica de las últimas 3 horas del pronóstico.
    df['temp_fcst_roll3_mean'] = df.groupby('station_id')['temp_fcst'].transform(lambda x: x.rolling(3).mean())
    df['temp_fcst_roll3_std'] = df.groupby('station_id')['temp_fcst'].transform(lambda x: x.rolling(3).std())

    # 4. Limpieza y Preparación Final
    # Eliminamos registros donde no tengamos la "verdad" (target) para entrenar
    target_cols = ['temp_real', 'hum_real']
    df = df.dropna(subset=target_cols)
    
    # Definición de las features finales que verá el modelo
    feature_cols = [
        'temp_fcst', 'hum_fcst', 
        'hour_sin', 'hour_cos', 
        'is_daytime_fcst',
        'lat_estacion', 'lon_estacion',
        'temp_fcst_diff', 'hum_fcst_diff',
        'temp_error_lag1', 'hum_error_lag1',
        'temp_fcst_roll3_mean', 'temp_fcst_roll3_std'
    ]
    
    # Imputación de NaNs en features (producidos por lags/rolling) usando ffill/bfill por estación
    for col in feature_cols:
        df[col] = df.groupby('station_id')[col].transform(lambda x: x.ffill().bfill())
    
    # Caso borde: si una estación no tiene datos suficientes, llenamos con 0 para evitar errores en XGBoost
    df[feature_cols] = df[feature_cols].fillna(0)

    print(f"Dataset procesado: {len(df)} registros listos para modelado.")
    df.to_parquet(output_path, index=False)
    print(f"Features guardadas en: {output_path}")
    
    return df

if __name__ == "__main__":
    prepare_data()
