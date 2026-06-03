import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

# Configuración
FEATURE_PATH = "clima_features.parquet"
TARGETS = ['temp_real', 'hum_real']
FEATURES = [
    'temp_fcst', 'hum_fcst', 
    'hour_sin', 'hour_cos', 
    'is_daytime_fcst',
    'lat_estacion', 'lon_estacion',
    'temp_fcst_diff', 'hum_fcst_diff',
    'temp_error_lag1', 'hum_error_lag1',
    'temp_fcst_roll3_mean', 'temp_fcst_roll3_std'
]

def run_lookback_optimization():
    if not os.path.exists(FEATURE_PATH):
        print(f"Error: No se encuentra {FEATURE_PATH}")
        return

    print("Cargando features...")
    df = pd.read_parquet(FEATURE_PATH)
    df['obs_timestamp'] = pd.to_datetime(df['obs_timestamp'])
    df = df.sort_values('obs_timestamp')
    
    # Definir Test Set: Últimos 7 días
    test_end = df['obs_timestamp'].max()
    test_start = test_end - pd.Timedelta(days=7)
    
    df_test = df[df['obs_timestamp'] >= test_start].copy()
    df_train_pool = df[df['obs_timestamp'] < test_start].copy()
    
    print(f"Test Set: {test_start} a {test_end} ({len(df_test)} registros)")
    
    windows = [1, 3, 7, 14, 21, 28, 35]
    results = []
    
    # Parámetros base razonables
    params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }
    
    for w in windows:
        window_start = test_start - pd.Timedelta(days=w)
        if window_start < df['obs_timestamp'].min():
            print(f"Ventana de {w} días excede los datos disponibles. Saltando.")
            continue
            
        df_train = df_train_pool[df_train_pool['obs_timestamp'] >= window_start]
        print(f"Entrenando con ventana de {w} días ({len(df_train)} registros)...")
        
        model = MultiOutputRegressor(xgb.XGBRegressor(**params))
        model.fit(df_train[FEATURES], df_train[TARGETS])
        
        preds = model.predict(df_test[FEATURES])
        mae_temp = mean_absolute_error(df_test['temp_real'], preds[:, 0])
        mae_hum = mean_absolute_error(df_test['hum_real'], preds[:, 1])
        
        results.append({
            'window_days': w,
            'mae_temp': mae_temp,
            'mae_hum': mae_hum,
            'n_samples': len(df_train)
        })
    
    res_df = pd.DataFrame(results)
    print("\nResultados de Optimización:")
    print(res_df)
    
    # Graficar
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(res_df['window_days'], res_df['mae_temp'], marker='o', color='salmon', linewidth=2)
    plt.title("Look-back Window vs MAE Temperatura")
    plt.xlabel("Días de entrenamiento")
    plt.ylabel("MAE (°C)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(res_df['window_days'], res_df['mae_hum'], marker='s', color='skyblue', linewidth=2)
    plt.title("Look-back Window vs MAE Humedad")
    plt.xlabel("Días de entrenamiento")
    plt.ylabel("MAE (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("lookback_optimization.png")
    print("\nGráfico guardado en lookback_optimization.png")
    
    # Guardar CSV de resultados
    res_df.to_csv("lookback_results.csv", index=False)
    print("Resultados guardados en lookback_results.csv")

if __name__ == "__main__":
    run_lookback_optimization()
