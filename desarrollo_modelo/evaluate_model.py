import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Configuración
FEATURE_PATH = "clima_features.parquet"
MODEL_PATH = "bias_correction_model.pkl"
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

def run_evaluation():
    if not os.path.exists(FEATURE_PATH) or not os.path.exists(MODEL_PATH):
        print("Error: Faltan archivos para evaluación.")
        return

    df = pd.read_parquet(FEATURE_PATH)
    df = df.sort_values('obs_timestamp')
    
    # Usar el mismo split que en el entrenamiento para el test final
    split_idx = int(len(df) * 0.85)
    df_test = df.iloc[split_idx:].copy()
    
    model = joblib.load(MODEL_PATH)
    
    X_test = df_test[FEATURES]
    y_test = df_test[TARGETS]
    
    preds = model.predict(X_test)
    df_test['temp_corr'] = preds[:, 0]
    df_test['hum_corr'] = preds[:, 1]
    
    # 1. Gráfico de Dispersión: Predicción vs Realidad
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Temperatura
    sns.scatterplot(x=df_test['temp_real'], y=df_test['temp_corr'], ax=ax1, alpha=0.3)
    ax1.plot([df_test['temp_real'].min(), df_test['temp_real'].max()], 
             [df_test['temp_real'].min(), df_test['temp_real'].max()], 'r--')
    ax1.set_title(f"Temperatura: Real vs Corregido\nMAE: {mean_absolute_error(y_test['temp_real'], df_test['temp_corr']):.2f}")
    
    # Humedad
    sns.scatterplot(x=df_test['hum_real'], y=df_test['hum_corr'], ax=ax2, alpha=0.3)
    ax2.plot([df_test['hum_real'].min(), df_test['hum_real'].max()], 
             [df_test['hum_real'].min(), df_test['hum_real'].max()], 'r--')
    ax2.set_title(f"Humedad: Real vs Corregido\nMAE: {mean_absolute_error(y_test['hum_real'], df_test['hum_corr']):.2f}")
    
    plt.tight_layout()
    plt.savefig("eval_scatter_real_vs_corr.png")
    
    # 2. Análisis de Sesgo por Hora (Ciclo Diurno)
    df_test['hour'] = df_test['obs_timestamp'].dt.hour
    
    # Errores Temperatura
    df_test['error_orig_temp'] = df_test['temp_fcst'] - df_test['temp_real']
    df_test['error_corr_temp'] = df_test['temp_corr'] - df_test['temp_real']
    
    # Errores Humedad
    df_test['error_orig_hum'] = df_test['hum_fcst'] - df_test['hum_real']
    df_test['error_corr_hum'] = df_test['hum_corr'] - df_test['hum_real']
    
    hourly_bias = df_test.groupby('hour')[['error_orig_temp', 'error_corr_temp', 'error_orig_hum', 'error_corr_hum']].mean()
    
    # Plot Sesgo Temperatura
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_bias.index, hourly_bias['error_orig_temp'], label='Sesgo Original (NWS)', marker='o')
    plt.plot(hourly_bias.index, hourly_bias['error_corr_temp'], label='Sesgo Corregido (ML)', marker='s')
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Evolución del Sesgo de Temperatura por Hora")
    plt.xlabel("Hora del Día")
    plt.ylabel("Sesgo (Pronóstico - Real)")
    plt.legend()
    plt.grid(True)
    plt.savefig("eval_bias_hourly_temp.png")
    
    # Plot Sesgo Humedad
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_bias.index, hourly_bias['error_orig_hum'], label='Sesgo Original (NWS)', marker='o', color='blue')
    plt.plot(hourly_bias.index, hourly_bias['error_corr_hum'], label='Sesgo Corregido (ML)', marker='s', color='green')
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Evolución del Sesgo de Humedad por Hora")
    plt.xlabel("Hora del Día")
    plt.ylabel("Sesgo (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("eval_bias_hourly_hum.png")
    
    # 3. Importancia de Features para ambas variables
    # Temperatura (Estimator 0)
    xgb_temp = model.estimators_[0]
    importances_temp = pd.Series(xgb_temp.feature_importances_, index=FEATURES).sort_values(ascending=False)
    
    # Humedad (Estimator 1)
    xgb_hum = model.estimators_[1]
    importances_hum = pd.Series(xgb_hum.feature_importances_, index=FEATURES).sort_values(ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    importances_temp.plot(kind='bar', ax=ax1, color='salmon')
    ax1.set_title("Importancia de Features: Modelo Temperatura")
    
    importances_hum.plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_title("Importancia de Features: Modelo Humedad")
    
    plt.tight_layout()
    plt.savefig("eval_feature_importance_combined.png")

    print("Evaluación completada. Gráficos (Temp/Hum) guardados.")

    print("Evaluación completada. Gráficos guardados.")

if __name__ == "__main__":
    run_evaluation()
