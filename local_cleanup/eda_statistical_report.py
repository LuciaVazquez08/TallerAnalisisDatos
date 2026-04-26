import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def clean_wind_speed(val):
    if pd.isna(val) or val == 'nan':
        return np.nan
    if isinstance(val, str):
        # Extraer número de "10 mph" o "Calm"
        if 'mph' in val:
            return float(val.replace(' mph', ''))
        if val.lower() == 'calm':
            return 0.0
    return float(val)

def generate_report(file_path):
    df = pd.read_parquet(file_path)
    
    # Normalización de datos
    df['wind_speed_fcst_numeric'] = df['wind_speed_fcst'].apply(clean_wind_speed)
    
    # Filtrar solo registros con datos reales y pronosticados para métricas
    cols_to_compare = [
        ('temp_real', 'temp_fcst', 'Temperatura (C)'),
        ('dew_real', 'dew_fcst', 'Punto de Rocío (C)'),
        ('hum_real', 'hum_fcst', 'Humedad (%)'),
        ('wind_speed_real', 'wind_speed_fcst_numeric', 'Velocidad Viento (mph)')
    ]
    
    report_data = []
    
    print("=== ANALISIS DESCRIPTIVO DE ERRORES DE PRONOSTICO ===")
    
    for real_col, fcst_col, label in cols_to_compare:
        valid_df = df.dropna(subset=[real_col, fcst_col])
        if len(valid_df) == 0:
            continue
            
        y_true = valid_df[real_col]
        y_pred = valid_df[fcst_col]
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = (y_pred - y_true).mean()
        correl = y_true.corr(y_pred)
        
        report_data.append({
            'Variable': label,
            'MAE': mae,
            'RMSE': rmse,
            'Bias': bias,
            'Correlación': correl,
            'N': len(valid_df)
        })
    
    stats_df = pd.DataFrame(report_data)
    print("\nMetricas Globales:")
    print(stats_df.to_string(index=False))
    
    # Análisis por hora del día (Ciclo Diurno)
    df['hour'] = df['obs_timestamp'].dt.hour
    temp_analysis = df.dropna(subset=['temp_real', 'temp_fcst']).copy()
    temp_analysis['error'] = temp_analysis['temp_fcst'] - temp_analysis['temp_real']
    
    hourly_error = temp_analysis.groupby('hour')['error'].agg(['mean', 'std']).reset_index()
    print("\nSesgo de Temperatura por Hora (Bias):")
    print(hourly_error.to_string(index=False))
    
    # Correlaciones entre variables reales
    real_vars = ['temp_real', 'dew_real', 'hum_real', 'wind_speed_real', 'press_real']
    corr_matrix = df[real_vars].corr()
    print("\nMatriz de Correlación (Variables Reales):")
    print(corr_matrix)

    # Identificar estaciones con mayor error
    station_error = temp_analysis.groupby('station_id')['error'].apply(lambda x: np.abs(x).mean()).sort_values(ascending=False).head(10)
    print("\nTop 10 Estaciones con mayor MAE en Temperatura:")
    print(station_error)

if __name__ == "__main__":
    path = "local_cleanup/clima_unificado.parquet"
    if os.path.exists(path):
        generate_report(path)
    else:
        print(f"No se encontró {path}")
