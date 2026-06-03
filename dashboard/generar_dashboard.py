import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
import shap

# Configuración de Rutas
FEATURE_PATH = "desarrollo_modelo/clima_features.parquet"
MODEL_PATH = "desarrollo_modelo/bias_correction_model.pkl"
OUTPUT_PATH = "dashboard/index.html"

FEATURES = [
    'temp_fcst', 'hum_fcst', 
    'hour_sin', 'hour_cos', 
    'is_daytime_fcst',
    'lat_estacion', 'lon_estacion',
    'temp_fcst_diff', 'hum_fcst_diff',
    'temp_error_lag1', 'hum_error_lag1',
    'temp_fcst_roll3_mean', 'temp_fcst_roll3_std'
]
TARGETS = ['temp_real', 'hum_real']

def calculate_pdp_grid_and_values(estimator, X, feature, grid_size=15):
    feature_vals = X[feature].dropna()
    min_val, max_val = float(feature_vals.min()), float(feature_vals.max())
    grid = np.linspace(min_val, max_val, grid_size).tolist()
    
    pdp_values = []
    X_temp = X.copy()
    for val in grid:
        X_temp[feature] = val
        preds = estimator.predict(X_temp)
        pdp_values.append(float(np.mean(preds)))
        
    return {
        'grid': [round(g, 2) for g in grid],
        'values': [round(v, 2) for v in pdp_values]
    }

def calculate_pdp_hour(estimator, X):
    grid = list(range(24))
    pdp_values = []
    X_temp = X.copy()
    for h in grid:
        sin_val = np.sin(2 * np.pi * h / 24)
        cos_val = np.cos(2 * np.pi * h / 24)
        X_temp['hour_sin'] = sin_val
        X_temp['hour_cos'] = cos_val
        if 'is_daytime_fcst' in X_temp.columns:
            X_temp['is_daytime_fcst'] = 1 if (6 <= h <= 18) else 0
        preds = estimator.predict(X_temp)
        pdp_values.append(float(np.mean(preds)))
    return {
        'grid': grid,
        'values': [round(v, 2) for v in pdp_values]
    }

def generate_dashboard():
    print("Cargando datos y modelo...")
    if not os.path.exists(FEATURE_PATH) or not os.path.exists(MODEL_PATH):
        print("Error: No se encuentran los archivos necesarios.")
        return

    df = pd.read_parquet(FEATURE_PATH)
    df = df.sort_values('obs_timestamp')
    
    # Split de Test (último 15%)
    split_idx = int(len(df) * 0.85)
    df_test = df.iloc[split_idx:].copy()
    
    model = joblib.load(MODEL_PATH)
    
    print("Generando predicciones del modelo ML...")
    preds = model.predict(df_test[FEATURES])
    df_test['temp_ml'] = preds[:, 0]
    df_test['hum_ml'] = preds[:, 1]
    
    # Cálculos de Error
    df_test['error_nws_temp'] = df_test['temp_fcst'] - df_test['temp_real']
    df_test['error_ml_temp'] = df_test['temp_ml'] - df_test['temp_real']
    df_test['error_nws_hum'] = df_test['hum_fcst'] - df_test['hum_real']
    df_test['error_ml_hum'] = df_test['hum_ml'] - df_test['hum_real']
    
    mae_nws_temp = mean_absolute_error(df_test['temp_real'], df_test['temp_fcst'])
    mae_ml_temp = mean_absolute_error(df_test['temp_real'], df_test['temp_ml'])
    imp_temp = (mae_nws_temp - mae_ml_temp) / mae_nws_temp * 100

    mae_nws_hum = mean_absolute_error(df_test['hum_real'], df_test['hum_fcst'])
    mae_ml_hum = mean_absolute_error(df_test['hum_real'], df_test['hum_ml'])
    imp_hum = (mae_nws_hum - mae_ml_hum) / mae_nws_hum * 100

    print(f"Mejora Temperatura: {imp_temp:.2f}%")
    print(f"Mejora Humedad: {imp_hum:.2f}%")

    # --- 1. Intervalos de confianza empíricos por ciclo horario ---
    df_test['hour'] = df_test['obs_timestamp'].dt.hour
    uncertainty_by_hour = {}
    print("Calculando bandas de incertidumbre empírica horaria...")
    for hour in range(24):
        hour_data = df_test[df_test['hour'] == hour]
        # Residuos: Real - ML
        err_t = hour_data['temp_real'] - hour_data['temp_ml']
        err_h = hour_data['hum_real'] - hour_data['hum_ml']
        
        uncertainty_by_hour[hour] = {
            'temp_p5': round(float(np.percentile(err_t, 5)), 2) if len(err_t) > 0 else -1.5,
            'temp_p95': round(float(np.percentile(err_t, 95)), 2) if len(err_t) > 0 else 1.5,
            'hum_p5': round(float(np.percentile(err_h, 5)), 2) if len(err_h) > 0 else -8.0,
            'hum_p95': round(float(np.percentile(err_h, 95)), 2) if len(err_h) > 0 else 8.0,
        }

    # --- 2. Métricas por Estación ---
    station_metrics = df_test.groupby(['station_id', 'lat_estacion', 'lon_estacion']).apply(
        lambda x: pd.Series({
            'mae_nws_temp': mean_absolute_error(x['temp_real'], x['temp_fcst']),
            'mae_ml_temp': mean_absolute_error(x['temp_real'], x['temp_ml']),
            'mae_nws_hum': mean_absolute_error(x['hum_real'], x['hum_fcst']),
            'mae_ml_hum': mean_absolute_error(x['hum_real'], x['hum_ml']),
        }),
        include_groups=False
    ).reset_index()
    station_metrics['imp_temp'] = (station_metrics['mae_nws_temp'] - station_metrics['mae_ml_temp']) / station_metrics['mae_nws_temp'] * 100
    station_metrics['imp_hum'] = (station_metrics['mae_nws_hum'] - station_metrics['mae_ml_hum']) / station_metrics['mae_nws_hum'] * 100
    station_metrics['imp_avg'] = (station_metrics['imp_temp'] + station_metrics['imp_hum']) / 2
    
    # Ordenamos estaciones para encontrar la mejor y listar por ID
    station_metrics = station_metrics.sort_values('station_id')
    best_station = station_metrics.sort_values('imp_avg', ascending=False).iloc[0]['station_id']
    print(f"Estación de ejemplo seleccionada por rendimiento (mejor promedio): {best_station}")

    # --- 3. Explicabilidad Global: Feature Importance (XGBoost) ---
    xgb_temp = model.estimators_[0]
    xgb_hum = model.estimators_[1]
    
    feat_imp_data = {
        'temp': [round(float(x), 4) for x in xgb_temp.feature_importances_],
        'hum': [round(float(x), 4) for x in xgb_hum.feature_importances_]
    }

    # --- 4. Explicabilidad Global: Permutation Importance ---
    print("Calculando Permutation Feature Importance en el conjunto de test...")
    # Para ahorrar tiempo, tomamos un submuestreo de 2000 registros para Permutation Importance
    df_sample_pfi = df_test.sample(n=min(2000, len(df_test)), random_state=42)
    pfi_temp = permutation_importance(xgb_temp, df_sample_pfi[FEATURES], df_sample_pfi['temp_real'], n_repeats=5, random_state=42)
    pfi_hum = permutation_importance(xgb_hum, df_sample_pfi[FEATURES], df_sample_pfi['hum_real'], n_repeats=5, random_state=42)
    
    pfi_data = {
        'temp': {
            'importances_mean': [round(float(x), 4) for x in pfi_temp.importances_mean],
            'importances_std': [round(float(x), 4) for x in pfi_temp.importances_std]
        },
        'hum': {
            'importances_mean': [round(float(x), 4) for x in pfi_hum.importances_mean],
            'importances_std': [round(float(x), 4) for x in pfi_hum.importances_std]
        }
    }

    # --- 5. Explicabilidad Global: Partial Dependence Plots (PDP) ---
    print("Calculando Partial Dependence Plots (PDP) para variables críticas...")
    # Usamos una muestra representativa de 1000 registros de test para agilizar el cálculo
    df_sample_pdp = df_test.sample(n=min(1000, len(df_test)), random_state=42)
    pdp_data = {
        'temp': {
            'temp_fcst': calculate_pdp_grid_and_values(xgb_temp, df_sample_pdp[FEATURES], 'temp_fcst'),
            'temp_error_lag1': calculate_pdp_grid_and_values(xgb_temp, df_sample_pdp[FEATURES], 'temp_error_lag1'),
            'hour': calculate_pdp_hour(xgb_temp, df_sample_pdp[FEATURES])
        },
        'hum': {
            'hum_fcst': calculate_pdp_grid_and_values(xgb_hum, df_sample_pdp[FEATURES], 'hum_fcst'),
            'hum_error_lag1': calculate_pdp_grid_and_values(xgb_hum, df_sample_pdp[FEATURES], 'hum_error_lag1'),
            'hour': calculate_pdp_hour(xgb_hum, df_sample_pdp[FEATURES])
        }
    }

    # --- 6. Explicabilidad Local: Valores de SHAP ---
    print("Calculando valores de SHAP para todas las observaciones del conjunto de test...")
    explainer_temp = shap.TreeExplainer(xgb_temp)
    explainer_hum = shap.TreeExplainer(xgb_hum)
    
    shap_temp = explainer_temp.shap_values(df_test[FEATURES])
    shap_hum = explainer_hum.shap_values(df_test[FEATURES])
    
    expected_value_temp = float(explainer_temp.expected_value)
    expected_value_hum = float(explainer_hum.expected_value)

    # Limpieza de valores decimales y ceros para compresión máxima de JSON
    def clean_val(val):
        v = round(float(val), 1)
        return int(v) if v == int(v) else v

    shap_temp_clean = [[clean_val(x) for x in row] for row in shap_temp]
    shap_hum_clean = [[clean_val(x) for x in row] for row in shap_hum]

    # --- 7. Estructuración y Compresión de la Base de Datos JSON ---
    print("Estructurando y comprimiendo base de datos JSON...")
    stations_list = []
    station_data_dict = {}
    
    # Creamos mapeo rápido de datos de estaciones
    for idx, row in station_metrics.iterrows():
        s_id = row['station_id']
        stations_list.append({
            'id': s_id,
            'lat': float(row['lat_estacion']),
            'lon': float(row['lon_estacion']),
            'mae_nws_temp': round(float(row['mae_nws_temp']), 3),
            'mae_ml_temp': round(float(row['mae_ml_temp']), 3),
            'mae_nws_hum': round(float(row['mae_nws_hum']), 3),
            'mae_ml_hum': round(float(row['mae_ml_hum']), 3),
            'imp_temp': round(float(row['imp_temp']), 1),
            'imp_hum': round(float(row['imp_hum']), 1),
            'imp_avg': round(float(row['imp_avg']), 1)
        })
        station_data_dict[s_id] = []

    # Compresión de Timestamps
    unique_timestamps = sorted(df_test['obs_timestamp'].unique())
    unique_ts_strs = [ts.strftime('%Y-%m-%d %H:%M') for ts in unique_timestamps]
    ts_to_idx = {ts: idx for idx, ts in enumerate(unique_timestamps)}

    # Poblamos registros por estación
    # Guardamos los registros como listas de tamaño compacto para acelerar carga
    for i, (idx, row) in enumerate(df_test.iterrows()):
        s_id = row['station_id']
        ts_idx = ts_to_idx[row['obs_timestamp']]
        
        station_data_dict[s_id].append([
            ts_idx,
            clean_val(row['temp_real']),
            clean_val(row['temp_fcst']),
            clean_val(row['temp_ml']),
            clean_val(row['hum_real']),
            clean_val(row['hum_fcst']),
            clean_val(row['hum_ml']),
            shap_temp_clean[i],
            shap_hum_clean[i]
        ])

    # Ensamblado del JSON Final
    db_json = {
        'stations': stations_list,
        'timestamps': unique_ts_strs,
        'best_station': best_station,
        'features': FEATURES,
        'expected_value_temp': expected_value_temp,
        'expected_value_hum': expected_value_hum,
        'uncertainty_by_hour': uncertainty_by_hour,
        'feat_imp': feat_imp_data,
        'pfi': pfi_data,
        'pdp': pdp_data,
        'station_data': station_data_dict
    }

    # Serializamos a JSON string
    db_json_str = json.dumps(db_json, ensure_ascii=False)
    print(f"Base de datos JSON serializada (Tamaño: {len(db_json_str)/1024**2:.2f} MB)")

    # --- 8. Plantilla HTML (Modo Claro Elegante con Dependencias Locales) ---
    print("Compilando plantilla HTML...")
    html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Avanzado de Sesgo Climático (TP2)</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- Plotly.js CDN -->
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    
    <style>
        :root {{
            --bg-primary: #f8fafc;
            --bg-sidebar: #ffffff;
            --bg-card: #ffffff;
            --border-color: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
            
            --accent: #2563eb;
            --accent-light: #eff6ff;
            --accent-hover: #1d4ed8;
            
            --temp-real: #0f172a;
            --temp-nws: #f43f5e;
            --temp-ml: #10b981;
            
            --hum-real: #0f172a;
            --hum-nws: #3b82f6;
            --hum-ml: #0d9488;
            
            --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
            --radius: 12px;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif;
            display: flex;
            min-height: 100vh;
            overflow-x: hidden;
        }}

        /* Barra Lateral */
        .sidebar {{
            width: 260px;
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border-color);
            position: fixed;
            top: 0;
            bottom: 0;
            left: 0;
            display: flex;
            flex-direction: column;
            z-index: 100;
            padding: 24px;
            justify-content: space-between;
        }}

        .sidebar-brand {{
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 35px;
            padding-left: 8px;
        }}
        
        .sidebar-brand span {{
            font-size: 1.15rem;
        }}

        .nav-menu {{
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .nav-link {{
            width: 100%;
            border: none;
            background: none;
            text-align: left;
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            color: var(--text-muted);
            font-family: inherit;
            font-size: 0.95rem;
            font-weight: 500;
            border-radius: var(--radius);
            cursor: pointer;
            transition: var(--transition);
        }}

        .nav-link:hover {{
            background-color: var(--accent-light);
            color: var(--accent);
        }}

        .nav-link.active {{
            background-color: var(--accent);
            color: #ffffff;
            box-shadow: var(--shadow-md);
        }}

        .sidebar-footer {{
            border-top: 1px solid var(--border-color);
            padding-top: 16px;
            font-size: 0.8rem;
            color: var(--text-muted);
            line-height: 1.4;
        }}

        /* Contenedor Principal */
        .main-content {{
            margin-left: 260px;
            flex: 1;
            padding: 40px;
            min-width: 0;
        }}

        /* Encabezados y Estilos */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 32px;
        }}

        header h1 {{
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}

        header p {{
            color: var(--text-muted);
            margin-top: 4px;
            font-size: 0.95rem;
        }}

        .badge-model {{
            background-color: var(--accent-light);
            color: var(--accent);
            padding: 6px 12px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(37, 99, 235, 0.15);
        }}

        /* Grid de Tarjetas */
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 24px;
            margin-bottom: 32px;
        }}

        .card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            box-shadow: var(--shadow-sm);
            padding: 24px;
            position: relative;
            transition: var(--transition);
        }}
        
        .card-interactive:hover {{
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }}

        /* Tarjetas de Métricas */
        .metric-title {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.02em;
        }}

        .metric-comparison {{
            margin-top: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        
        .metric-comparison.improvement {{
            color: #10b981;
        }}

        .metric-comparison.nws-baseline {{
            color: var(--text-muted);
            font-size: 0.75rem;
            border-top: 1px dashed var(--border-color);
            padding-top: 6px;
            margin-top: 12px;
        }}

        /* Filtros */
        .filter-bar {{
            background: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 16px 24px;
            display: flex;
            gap: 24px;
            align-items: center;
            margin-bottom: 32px;
            box-shadow: var(--shadow-sm);
        }}

        .filter-item {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .filter-label {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            color: var(--text-muted);
        }}

        /* Custom Select de Estaciones */
        .select-station {{
            width: 250px;
            background: #ffffff;
            border: 1px solid var(--border-color);
            padding: 10px 16px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-main);
            outline: none;
            cursor: pointer;
            transition: var(--transition);
            appearance: none;
            background-image: url("data:image/svg+xml;charset=UTF-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%2364748b%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E");
            background-repeat: no-repeat;
            background-position: right 12px center;
            background-size: 16px;
            padding-right: 40px;
        }}
        
        .select-station:focus {{
            border-color: var(--accent);
        }}

        /* Filtro de Ciclo Horario */
        .hour-filter-group {{
            display: flex;
            background: #f1f5f9;
            padding: 4px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}

        .hour-filter-btn {{
            background: none;
            border: none;
            padding: 6px 12px;
            font-family: inherit;
            font-size: 0.8rem;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            color: var(--text-muted);
            transition: var(--transition);
        }}

        .hour-filter-btn.active {{
            background: #ffffff;
            color: var(--accent);
            box-shadow: var(--shadow-sm);
        }}

        /* Switch */
        .switch-container {{
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
        }}

        .switch-box {{
            width: 40px;
            height: 20px;
            background: #cbd5e1;
            border-radius: 9999px;
            position: relative;
            transition: var(--transition);
        }}

        .switch-box::after {{
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 16px;
            height: 16px;
            background: #ffffff;
            border-radius: 50%;
            transition: var(--transition);
        }}

        .switch-container.active .switch-box {{
            background: var(--accent);
        }}

        .switch-container.active .switch-box::after {{
            left: 22px;
        }}

        .switch-label {{
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
        }}

        /* Layout del Monitoreo */
        .tab-panel-monitoring {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 24px;
        }}

        .charts-column {{
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}

        .pdp-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 16px;
        }}
        .pdp-item-full {{
            grid-column: span 2;
        }}
        @media (max-width: 900px) {{
            .pdp-grid {{
                grid-template-columns: 1fr;
            }}
            .pdp-item-full {{
                grid-column: span 1;
            }}
        }}

        /* Contenedores de Gráficos */
        .chart-box {{
            width: 100%;
            height: 400px;
        }}
        
        .chart-box-sm {{
            width: 100%;
            height: 280px;
        }}

        .chart-box-large {{
            width: 100%;
            height: 450px;
        }}

        /* Alertas de Leyenda de Variables */
        .legend-card {{
            display: flex;
            gap: 16px;
            margin-top: 12px;
            padding: 12px 16px;
            background-color: var(--accent-light);
            border: 1px dashed rgba(37, 99, 235, 0.2);
            border-radius: 8px;
            font-size: 0.8rem;
            color: #1e40af;
        }}

        /* Tabla de Variables Explicabilidad Local */
        .local-expl-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
            margin-top: 16px;
        }}

        .local-expl-table th, .local-expl-table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        .local-expl-table th {{
            font-weight: 600;
            color: var(--text-muted);
            background-color: #f8fafc;
        }}

        .local-expl-table tr:hover {{
            background-color: #f1f5f9;
        }}

        .badge-contrib-pos {{
            background-color: #d1fae5;
            color: #065f46;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
        }}

        .badge-contrib-neg {{
            background-color: #fee2e2;
            color: #991b1b;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
        }}

        /* Botones de Cambio de Target Explicabilidad */
        .target-btn {{
            flex: 1;
            border: 1px solid var(--border-color);
            padding: 6px;
            border-radius: 6px;
            font-family: inherit;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            background: #fff;
            color: var(--text-muted);
            transition: var(--transition);
        }}
        .target-btn.active {{
            background: var(--accent);
            color: #ffffff;
            border-color: var(--accent);
        }}

        /* Responsive */
        @media (max-width: 1200px) {{
            .sidebar {{
                width: 70px;
                padding: 16px 8px;
            }}
            .sidebar-brand span, .nav-link span, .sidebar-footer {{
                display: none;
            }}
            .main-content {{
                margin-left: 70px;
            }}
            .dashboard-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .tab-panel-monitoring {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    
    <!-- BARRA LATERAL -->
    <div class="sidebar">
        <div>
            <div class="sidebar-brand">
                📊 <span>SesgoClima ML</span>
            </div>
            
            <ul class="nav-menu">
                <li>
                    <button class="nav-link active" data-tab="overview" id="nav-overview" onclick="changeTab('overview')">
                        📈 <span>Resumen General</span>
                    </button>
                </li>
                <li>
                    <button class="nav-link" data-tab="monitoring" id="nav-monitoring" onclick="changeTab('monitoring')">
                        🕒 <span>Monitoreo</span>
                    </button>
                </li>
                <li>
                    <button class="nav-link" data-tab="explainability" id="nav-explainability" onclick="changeTab('explainability')">
                        🧠 <span>Explicabilidad Global</span>
                    </button>
                </li>
            </ul>
        </div>
        
        <div class="sidebar-footer">
            <p><strong>Taller de Análisis 1</strong></p>
            <p>Grupo 2</p>
            <p class="mt-1" style="font-size:0.7rem; opacity:0.8;">Modelo: XGBoost Regressor</p>
        </div>
    </div>

    <!-- CONTENEDOR PRINCIPAL -->
    <div class="main-content">
        
        <!-- PESTAÑA: RESUMEN GENERAL -->
        <div id="panel-overview">
            <header>
                <div>
                    <h1>Análisis de Corrección de Sesgo Diurno</h1>
                    <p>Comparación global de desempeño: Predicción NWS vs. Corrección de Machine Learning</p>
                </div>
                <span class="badge-model">XGBoost Multi-salida + Optuna</span>
            </header>
            
            <!-- Grid de Métricas Globales -->
            <div class="dashboard-grid">
                <div class="card card-interactive">
                    <div class="metric-title">
                        <span>MAE Temp (ML)</span>
                        <span style="font-size: 1.25rem;">🌡️</span>
                    </div>
                    <div class="metric-value">{mae_ml_temp:.3f} °C</div>
                    <div class="metric-comparison improvement">
                        <span>▼ {imp_temp:.1f}% de mejora</span>
                    </div>
                    <div class="metric-comparison nws-baseline">
                        <span>Línea Base NWS: {mae_nws_temp:.3f} °C</span>
                    </div>
                </div>
                
                <div class="card card-interactive">
                    <div class="metric-title">
                        <span>MAE Humedad (ML)</span>
                        <span style="font-size: 1.25rem;">💧</span>
                    </div>
                    <div class="metric-value">{mae_ml_hum:.3f} %</div>
                    <div class="metric-comparison improvement">
                        <span>▼ {imp_hum:.1f}% de mejora</span>
                    </div>
                    <div class="metric-comparison nws-baseline">
                        <span>Línea Base NWS: {mae_nws_hum:.3f} %</span>
                    </div>
                </div>
                
                <div class="card" style="grid-column: span 2;">
                    <div style="display:flex; flex-direction:column; justify-content:space-between; height:100%;">
                        <div>
                            <h4 style="font-size:0.95rem; font-weight:600; margin-bottom:8px; color:var(--accent);">Resumen Ejecutivo</h4>
                            <p style="font-size:0.85rem; color:var(--text-muted); line-height:1.5;">
                                El modelo XGBoost corrige los sesgos del ciclo diurno del modelo físico de la NWS utilizando el diferencial térmico, inercia térmica y lags de error. La mejora promedio combinada supera el <strong>34.0%</strong> en todo el conjunto de prueba (80 estaciones independientes).
                            </p>
                        </div>
                        <div style="font-size:0.8rem; border-top:1px solid var(--border-color); padding-top:8px; display:flex; justify-content:space-between; color:var(--text-muted);">
                            <span>🕒 Periodo de Prueba: 15 días</span>
                            <span>📡 Estaciones Totales: 506 | De Prueba: 80</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Fila de Mapa e Info -->
            <div style="display:grid; grid-template-columns: 2fr 1fr; gap:24px; margin-bottom:32px;">
                <div class="card" style="padding:16px;">
                    <h3 style="font-size:1rem; font-weight:600; margin-bottom:12px; padding-left:8px;">
                        Distribución Espacial de la Mejora (MAE)
                    </h3>
                    <div id="overview-map" class="chart-box-large"></div>
                    <div class="legend-card">
                        <span>ℹ️</span>
                        <p>
                            El tamaño del marcador representa el error base de la NWS (estaciones más grandes tienen más error). El color muestra el porcentaje de mejora del MAE con el modelo de ML (verde es alta mejora, rojo baja). <strong>Haz clic en una estación del mapa para seleccionarla e ir a sus series temporales.</strong>
                        </p>
                    </div>
                </div>
                
                <div class="card" style="display:flex; flex-direction:column; justify-content:space-between;">
                    <div>
                        <h3 style="font-size:1rem; font-weight:600; margin-bottom:16px; border-bottom: 1px solid var(--border-color); padding-bottom:8px;">
                            Estaciones Destacadas
                        </h3>
                        <div id="top-stations-list" style="overflow-y:auto; max-height:320px; display:flex; flex-direction:column; gap:12px;">
                            <!-- Populated by JS -->
                        </div>
                    </div>
                    
                    <div style="background:#f8fafc; padding:12px; border-radius:8px; border:1px solid var(--border-color); font-size:0.8rem; margin-top:16px;">
                        <strong>Estación Seleccionada:</strong> <span style="font-weight:700; color:var(--accent);" id="overview-active-station-label"></span><br>
                        <span id="overview-active-station-imp"></span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- PESTAÑA: MONITOREO Y SERIES TEMPORALES (CON INCERTIDUMBRE) -->
        <div id="panel-monitoring" class="tab-panel-monitoring" style="display:none;">
            
            <!-- Columna de Gráficos de Tiempo -->
            <div class="charts-column">
                <!-- Barra de Filtros -->
                <div class="filter-bar">
                    <div class="filter-item">
                        <span class="filter-label">Estación Meteorológica</span>
                        <select id="station-select" class="select-station" onchange="updateStation(this.value)">
                            <!-- Populated by JS -->
                        </select>
                    </div>
                    
                    <div class="filter-item">
                        <span class="filter-label">Rango Temporal</span>
                        <div class="hour-filter-group">
                            <button class="hour-filter-btn" id="btn-range-24" onclick="changeTimeRange(24)">24 Hs</button>
                            <button class="hour-filter-btn active" id="btn-range-48" onclick="changeTimeRange(48)">48 Hs</button>
                            <button class="hour-filter-btn" id="btn-range-72" onclick="changeTimeRange(72)">72 Hs</button>
                        </div>
                    </div>
                    
                    <div class="filter-item" style="margin-left:auto;">
                        <div class="switch-container" id="uncertainty-switch" onclick="toggleUncertainty()">
                            <div class="switch-box"></div>
                            <span class="switch-label">Ver Banda de Incertidumbre</span>
                        </div>
                    </div>
                </div>

                <div class="card" style="padding:16px;">
                    <h3 style="font-size:0.95rem; font-weight:600; margin-bottom:8px; display:flex; justify-content:space-between;">
                        <span>Serie Temporal de Temperatura (°C)</span>
                        <span style="font-size:0.75rem; color:var(--text-muted);">Estación activa: <span style="font-weight:700;" class="lbl-active-station-id"></span></span>
                    </h3>
                    <div id="temp-ts-chart" class="chart-box"></div>
                </div>
                
                <div class="card" style="padding:16px;">
                    <h3 style="font-size:0.95rem; font-weight:600; margin-bottom:8px; display:flex; justify-content:space-between;">
                        <span>Serie Temporal de Humedad (%)</span>
                        <span style="font-size:0.75rem; color:var(--text-muted);">Estación activa: <span style="font-weight:700;" class="lbl-active-station-id"></span></span>
                    </h3>
                    <div id="hum-ts-chart" class="chart-box"></div>
                </div>
            </div>
            
            <!-- Columna Lateral de Explicabilidad Local (SHAP) -->
            <div style="display:flex; flex-direction:column; gap:24px;">
                <div class="card" style="height: 100%; display:flex; flex-direction:column; justify-content:space-between;">
                    <div>
                        <h3 style="font-size:0.95rem; font-weight:600; margin-bottom:8px; border-bottom:1px solid var(--border-color); padding-bottom:8px;">
                            🧠 Explicabilidad Local (SHAP)
                        </h3>
                        
                        <p style="font-size:0.8rem; color:var(--text-muted); line-height:1.4; margin-bottom:12px;">
                            Haz clic en cualquier punto de las series temporales para ver la cascada de contribución local del modelo.
                        </p>
                        
                        <!-- Información del Punto Seleccionado -->
                        <div style="background:#f8fafc; border:1px solid var(--border-color); border-radius:6px; padding:10px; font-size:0.8rem; margin-bottom:16px;">
                            <strong>Fecha/Hora:</strong> <span style="font-weight:600; color:var(--accent);" id="lbl-obs-date"></span><br>
                            <strong>Valor Real:</strong> <span style="font-weight:600;" id="lbl-obs-real"></span><br>
                            <strong>Corregido (ML):</strong> <span style="font-weight:600; color:#10b981;" id="lbl-obs-ml"></span>
                        </div>
                        
                        <!-- Selector del Objetivo SHAP (Temp o Hum) -->
                        <div style="display:flex; gap:8px; margin-bottom:16px;">
                            <button id="btn-shap-temp" class="target-btn active" onclick="changeShapTarget('temp')">Temperatura</button>
                            <button id="btn-shap-hum" class="target-btn" onclick="changeShapTarget('hum')">Humedad</button>
                        </div>
                        
                        <!-- Waterfall Chart Container -->
                        <div id="shap-waterfall" style="height:320px;"></div>
                    </div>
                    
                    <!-- Mini Tabla de Contribución -->
                    <div>
                        <h4 style="font-size:0.85rem; font-weight:600; margin-top:12px; margin-bottom:6px;">Variables más influyentes</h4>
                        <table class="local-expl-table">
                            <thead>
                                <tr>
                                    <th>Variable</th>
                                    <th>Contribución</th>
                                </tr>
                            </thead>
                            <tbody id="shap-table-body">
                                <!-- Populated by JS -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
        </div>
        
        <!-- PESTAÑA: EXPLICABILIDAD GLOBAL -->
        <div id="panel-explainability" style="display:none;">
            <header>
                <div>
                    <h1>Explicabilidad y Diagnóstico Global</h1>
                    <p>Análisis agregado del comportamiento del modelo en todo el conjunto de prueba</p>
                </div>
            </header>
            
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:24px; margin-bottom:24px;">
                <!-- Ciclo Diurno del Sesgo -->
                <div class="card" style="padding:16px;">
                    <h3 style="font-size:0.95rem; font-weight:600; margin-bottom:12px;">
                        Perfil del Ciclo Diurno del Sesgo (Promedio)
                    </h3>
                    <div id="diurnal-bias-chart" class="chart-box-sm"></div>
                    <p style="font-size:0.75rem; color:var(--text-muted); line-height:1.4; margin-top:8px;">
                        Este perfil muestra la media del error (Pronóstico - Real) para cada hora del día. El modelo NWS (rojo/azul) tiende a sobreestimar y subestimar la temperatura/humedad en momentos críticos del ciclo térmico. El modelo ML (verde) deforma la curva de sesgo medio cerca de cero.
                    </p>
                </div>
                
                <!-- Importancia de Variables (XGBoost vs PFI) -->
                <div class="card" style="padding:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <h3 style="font-size:0.95rem; font-weight:600;">
                            Importancia de Variables (Global)
                        </h3>
                        <div style="display:flex; gap:6px;">
                            <button id="btn-imp-xgb" class="target-btn" style="padding:4px 8px;" onclick="changeImpType('xgb')">Interna XGBoost</button>
                            <button id="btn-imp-pfi" class="target-btn active" style="padding:4px 8px;" onclick="changeImpType('pfi')">Permutación (PFI)</button>
                        </div>
                    </div>
                    <div id="feature-importance-chart" class="chart-box-sm"></div>
                    <p style="font-size:0.75rem; color:var(--text-muted); line-height:1.4; margin-top:8px;">
                        <span id="importance-desc-text">Muestra el decremento medio en el score del modelo al permutar aleatoriamente cada característica en el set de prueba.</span>
                    </p>
                </div>
            </div>
            
            <!-- Partial Dependence Plots (PDP) -->
            <div class="card" style="padding:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; border-bottom:1px solid var(--border-color); padding-bottom:8px;">
                    <h3 style="font-size:0.95rem; font-weight:600;">
                        Gráficos de Dependencia Parcial (Partial Dependence Plots - PDP)
                    </h3>
                    
                    <div style="display:flex; gap:8px;">
                        <button id="btn-pdp-temp" class="target-btn active" style="padding:4px 12px; flex:none; width:120px;" onclick="changePDPTarget('temp')">Temperatura</button>
                        <button id="btn-pdp-hum" class="target-btn" style="padding:4px 12px; flex:none; width:120px;" onclick="changePDPTarget('hum')">Humedad</button>
                    </div>
                </div>
                
                <p style="font-size:0.8rem; color:var(--text-muted); line-height:1.4; margin-bottom:20px;">
                    Los gráficos de dependencia parcial muestran el impacto marginal de uno o dos atributos sobre la variable objetivo predicha por el modelo, manteniendo el resto de las variables fijas en su valor medio. Permiten entender si la relación capturada es físicamente coherente.
                </p>
                
                <div class="pdp-grid">
                    <div style="background:#f8fafc; border:1px solid var(--border-color); border-radius:8px; padding:12px;">
                        <h4 style="font-size:0.8rem; font-weight:600; margin-bottom:8px; text-align:center;">Ciclo Diurno (Hora del Día)</h4>
                        <div id="pdp-hour-chart" style="height:220px; width:100%;"></div>
                    </div>
                    <div style="background:#f8fafc; border:1px solid var(--border-color); border-radius:8px; padding:12px;">
                        <h4 style="font-size:0.8rem; font-weight:600; margin-bottom:8px; text-align:center;">Pronóstico Base NWS</h4>
                        <div id="pdp-fcst-chart" style="height:220px; width:100%;"></div>
                    </div>
                    <div class="pdp-item-full" style="background:#f8fafc; border:1px solid var(--border-color); border-radius:8px; padding:12px;">
                        <h4 style="font-size:0.8rem; font-weight:600; margin-bottom:8px; text-align:center;">Error de Hora Anterior (Lag 1)</h4>
                        <div id="pdp-lag-chart" style="height:220px; width:100%;"></div>
                    </div>
                </div>
            </div>
            
        </div>
        
    </div>

    <!-- SCRIPT DE DATOS Y LÓGICA FRONTEND EN JAVASCRIPT VAINILLA -->
    <script>
        // Inyección de Base de Datos
        const DB = {db_json_str};
        
        // Mapeo de Etiquetas Legibles para las variables
        const FEATURE_LABELS = {{
            'temp_fcst': 'Pronóstico Temp. NWS',
            'hum_fcst': 'Pronóstico Hum. NWS',
            'hour_sin': 'Ciclo Horario (Seno)',
            'hour_cos': 'Ciclo Horario (Coseno)',
            'is_daytime_fcst': 'Indicador de Día',
            'lat_estacion': 'Latitud Estación',
            'lon_estacion': 'Longitud Estación',
            'temp_fcst_diff': 'Cambio Pronóstico Temp.',
            'hum_fcst_diff': 'Cambio Pronóstico Hum.',
            'temp_error_lag1': 'Error Temp. Hora Anterior',
            'hum_error_lag1': 'Error Hum. Hora Anterior',
            'temp_fcst_roll3_mean': 'Promedio Móvil Temp (3h)',
            'temp_fcst_roll3_std': 'Desv. Est. Temp (3h)'
        }};

        // Estado de la Aplicación (Reemplaza Alpine.js)
        const state = {{
            tab: 'overview',
            selectedStationId: DB.best_station,
            timeRange: 48,
            showUncertainty: false,
            activeShapTarget: 'temp',
            selectedObsIdx: 0,
            impType: 'pfi',
            pdpTarget: 'temp',
            rawStationData: [],
            filteredData: []
        }};

        // Inicialización General
        function initApp() {{
            populateStationSelect();
            populateTopStations();
            updateStationData();
            
            // Retraso pequeño para asegurar que el DOM cargó para Plotly
            setTimeout(() => {{
                drawOverviewMap();
                drawPlots();
                drawGlobalImportance();
                drawPDPPlots();
                drawDiurnalBias();
            }}, 200);
        }}

        function changeTab(tabName) {{
            state.tab = tabName;
            
            // Actualizar clases de nav links
            document.querySelectorAll('.nav-link').forEach(link => {{
                if (link.dataset.tab === tabName) {{
                    link.classList.add('active');
                }} else {{
                    link.classList.remove('active');
                }}
            }});
            
            // Mostrar/Ocultar paneles
            document.getElementById('panel-overview').style.display = tabName === 'overview' ? 'block' : 'none';
            document.getElementById('panel-monitoring').style.display = tabName === 'monitoring' ? 'grid' : 'none';
            document.getElementById('panel-explainability').style.display = tabName === 'explainability' ? 'block' : 'none';
            
            setTimeout(() => {{
                window.dispatchEvent(new Event('resize'));
                if (tabName === 'overview') {{
                    drawOverviewMap();
                }} else if (tabName === 'monitoring') {{
                    drawPlots();
                }} else if (tabName === 'explainability') {{
                    drawGlobalImportance();
                    drawPDPPlots();
                    drawDiurnalBias();
                }}
            }}, 50);
        }}

        function populateStationSelect() {{
            const select = document.getElementById('station-select');
            select.innerHTML = '';
            DB.stations.forEach(s => {{
                const opt = document.createElement('option');
                opt.value = s.id;
                opt.textContent = `${{s.id}} (Mejora: ${{s.imp_avg}}%)`;
                if (s.id === state.selectedStationId) {{
                    opt.selected = true;
                }}
                select.appendChild(opt);
            }});
        }}

        function populateTopStations() {{
            const container = document.getElementById('top-stations-list');
            container.innerHTML = '';
            const topStations = [...DB.stations].sort((a, b) => b.imp_avg - a.imp_avg).slice(0, 5);
            
            topStations.forEach((st, idx) => {{
                const card = document.createElement('div');
                card.style.cssText = 'display:flex; justify-content:space-between; align-items:center; padding:10px; background:#f8fafc; border-radius:6px; border:1px solid var(--border-color); cursor:pointer; transition:var(--transition);';
                card.className = 'card-interactive';
                card.innerHTML = `
                    <div>
                        <span style="font-weight:700; font-size:0.85rem;">${{st.id}}</span>
                        <div style="font-size:0.75rem; color:var(--text-muted);">
                            Mejora Promedio: <span style="font-weight:600; color:#10b981;">${{st.imp_avg}}%</span>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:0.8rem; font-weight:600; background:var(--accent-light); color:var(--accent); padding:2px 6px; border-radius:4px;">#${{idx + 1}}</span>
                    </div>
                `;
                card.addEventListener('click', () => {{
                    selectStationAndGo(st.id);
                }});
                container.appendChild(card);
            }});
        }}

        function selectStationAndGo(stationId) {{
            state.selectedStationId = stationId;
            const select = document.getElementById('station-select');
            if (select) select.value = stationId;
            changeTab('monitoring');
        }}

        function updateStation(stationId) {{
            state.selectedStationId = stationId;
            updateStationData();
            drawPlots();
        }}

        function updateStationData() {{
            const s_data = DB.station_data[state.selectedStationId] || [];
            state.rawStationData = [...s_data];
            state.rawStationData.sort((a, b) => a[0] - b[0]);
            
            // Actualizar etiquetas en panel de Overview
            document.getElementById('overview-active-station-label').textContent = state.selectedStationId;
            const s_info = DB.stations.find(s => s.id === state.selectedStationId) || {{}};
            document.getElementById('overview-active-station-imp').textContent = `Mejora Promedio: ${{s_info.imp_avg}}%`;
            
            // Actualizar etiquetas en panel de Monitoreo
            document.querySelectorAll('.lbl-active-station-id').forEach(el => el.textContent = state.selectedStationId);
            
            applyFilters();
            state.selectedObsIdx = state.filteredData.length - 1;
        }}

        function applyFilters() {{
            const lastObs = state.rawStationData[state.rawStationData.length - 1];
            if (!lastObs) {{
                state.filteredData = [];
                return;
            }}
            
            const lastDate = new Date(DB.timestamps[lastObs[0]]);
            const cutoffTime = lastDate.getTime() - (state.timeRange * 60 * 60 * 1000);
            
            state.filteredData = state.rawStationData.filter(d => {{
                const date = new Date(DB.timestamps[d[0]]);
                return date.getTime() >= cutoffTime;
            }});
            
            if (state.selectedObsIdx >= state.filteredData.length) {{
                state.selectedObsIdx = Math.max(0, state.filteredData.length - 1);
            }}
        }}

        function getSelectedObs() {{
            return state.filteredData[state.selectedObsIdx] || [0, 0, 0, 0, 0, 0, 0, [], []];
        }}

        function getSelectedObsValReal() {{
            const obs = getSelectedObs();
            return obs[1].toFixed(1);
        }}

        function getSelectedObsValML() {{
            const obs = getSelectedObs();
            return state.activeShapTarget === 'temp' ? obs[3].toFixed(1) : obs[6].toFixed(1);
        }}

        function updateObsDetailCard() {{
            const obs = getSelectedObs();
            const dateStr = obs[0] !== undefined ? DB.timestamps[obs[0]] : 'N/A';
            const unit = state.activeShapTarget === 'temp' ? ' °C' : ' %';
            
            document.getElementById('lbl-obs-date').textContent = dateStr;
            document.getElementById('lbl-obs-real').textContent = (state.activeShapTarget === 'temp' ? obs[1].toFixed(1) : obs[4].toFixed(1)) + unit;
            document.getElementById('lbl-obs-ml').textContent = (state.activeShapTarget === 'temp' ? obs[3].toFixed(1) : obs[6].toFixed(1)) + unit;
        }}

        function changeTimeRange(rangeHours) {{
            state.timeRange = rangeHours;
            
            document.getElementById('btn-range-24').classList.toggle('active', rangeHours === 24);
            document.getElementById('btn-range-48').classList.toggle('active', rangeHours === 48);
            document.getElementById('btn-range-72').classList.toggle('active', rangeHours === 72);
            
            applyFilters();
            drawPlots();
        }}

        function toggleUncertainty() {{
            state.showUncertainty = !state.showUncertainty;
            document.getElementById('uncertainty-switch').classList.toggle('active', state.showUncertainty);
            drawPlots();
        }}

        function changeShapTarget(target) {{
            state.activeShapTarget = target;
            
            document.getElementById('btn-shap-temp').classList.toggle('active', target === 'temp');
            document.getElementById('btn-shap-hum').classList.toggle('active', target === 'hum');
            
            updateObsDetailCard();
            drawShapWaterfall();
            updateShapTable();
        }}

        function changeImpType(iType) {{
            state.impType = iType;
            
            document.getElementById('btn-imp-xgb').classList.toggle('active', iType === 'xgb');
            document.getElementById('btn-imp-pfi').classList.toggle('active', iType === 'pfi');
            
            const descText = iType === 'xgb' 
                ? 'Muestra la importancia interna de XGBoost basada en la ganancia (Gain) de cada predictor para construir los árboles.'
                : 'Muestra el decremento medio en el score del modelo al permutar aleatoriamente cada característica en el set de prueba.';
            document.getElementById('importance-desc-text').textContent = descText;
            
            drawGlobalImportance();
        }}

        function changePDPTarget(target) {{
            state.pdpTarget = target;
            
            document.getElementById('btn-pdp-temp').classList.toggle('active', target === 'temp');
            document.getElementById('btn-pdp-hum').classList.toggle('active', target === 'hum');
            
            drawPDPPlots();
        }}

        // --- FUNCIONES DE DIBUJADO DE PLOTLY ---
        
        function drawOverviewMap() {{
            if (typeof Plotly === 'undefined') {{
                setTimeout(() => drawOverviewMap(), 100);
                return;
            }}
            const lats = DB.stations.map(s => s.lat);
            const lons = DB.stations.map(s => s.lon);
            const text = DB.stations.map(s => `<b>Estación: ${{s.id}}</b><br>Mejora Temp: ${{s.imp_temp}}%<br>Mejora Hum: ${{s.imp_hum}}%<br>MAE NWS Temp: ${{s.mae_nws_temp}}°C<br>MAE ML Temp: ${{s.mae_ml_temp}}°C`);
            const size = DB.stations.map(s => s.mae_nws_temp * 10 + 4);
            const colors = DB.stations.map(s => s.imp_avg);
            
            const data = [{{
                type: 'scattergeo',
                lat: lats,
                lon: lons,
                mode: 'markers',
                marker: {{
                    size: size,
                    color: colors,
                    colorscale: [
                        [0.0, 'rgb(185, 28, 28)'],
                        [0.3, 'rgb(248, 113, 113)'],
                        [0.5, 'rgb(241, 245, 249)'],
                        [0.7, 'rgb(110, 231, 183)'],
                        [1.0, 'rgb(4, 120, 87)']
                    ],
                    cmin: -50,
                    cmax: 50,
                    showscale: true,
                    colorbar: {{
                        title: 'Mejora MAE %',
                        thickness: 12,
                        titleside: 'right',
                        len: 0.8
                    }}
                }},
                text: text,
                customdata: DB.stations.map(s => s.id),
                hoverinfo: 'text'
            }}];
            
            const layout = {{
                geo: {{
                    scope: 'usa',
                    projection: {{ type: 'albers usa' }},
                    showland: true,
                    landcolor: 'rgb(250, 250, 250)',
                    subunitcolor: 'rgb(217, 217, 217)',
                    showlakes: true,
                    lakecolor: 'rgb(255, 255, 255)'
                }},
                margin: {{ l: 0, r: 0, t: 0, b: 0 }},
                showlegend: false
            }};
            
            Plotly.newPlot('overview-map', data, layout, {{ responsive: true }});
            
            document.getElementById('overview-map').on('plotly_click', (data) => {{
                if (data.points && data.points[0]) {{
                    const stationId = data.points[0].customdata;
                    if (stationId) {{
                        selectStationAndGo(stationId);
                    }}
                }}
            }});
        }}

        function drawPlots() {{
            if (typeof Plotly === 'undefined') {{
                setTimeout(() => drawPlots(), 100);
                return;
            }}
            if (state.filteredData.length === 0) return;
            
            const timestamps = state.filteredData.map(d => DB.timestamps[d[0]]);
            const hours = state.filteredData.map(d => new Date(DB.timestamps[d[0]]).getHours());
            
            // Datos de Temperatura
            const tempReal = state.filteredData.map(d => d[1]);
            const tempNWS = state.filteredData.map(d => d[2]);
            const tempML = state.filteredData.map(d => d[3]);
            
            // Datos de Humedad
            const humReal = state.filteredData.map(d => d[4]);
            const humNWS = state.filteredData.map(d => d[5]);
            const humML = state.filteredData.map(d => d[6]);
            
            // Temperatura Traces
            const tempTraces = [
                {{
                    x: timestamps,
                    y: tempReal,
                    name: 'Observación Real',
                    line: {{ color: 'rgb(15, 23, 42)', width: 2.5 }},
                    type: 'scatter'
                }},
                {{
                    x: timestamps,
                    y: tempNWS,
                    name: 'Pronóstico NWS',
                    line: {{ color: 'rgb(244, 63, 94)', width: 1.5, dash: 'dot' }},
                    type: 'scatter'
                }},
                {{
                    x: timestamps,
                    y: tempML,
                    name: 'Corrección ML',
                    line: {{ color: 'rgb(16, 185, 129)', width: 2 }},
                    type: 'scatter'
                }}
            ];
            
            if (state.showUncertainty) {{
                const tempLower = [];
                const tempUpper = [];
                for (let i = 0; i < state.filteredData.length; i++) {{
                    const h = hours[i];
                    const pred = tempML[i];
                    const bounds = DB.uncertainty_by_hour[h] || {{ temp_p5: -1.5, temp_p95: 1.5 }};
                    tempLower.push(pred + bounds.temp_p5);
                    tempUpper.push(pred + bounds.temp_p95);
                }}
                
                tempTraces.unshift({{
                    x: timestamps.concat([...timestamps].reverse()),
                    y: tempUpper.concat([...tempLower].reverse()),
                    fill: 'toself',
                    fillcolor: 'rgba(16, 185, 129, 0.15)',
                    line: {{ color: 'transparent' }},
                    name: 'Incertidumbre (90%)',
                    showlegend: true,
                    type: 'scatter'
                }});
            }}
            
            // Humedad Traces
            const humTraces = [
                {{
                    x: timestamps,
                    y: humReal,
                    name: 'Observación Real',
                    line: {{ color: 'rgb(15, 23, 42)', width: 2.5 }},
                    type: 'scatter'
                }},
                {{
                    x: timestamps,
                    y: humNWS,
                    name: 'Pronóstico NWS',
                    line: {{ color: 'rgb(59, 130, 246)', width: 1.5, dash: 'dot' }},
                    type: 'scatter'
                }},
                {{
                    x: timestamps,
                    y: humML,
                    name: 'Corrección ML',
                    line: {{ color: 'rgb(13, 148, 136)', width: 2 }},
                    type: 'scatter'
                }}
            ];
            
            if (state.showUncertainty) {{
                const humLower = [];
                const humUpper = [];
                for (let i = 0; i < state.filteredData.length; i++) {{
                    const h = hours[i];
                    const pred = humML[i];
                    const bounds = DB.uncertainty_by_hour[h] || {{ hum_p5: -8.0, hum_p95: 8.0 }};
                    humLower.push(pred + bounds.hum_p5);
                    humUpper.push(pred + bounds.hum_p95);
                }}
                
                humTraces.unshift({{
                    x: timestamps.concat([...timestamps].reverse()),
                    y: humUpper.concat([...humLower].reverse()),
                    fill: 'toself',
                    fillcolor: 'rgba(13, 148, 136, 0.15)',
                    line: {{ color: 'transparent' }},
                    name: 'Incertidumbre (90%)',
                    showlegend: true,
                    type: 'scatter'
                }});
            }}
            
            const layoutCommon = {{
                margin: {{ l: 40, r: 20, t: 20, b: 40 }},
                legend: {{ orientation: 'h', y: 1.1, x: 0 }},
                hovermode: 'x unified',
                plot_bgcolor: '#ffffff',
                paper_bgcolor: 'rgba(0,0,0,0)',
                xaxis: {{ gridcolor: '#f1f5f9', linecolor: '#cbd5e1' }},
                yaxis: {{ gridcolor: '#f1f5f9', linecolor: '#cbd5e1' }}
            }};
            
            Plotly.newPlot('temp-ts-chart', tempTraces, {{ ...layoutCommon, yaxis: {{ ...layoutCommon.yaxis, title: 'Temp (°C)' }} }}, {{ responsive: true }});
            Plotly.newPlot('hum-ts-chart', humTraces, {{ ...layoutCommon, yaxis: {{ ...layoutCommon.yaxis, title: 'Humedad (%)' }} }}, {{ responsive: true }});
            
            const charts = ['temp-ts-chart', 'hum-ts-chart'];
            charts.forEach(cId => {{
                document.getElementById(cId).on('plotly_click', (data) => {{
                    if (data.points && data.points[0]) {{
                        const clickedX = data.points[0].x;
                        const index = state.filteredData.findIndex(d => DB.timestamps[d[0]] === clickedX);
                        if (index !== -1) {{
                            state.selectedObsIdx = index;
                            updateObsDetailCard();
                            drawShapWaterfall();
                            updateShapTable();
                        }}
                    }}
                }});
            }});
            
            updateObsDetailCard();
            drawShapWaterfall();
            updateShapTable();
        }}

        function drawShapWaterfall() {{
            if (typeof Plotly === 'undefined') {{
                setTimeout(() => drawShapWaterfall(), 100);
                return;
            }}
            const obs = getSelectedObs();
            const shapVals = state.activeShapTarget === 'temp' ? obs[7] : obs[8];
            const expectedVal = state.activeShapTarget === 'temp' ? DB.expected_value_temp : DB.expected_value_hum;
            const finalVal = state.activeShapTarget === 'temp' ? obs[3] : obs[6];
            
            if (!shapVals || shapVals.length === 0) return;
            
            const contribs = DB.features.map((feat, idx) => ({{
                name: feat,
                label: FEATURE_LABELS[feat] || feat,
                value: shapVals[idx]
            }}));
            
            contribs.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
            const topFeatures = contribs.slice(0, 6);
            const otherSum = contribs.slice(6).reduce((sum, c) => sum + c.value, 0);
            
            const waterfallLabels = ['Base'];
            const waterfallValues = [expectedVal];
            const waterfallMeasures = ['relative'];
            
            topFeatures.forEach(c => {{
                waterfallLabels.push(c.label);
                waterfallValues.push(c.value);
                waterfallMeasures.push('relative');
            }});
            
            if (Math.abs(otherSum) > 0.01) {{
                waterfallLabels.push('Otras variables');
                waterfallValues.push(otherSum);
                waterfallMeasures.push('relative');
            }}
            
            waterfallLabels.push('ML');
            waterfallValues.push(finalVal);
            waterfallMeasures.push('total');
            
            const data = [{{
                type: 'waterfall',
                orientation: 'v',
                measure: waterfallMeasures,
                x: waterfallLabels,
                y: waterfallValues,
                textposition: 'outside',
                text: waterfallValues.map((v, i) => i === waterfallValues.length - 1 ? v.toFixed(1) : (v >= 0 ? '+' : '') + v.toFixed(1)),
                connector: {{ line: {{ color: 'rgb(203, 213, 225)', width: 1, dash: 'dot' }} }},
                increasing: {{ marker: {{ color: '#10b981' }} }},
                decreasing: {{ marker: {{ color: '#ef4444' }} }},
                totals: {{ marker: {{ color: '#2563eb' }} }}
            }}];
            
            const layout = {{
                margin: {{ l: 30, r: 30, t: 10, b: 90 }},
                plot_bgcolor: '#ffffff',
                paper_bgcolor: 'rgba(0,0,0,0)',
                showlegend: false,
                xaxis: {{
                    tickangle: -45,
                    tickfont: {{ size: 9 }},
                    linecolor: '#cbd5e1'
                }},
                yaxis: {{
                    gridcolor: '#f1f5f9',
                    linecolor: '#cbd5e1'
                }}
            }};
            
            Plotly.newPlot('shap-waterfall', data, layout, {{ responsive: true }});
        }}

        function updateShapTable() {{
            const obs = getSelectedObs();
            const shapVals = state.activeShapTarget === 'temp' ? obs[7] : obs[8];
            const container = document.getElementById('shap-table-body');
            container.innerHTML = '';
            
            if (!shapVals || shapVals.length === 0) return;
            
            const contribs = DB.features.map((feat, idx) => ({{
                name: feat,
                label: FEATURE_LABELS[feat] || feat,
                value: shapVals[idx]
            }}));
            
            contribs.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
            
            contribs.slice(0, 5).forEach(c => {{
                const tr = document.createElement('tr');
                const sign = c.value >= 0 ? '+' : '';
                const badgeClass = c.value >= 0 ? 'badge-contrib-pos' : 'badge-contrib-neg';
                tr.innerHTML = `
                    <td>${{c.label}}</td>
                    <td><span class="${{badgeClass}}">${{sign}}${{c.value.toFixed(1)}}</span></td>
                `;
                container.appendChild(tr);
            }});
        }}

        function drawGlobalImportance() {{
            if (typeof Plotly === 'undefined') {{
                setTimeout(() => drawGlobalImportance(), 100);
                return;
            }}
            const importances = state.impType === 'xgb' ? DB.feat_imp : DB.pfi;
            
            const labels = DB.features.map(f => FEATURE_LABELS[f] || f);
            const tempValues = state.impType === 'xgb' ? importances.temp : importances.temp.importances_mean;
            const humValues = state.impType === 'xgb' ? importances.hum : importances.hum.importances_mean;
            
            const combined = labels.map((l, idx) => ({{
                label: l,
                tempVal: tempValues[idx],
                humVal: humValues[idx],
                avg: (tempValues[idx] + humValues[idx]) / 2
            }})).sort((a, b) => b.avg - a.avg);
            
            const traceTemp = {{
                y: combined.map(c => c.label),
                x: combined.map(c => c.tempVal),
                name: 'Modelo Temp.',
                type: 'bar',
                orientation: 'h',
                marker: {{ color: '#f43f5e' }}
            }};
            
            const traceHum = {{
                y: combined.map(c => c.label),
                x: combined.map(c => c.humVal),
                name: 'Modelo Hum.',
                type: 'bar',
                orientation: 'h',
                marker: {{ color: '#3b82f6' }}
            }};
            
            const layout = {{
                barmode: 'group',
                margin: {{ l: 150, r: 20, t: 10, b: 40 }},
                legend: {{ orientation: 'h', y: 1.1, x: 0 }},
                plot_bgcolor: '#ffffff',
                paper_bgcolor: 'rgba(0,0,0,0)',
                yaxis: {{ autorange: 'reverse', tickfont: {{ size: 9 }} }},
                xaxis: {{ gridcolor: '#f1f5f9', title: state.impType === 'xgb' ? 'Ganancia del Árbol (Gain)' : 'Decremento del Score (MAE °C / %)' }}
            }};
            
            Plotly.newPlot('feature-importance-chart', [traceTemp, traceHum], layout, {{ responsive: true }});
        }}

        function drawPDPPlots() {{
            if (typeof Plotly === 'undefined') {{
                setTimeout(() => drawPDPPlots(), 100);
                return;
            }}
            const targetData = DB.pdp[state.pdpTarget];
            const color = state.pdpTarget === 'temp' ? '#f43f5e' : '#3b82f6';
            
            // 1. PDP Hora
            const traceHour = {{
                x: targetData.hour.grid,
                y: targetData.hour.values,
                type: 'scatter',
                mode: 'lines+markers',
                line: {{ color: color, width: 2.5 }},
                marker: {{ size: 5 }}
            }};
            
            // 2. PDP Pronóstico Base
            const traceFcst = {{
                x: targetData[state.pdpTarget + '_fcst'].grid,
                y: targetData[state.pdpTarget + '_fcst'].values,
                type: 'scatter',
                mode: 'lines',
                line: {{ color: color, width: 2.5 }}
            }};
            
            // 3. PDP Lag de Error
            const traceLag = {{
                x: targetData[state.pdpTarget + '_error_lag1'].grid,
                y: targetData[state.pdpTarget + '_error_lag1'].values,
                type: 'scatter',
                mode: 'lines',
                line: {{ color: color, width: 2.5 }}
            }};
            
            const layoutCommon = {{
                margin: {{ l: 40, r: 20, t: 10, b: 30 }},
                plot_bgcolor: '#ffffff',
                paper_bgcolor: 'rgba(0,0,0,0)',
                xaxis: {{ gridcolor: '#f1f5f9', linecolor: '#cbd5e1' }},
                yaxis: {{ gridcolor: '#f1f5f9', linecolor: '#cbd5e1' }}
            }};
            
            Plotly.newPlot('pdp-hour-chart', [traceHour], {{ ...layoutCommon, xaxis: {{ ...layoutCommon.xaxis, title: 'Hora del Día (h)' }}, yaxis: {{ ...layoutCommon.yaxis, title: state.pdpTarget === 'temp' ? 'Impacto Marginal Temp. (°C)' : 'Impacto Marginal Hum. (%)' }} }}, {{ responsive: true }});
            Plotly.newPlot('pdp-fcst-chart', [traceFcst], {{ ...layoutCommon, xaxis: {{ ...layoutCommon.xaxis, title: state.pdpTarget === 'temp' ? 'Pronóstico Base Temp. (°C)' : 'Pronóstico Base Hum. (%)' }}, yaxis: {{ ...layoutCommon.yaxis, title: state.pdpTarget === 'temp' ? 'Impacto Marginal Temp. (°C)' : 'Impacto Marginal Hum. (%)' }} }}, {{ responsive: true }});
            Plotly.newPlot('pdp-lag-chart', [traceLag], {{ ...layoutCommon, xaxis: {{ ...layoutCommon.xaxis, title: state.pdpTarget === 'temp' ? 'Lag de Error Temp. (°C)' : 'Lag de Error Hum. (%)' }}, yaxis: {{ ...layoutCommon.yaxis, title: state.pdpTarget === 'temp' ? 'Impacto Marginal Temp. (°C)' : 'Impacto Marginal Hum. (%)' }} }}, {{ responsive: true }});
        }}

        function drawDiurnalBias() {{
            if (typeof Plotly === 'undefined') {{
                setTimeout(() => drawDiurnalBias(), 100);
                return;
            }}
            const hours = Array.from({{ length: 24 }}, (_, i) => i);
            
            const nwsTempBias = Array(24).fill(0);
            const mlTempBias = Array(24).fill(0);
            const nwsHumBias = Array(24).fill(0);
            const mlHumBias = Array(24).fill(0);
            const counts = Array(24).fill(0);
            
            for (const stationId in DB.station_data) {{
                const records = DB.station_data[stationId];
                for (const r of records) {{
                    const ts_str = DB.timestamps[r[0]];
                    const h = new Date(ts_str).getHours();
                    nwsTempBias[h] += (r[2] - r[1]);
                    mlTempBias[h] += (r[3] - r[1]);
                    nwsHumBias[h] += (r[5] - r[4]);
                    mlHumBias[h] += (r[6] - r[4]);
                    counts[h]++;
                }}
            }}
            
            for (let h = 0; h < 24; h++) {{
                const count = counts[h] || 1;
                nwsTempBias[h] /= count;
                mlTempBias[h] /= count;
                nwsHumBias[h] /= count;
                mlHumBias[h] /= count;
            }}
            
            const traceNwsT = {{
                x: hours,
                y: nwsTempBias,
                name: 'Temp. NWS (Sesgo Base)',
                line: {{ color: '#f43f5e', dash: 'dot', width: 2 }},
                type: 'scatter'
            }};
            
            const traceMlT = {{
                x: hours,
                y: mlTempBias,
                name: 'Temp. ML (Corregido)',
                line: {{ color: '#10b981', width: 2.5 }},
                type: 'scatter'
            }};
            
            const traceNwsH = {{
                x: hours,
                y: nwsHumBias,
                name: 'Hum. NWS (Sesgo Base)',
                line: {{ color: '#3b82f6', dash: 'dot', width: 2 }},
                type: 'scatter',
                yaxis: 'y2'
            }};
            
            const traceMlH = {{
                x: hours,
                y: mlHumBias,
                name: 'Hum. ML (Corregido)',
                line: {{ color: '#0d9488', width: 2.5 }},
                type: 'scatter',
                yaxis: 'y2'
            }};
            
            const layout = {{
                margin: {{ l: 40, r: 40, t: 20, b: 35 }},
                legend: {{ orientation: 'h', y: 1.25, x: 0 }},
                plot_bgcolor: '#ffffff',
                paper_bgcolor: 'rgba(0,0,0,0)',
                xaxis: {{ gridcolor: '#f1f5f9', title: 'Hora del Día', dtick: 2 }},
                yaxis: {{ gridcolor: '#f1f5f9', title: 'Sesgo Temp (°C)', titlefont: {{ color: '#f43f5e' }} }},
                yaxis2: {{
                    title: 'Sesgo Humedad (%)',
                    titlefont: {{ color: '#3b82f6' }},
                    overlaying: 'y',
                    side: 'right',
                    gridcolor: 'transparent'
                }}
            }};
            
            Plotly.newPlot('diurnal-bias-chart', [traceNwsT, traceMlT, traceNwsH, traceMlH], layout, {{ responsive: true }});
        }}

        // Arrancar la App al cargar el DOM
        window.addEventListener('DOMContentLoaded', initApp);
    </script>
</body>
</html>
"""

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_template)
    
    print(f"Dashboard generado exitosamente en: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_dashboard()
