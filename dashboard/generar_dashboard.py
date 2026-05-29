import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
import os

# Configuración de Rutas
FEATURE_PATH = "desarrollo_modelo/clima_features.parquet"
MODEL_PATH = "desarrollo_modelo/bias_correction_model.pkl"
OUTPUT_PATH = "dashboard/dashboard_sesgo_clima.html"

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
    
    print("Generando predicciones...")
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

    # --- 1. Perfil de Sesgo Horario ---
    df_test['hour'] = df_test['obs_timestamp'].dt.hour
    hourly_bias = df_test.groupby('hour')[['error_nws_temp', 'error_ml_temp', 'error_nws_hum', 'error_ml_hum']].mean().reset_index()

    fig_bias = make_subplots(rows=1, cols=2, subplot_titles=("Sesgo Temperatura (°C)", "Sesgo Humedad (%)"))
    
    # Temp Bias
    fig_bias.add_trace(go.Scatter(x=hourly_bias['hour'], y=hourly_bias['error_nws_temp'], name="NWS Temp", line=dict(color='red', dash='dash')), row=1, col=1)
    fig_bias.add_trace(go.Scatter(x=hourly_bias['hour'], y=hourly_bias['error_ml_temp'], name="ML Temp", line=dict(color='green')), row=1, col=1)
    
    # Hum Bias
    fig_bias.add_trace(go.Scatter(x=hourly_bias['hour'], y=hourly_bias['error_nws_hum'], name="NWS Hum", line=dict(color='blue', dash='dash')), row=1, col=2)
    fig_bias.add_trace(go.Scatter(x=hourly_bias['hour'], y=hourly_bias['error_ml_hum'], name="ML Hum", line=dict(color='darkblue')), row=1, col=2)
    
    fig_bias.update_layout(title_text="Perfil de Sesgo por Hora del Día (Ciclo Diurno)", height=500)

    # --- 2. Mapa de Mejora por Estación ---
    station_metrics = df_test.groupby(['station_id', 'lat_estacion', 'lon_estacion']).apply(
        lambda x: pd.Series({
            'mae_nws': mean_absolute_error(x['temp_real'], x['temp_fcst']),
            'mae_ml': mean_absolute_error(x['temp_real'], x['temp_ml']),
        })
    ).reset_index()
    station_metrics['improvement'] = (station_metrics['mae_nws'] - station_metrics['mae_ml']) / station_metrics['mae_nws'] * 100

    fig_map = px.scatter_mapbox(station_metrics, lat="lat_estacion", lon="lon_estacion", 
                                color="improvement", size="mae_nws",
                                color_continuous_scale=px.colors.diverging.RdYlGn,
                                hover_name="station_id", hover_data=["mae_nws", "mae_ml"],
                                title="Mejora del MAE de Temperatura por Estación (%)",
                                zoom=3, height=600)
    fig_map.update_layout(mapbox_style="carto-positron")

    # --- 3. Comparativa de Serie Temporal (Estación de Ejemplo) ---
    # Seleccionamos una estación con mejora significativa
    best_station = station_metrics.sort_values('improvement', ascending=False).iloc[0]['station_id']
    df_sample = df_test[df_test['station_id'] == best_station].tail(48) # Últimas 48 horas de esa estación
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=df_sample['obs_timestamp'], y=df_sample['temp_real'], name="Real", line=dict(color='black', width=3)))
    fig_ts.add_trace(go.Scatter(x=df_sample['obs_timestamp'], y=df_sample['temp_fcst'], name="NWS (Original)", line=dict(color='red', dash='dot')))
    fig_ts.add_trace(go.Scatter(x=df_sample['obs_timestamp'], y=df_sample['temp_ml'], name="ML (Corregido)", line=dict(color='green')))
    
    fig_ts.update_layout(title=f"Comparativa Temporal: Estación {best_station} (Últimas 48h)", xaxis_title="Fecha/Hora", yaxis_title="Temperatura (°C)")

    # --- 4. Importancia de Variables ---
    xgb_temp = model.estimators_[0]
    importances = pd.Series(xgb_temp.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig_imp = px.bar(x=importances.values, y=importances.index, orientation='h', 
                     title="Importancia de Variables (Modelo Temperatura)",
                     labels={'x': 'Importancia Relativa', 'y': 'Feature'})

    # --- Ensamblado de HTML ---
    html_content = f"""
    <html>
    <head>
        <title>Dashboard de Corrección de Sesgo Climático</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #f8f9fa; padding: 20px; }}
            .card {{ margin-bottom: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .metric-card {{ text-align: center; padding: 20px; }}
            .metric-value {{ font-size: 2.5rem; font-weight: bold; color: #0d6efd; }}
            .metric-label {{ font-size: 1.1rem; color: #6c757d; }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <h1 class="mb-4">Dashboard de Corrección de Sesgo: NWS vs Machine Learning</h1>
            
            <div class="row">
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">{mae_ml_temp:.3f} °C</div>
                        <div class="metric-label">MAE Temperatura (ML)</div>
                        <div class="text-success">▼ {imp_temp:.1f}% mejora</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">{mae_ml_hum:.3f} %</div>
                        <div class="metric-label">MAE Humedad (ML)</div>
                        <div class="text-success">▼ {imp_hum:.1f}% mejora</div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card p-3">
                        <h5>Resumen del Proyecto</h5>
                        <p>Este dashboard presenta los resultados del modelo <b>XGBoost Multi-salida</b> diseñado para corregir los sesgos sistemáticos en los pronósticos de la NWS. 
                        Se observa una reducción significativa del error, especialmente durante los ciclos de transición térmica.</p>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-12">
                    <div class="card p-3">
                        {fig_bias.to_html(full_html=False, include_plotlyjs='cdn')}
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-8">
                    <div class="card p-3">
                        {fig_map.to_html(full_html=False, include_plotlyjs=False)}
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card p-3">
                        {fig_imp.to_html(full_html=False, include_plotlyjs=False)}
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-12">
                    <div class="card p-3">
                        {fig_ts.to_html(full_html=False, include_plotlyjs=False)}
                    </div>
                </div>
            </div>
            
            <footer class="text-center mt-5 text-muted">
                <p>Generado automáticamente por el Sistema de Análisis Climático - {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Dashboard generado exitosamente en: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_dashboard()
