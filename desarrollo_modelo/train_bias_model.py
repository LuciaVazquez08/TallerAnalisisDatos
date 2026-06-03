import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import optuna
import matplotlib.pyplot as plt
import joblib

# Configuración Global
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

def blocked_timeseries_split(n_samples, n_splits=5):
    """
    Divide los datos en bloques temporales contiguos asumiendo orden cronológico.
    Evita que el modelo aprenda de datos aleatorios y garantiza que la validación
    sea siempre posterior al entrenamiento.
    
    Args:
        n_samples (int): Número total de registros.
        n_splits (int): Cantidad de folds para validación cruzada.
    """
    k_fold_size = n_samples // (n_splits + 1)
    indices = np.arange(n_samples)
    
    for i in range(n_splits):
        start_train = 0
        end_train = (i + 1) * k_fold_size
        start_val = end_train
        end_val = (i + 2) * k_fold_size if i < n_splits - 1 else n_samples
        
        yield indices[start_train:end_train], indices[start_val:end_val]

def objective(trial, X, y, cv_splits):
    """
    Función objetivo para Optuna. Define el espacio de búsqueda y evalúa
    el rendimiento (MAE) para un conjunto de hiperparámetros.
    """
    # Espacio de búsqueda de hiperparámetros
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'tree_method': 'hist',  # Optimización para datasets grandes
        'random_state': 42
    }
    
    scores = []
    
    # Evaluación Cross-Validation (Blocked)
    for train_idx, val_idx in cv_splits:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Envoltorio MultiOutput: entrena un XGBoost por cada variable objetivo (Temp y Hum)
        model = MultiOutputRegressor(xgb.XGBRegressor(**params))
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        scores.append(mae)
        
    return np.mean(scores)

def run_training():
    """
    Orquestador principal del entrenamiento: Carga datos, optimiza hiperparámetros,
    entrena el modelo final y evalúa contra el set de TEST fuera de tiempo.
    """
    if not os.path.exists(FEATURE_PATH):
        print(f"Error: No se encuentra {FEATURE_PATH}. Corre primero model_data_prep.py")
        return

    # Carga y ordenamiento temporal estricto
    df = pd.read_parquet(FEATURE_PATH)
    df = df.sort_values('obs_timestamp')
    
    X = df[FEATURES]
    y = df[TARGETS]
    
    # 1. Separación de Datos (Train-Val / Test)
    # Usamos el último 15% como TEST FINAL (~7 días en un dataset de 48 días)
    split_idx = int(len(df) * 0.85)
    df_train_all = df.iloc[:split_idx]
    X_test = df[FEATURES].iloc[split_idx:]
    y_test = df[TARGETS].iloc[split_idx:]
    
    # Aplicar ventana de Look-back Óptima (21 días)
    # Filtramos para usar solo los 21 días previos al set de test
    test_start_time = df.iloc[split_idx]['obs_timestamp']
    train_start_time = test_start_time - pd.Timedelta(days=21)
    
    df_train_filtered = df_train_all[df_train_all['obs_timestamp'] >= train_start_time].copy()
    X_train_full = df_train_filtered[FEATURES]
    y_train_full = df_train_filtered[TARGETS]
    
    print(f"Pool total de entreno: {len(df_train_all)} | Ventana 21d: {len(X_train_full)} | Test Final: {len(X_test)}")
    print(f"Rango Entreno: {df_train_filtered['obs_timestamp'].min()} a {df_train_filtered['obs_timestamp'].max()}")
    
    # 2. Generación de Splits Temporales
    cv_splits = list(blocked_timeseries_split(len(X_train_full), n_splits=3))
    
    # 3. Optimización con Optuna (Búsqueda Bayesiana)
    print("\n--- Iniciando Optimización Bayesiana (Optuna) ---")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_full, y_train_full, cv_splits), n_trials=20)
    
    print(f"\nMejor Score (MAE): {study.best_value:.4f}")
    print("Mejores Hiperparámetros:", study.best_params)
    
    # 4. Entrenamiento del Modelo Final
    # Usamos los mejores parámetros sobre TODO el conjunto de entrenamiento/validación
    best_params = study.best_params
    best_params['tree_method'] = 'hist'
    final_model = MultiOutputRegressor(xgb.XGBRegressor(**best_params))
    final_model.fit(X_train_full, y_train_full)
    
    # Guardado persistente del modelo
    joblib.dump(final_model, "bias_correction_model.pkl")
    print("\nModelo guardado exitosamente como bias_correction_model.pkl")
    
    # 5. Evaluación de Desempeño Final
    preds_test = final_model.predict(X_test)
    
    # Métricas contra realidad
    mae_temp = mean_absolute_error(y_test['temp_real'], preds_test[:, 0])
    mae_hum = mean_absolute_error(y_test['hum_real'], preds_test[:, 1])
    
    # Métricas del error original de la NWS (sin corregir)
    mae_temp_orig = mean_absolute_error(y_test['temp_real'], X_test['temp_fcst'])
    mae_hum_orig = mean_absolute_error(y_test['hum_real'], X_test['hum_fcst'])
    
    print(f"\n--- REPORTE FINAL DE DESEMPEÑO (SET DE TEST) ---")
    print(f"Temperatura MAE: {mae_temp:.4f} (Mejora: {((mae_temp_orig - mae_temp)/mae_temp_orig)*100:.2f}%)")
    print(f"Humedad MAE: {mae_hum:.4f} (Mejora: {((mae_hum_orig - mae_hum)/mae_hum_orig)*100:.2f}%)")

if __name__ == "__main__":
    run_training()
