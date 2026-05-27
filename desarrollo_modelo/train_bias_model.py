import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import optuna
import matplotlib.pyplot as plt
import joblib

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

def blocked_timeseries_split(n_samples, n_splits=5):
    """
    Divide los datos en bloques temporales contiguos asumiendo que ya están ordenados.
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
    # Hiperparámetros a optimizar
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'tree_method': 'hist',  # Rápido
        'random_state': 42
    }
    
    scores = []
    
    for train_idx, val_idx in cv_splits:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Envoltorio MultiOutput para XGBoost
        model = MultiOutputRegressor(xgb.XGBRegressor(**params))
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        # Evaluamos MAE promedio entre ambas variables
        mae = mean_absolute_error(y_val, preds)
        scores.append(mae)
        
    return np.mean(scores)

def run_training():
    if not os.path.exists(FEATURE_PATH):
        print(f"Error: No se encuentra {FEATURE_PATH}. Corre primero model_data_prep.py")
        return

    df = pd.read_parquet(FEATURE_PATH)
    df = df.sort_values('obs_timestamp')
    
    X = df[FEATURES]
    y = df[TARGETS]
    
    # Separar un set de TEST final (último 15% de los datos)
    split_idx = int(len(df) * 0.85)
    X_train_full, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_full, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Entrenamiento/Val: {len(X_train_full)} | Test Final: {len(X_test)}")
    
    # Generar splits para validación cruzada sobre el set de entrenamiento
    cv_splits = list(blocked_timeseries_split(len(X_train_full), n_splits=3))
    
    # Optimización con Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_full, y_train_full, cv_splits), n_trials=20)
    
    print("Mejores hiperparámetros:", study.best_params)
    
    # Entrenar modelo final con mejores parámetros
    best_params = study.best_params
    best_params['tree_method'] = 'hist'
    final_model = MultiOutputRegressor(xgb.XGBRegressor(**best_params))
    final_model.fit(X_train_full, y_train_full)
    
    # Guardar modelo
    joblib.dump(final_model, "bias_correction_model.pkl")
    print("Modelo guardado como bias_correction_model.pkl")
    
    # Evaluación en TEST
    preds_test = final_model.predict(X_test)
    mae_temp = mean_absolute_error(y_test['temp_real'], preds_test[:, 0])
    mae_hum = mean_absolute_error(y_test['hum_real'], preds_test[:, 1])
    
    # Comparación con el error original (NWS sin corregir)
    mae_temp_orig = mean_absolute_error(y_test['temp_real'], X_test['temp_fcst'])
    mae_hum_orig = mean_absolute_error(y_test['hum_real'], X_test['hum_fcst'])
    
    print(f"\n--- RESULTADOS EN SET DE TEST ---")
    print(f"Temperatura MAE: {mae_temp:.4f} (Original: {mae_temp_orig:.4f})")
    print(f"Humedad MAE: {mae_hum:.4f} (Original: {mae_hum_orig:.4f})")
    print(f"Mejora Temp: {((mae_temp_orig - mae_temp)/mae_temp_orig)*100:.2f}%")
    print(f"Mejora Hum: {((mae_hum_orig - mae_hum)/mae_hum_orig)*100:.2f}%")

if __name__ == "__main__":
    run_training()
