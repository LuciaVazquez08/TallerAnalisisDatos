import pandas as pd
import numpy as np
import os

INPUT_PARQUET = "local_cleanup/clima_unificado.parquet"

def diagnose():
    if not os.path.exists(INPUT_PARQUET):
        # Intentar ruta relativa si se ejecuta desde local_cleanup
        if os.path.exists("clima_unificado.parquet"):
            df = pd.read_parquet("clima_unificado.parquet")
        else:
            print("No se encontró clima_unificado.parquet")
            return
    else:
        df = pd.read_parquet(INPUT_PARQUET)

    print(f"--- Diagnóstico de Diferencia de Temperatura ---")
    print(f"Total de registros: {len(df)}")

    # 1. Filtrar nulos
    temp_df = df.dropna(subset=['temp_real', 'temp_fcst']).copy()
    print(f"Registros con ambos datos: {len(temp_df)}")

    if len(temp_df) == 0:
        print("No hay datos suficientes para comparar.")
        return

    # 2. Métricas Básicas
    temp_df['error'] = temp_df['temp_real'] - temp_df['temp_fcst']
    mae = temp_df['error'].abs().mean()
    bias = temp_df['error'].mean()
    rmse = np.sqrt((temp_df['error']**2).mean())

    print(f"\nMAE (Error Absoluto Medio): {mae:.4f} °C")
    print(f"BIAS (Error Medio - indica si siempre es mayor o menor): {bias:.4f} °C")
    print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.4f} °C")

    # 3. Verificar conversión de unidades (F vs C)
    # Si olvidamos convertir, el error sería enorme (ej: 20C vs 68F)
    print(f"\nEstadísticas descriptivas:")
    print(temp_df[['temp_real', 'temp_fcst']].describe())

    # 4. Análisis por Hora del Día (Check Timezone)
    # Si hay un desfase horario, el error será mayor en las horas de transición (amanecer/atardecer)
    temp_df['hour'] = temp_df['obs_timestamp'].dt.hour
    hourly_error = temp_df.groupby('hour')['error'].mean()
    print("\nError medio por hora del día (UTC):")
    print(hourly_error)

    # 5. Análisis por Estación (Check Bad Sensors/Locations)
    temp_df['abs_error'] = temp_df['error'].abs()
    station_error = temp_df.groupby('station_id')['abs_error'].mean().sort_values(ascending=False)
    print("\nTop 5 estaciones con MAYOR error (MAE):")
    print(station_error.head(5))

    # 6. Correlación
    correlation = temp_df['temp_real'].corr(temp_df['temp_fcst'])
    print(f"\nCorrelación de Pearson: {correlation:.4f}")

    # 7. Detección de posibles duplicados temporales que afecten el merge
    # (Ya lo controlamos en unify_data, pero check extra)
    
    # 8. Ejemplo de un registro
    print("\nEjemplo de registro individual:")
    print(temp_df[['station_id', 'obs_timestamp', 'temp_real', 'temp_fcst', 'error']].head(1))

if __name__ == "__main__":
    diagnose()
