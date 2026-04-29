
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def diagnostic():
    df = pd.read_parquet("local_cleanup/clima_unificado.parquet")
    
    # Cleaning wind_speed_fcst (as done in the notebook)
    def clean_wind(val):
        if isinstance(val, str) and "mph" in val:
            return float(val.replace(" mph", ""))
        try:
            return float(val)
        except:
            return np.nan

    df["wind_speed_fcst_raw"] = df["wind_speed_fcst"].apply(clean_wind)
    df["wind_speed_real"] = pd.to_numeric(df["wind_speed_real"], errors='coerce')
    
    # Drop NaNs
    df = df.dropna(subset=["wind_speed_fcst_raw", "wind_speed_real"])
    
    # Calculate current error
    df["err_wind_raw"] = df["wind_speed_fcst_raw"] - df["wind_speed_real"]
    
    # Calculate corrected error
    df["wind_speed_fcst_corr"] = df["wind_speed_fcst_raw"] * 1.60934
    df["err_wind_corr"] = df["wind_speed_fcst_corr"] - df["wind_speed_real"]
    
    print("--- Estadísticas de Error de Viento ---")
    print(f"Sesgo (Bias) Original: {df['err_wind_raw'].mean():.4f}")
    print(f"MAE Original: {df['err_wind_raw'].abs().mean():.4f}")
    print(f"Sesgo (Bias) Corregido (x1.61): {df['err_wind_corr'].mean():.4f}")
    print(f"MAE Corregido (x1.61): {df['err_wind_corr'].abs().mean():.4f}")
    
    # Correlation analysis
    corr_raw = df[["wind_speed_real", "err_wind_raw"]].corr().iloc[0,1]
    corr_corr = df[["wind_speed_real", "err_wind_corr"]].corr().iloc[0,1]
    
    print(f"\nCorrelación real vs error (Original): {corr_raw:.4f}")
    print(f"Correlación real vs error (Corregido): {corr_corr:.4f}")
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.regplot(data=df.sample(min(2000, len(df))), x="wind_speed_real", y="err_wind_raw", scatter_kws={'alpha':0.3})
    plt.title("Error Original vs Real")
    plt.xlabel("Viento Real (km/h)")
    plt.ylabel("Error (fcst - real)")
    
    plt.subplot(1, 2, 2)
    sns.regplot(data=df.sample(min(2000, len(df))), x="wind_speed_real", y="err_wind_corr", scatter_kws={'alpha':0.3})
    plt.title("Error Corregido (x1.61) vs Real")
    plt.xlabel("Viento Real (km/h)")
    plt.ylabel("Error (fcst_corr - real)")
    
    plt.tight_layout()
    plt.savefig("wind_diagnostic.png")
    print("\nDiagnóstico guardado en 'wind_diagnostic.png'")

if __name__ == "__main__":
    diagnostic()
