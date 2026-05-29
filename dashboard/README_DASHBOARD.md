# Dashboard de Corrección de Sesgo Climático

Este directorio contiene los archivos necesarios para generar y visualizar el dashboard interactivo del proyecto.

## Contenido
- `generar_dashboard.py`: Script de Python que procesa los datos, carga el modelo y genera el reporte interactivo.
- `dashboard_sesgo_clima.html`: El dashboard final generado (archivo estático, abrir en cualquier navegador).
- `README_DASHBOARD.md`: Este archivo.

## Instrucciones de Uso
Para actualizar el dashboard con nuevos datos o cambios en el modelo:
1. Asegúrate de tener las dependencias instaladas:
   ```bash
   pip install plotly xgboost pandas joblib scikit-learn pyarrow
   ```
2. Ejecuta el script de generación:
   ```bash
   python dashboard/generar_dashboard.py
   ```
3. Abre el archivo `dashboard/dashboard_sesgo_clima.html` en tu navegador preferido.

## Secciones del Dashboard
1. **Métricas de Resumen**: Comparativa global del MAE (Mean Absolute Error) entre el pronóstico original de la NWS y la corrección aplicada por el modelo de ML.
2. **Perfil de Sesgo por Hora**: Visualización del ciclo diurno del error, mostrando cómo el modelo de ML estabiliza el sesgo especialmente en las transiciones de temperatura.
3. **Mapa de Mejora por Estación**: Distribución geográfica del desempeño del modelo, permitiendo identificar zonas donde la corrección es más efectiva.
4. **Importancia de Variables**: Factores que más influyen en la corrección realizada por el modelo XGBoost.
5. **Serie Temporal de Ejemplo**: Comparativa detallada de las últimas 48 horas para la estación con mejor desempeño.
