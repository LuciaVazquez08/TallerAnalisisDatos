# Reporte de Desarrollo: Modelo de Corrección de Sesgo Diurno

## 1. Introducción
Este documento detalla el desarrollo de un modelo de Machine Learning diseñado para corregir los sesgos sistemáticos en los pronósticos de temperatura y humedad de la NWS (National Weather Service). Se observó un patrón de error ligado al ciclo diurno, especialmente crítico en las transiciones de amanecer y atardecer, donde el modelo base tiende a subestimar o sobreestimar el calentamiento/enfriamiento de la capa límite.

## 2. Metodología

### 2.1 Ingeniería de Features
Para capturar la naturaleza física y temporal del sesgo, se implementaron las siguientes variables:
*   **Ciclo Diurno:** Transformación de la hora del día en componentes seno y coseno (`hour_sin`, `hour_cos`) para representar la continuidad circular del tiempo.
*   **Diferencial Térmico:** Cambio en el pronóstico respecto a la hora anterior (`temp_fcst_diff`), identificando momentos de rápida transición.
*   **Contexto Geográfico:** Latitud y longitud de las estaciones para capturar variaciones espaciales en el desempeño del modelo NWS.
*   **Indicador de Día/Noche:** Variable binaria provista por el API de la NWS.
*   **Persistencia (Lags):** Error de la hora anterior para capturar la inercia térmica y de humedad.

### 2.2 Arquitectura del Modelo
Se utilizó un enfoque de **Regresión Multi-salida** (`MultiOutputRegressor`) con **XGBoost** como regresor base. Este enfoque permite predecir simultáneamente la corrección para la temperatura y la humedad, respetando la correlación física intrínseca entre ambas.

### 2.3 Estrategia de Validación
Dada la naturaleza secuencial de los datos, se empleó un **Blocked TimeSeries Split** (Ventana Móvil), asegurando que el modelo se evalúe en datos cronológicamente posteriores a su entrenamiento.

### 2.4 Optimización
Se utilizó **Optuna** para la búsqueda bayesiana de hiperparámetros, optimizando la profundidad, tasa de aprendizaje y complejidad del modelo.

## 3. Resultados

### 3.1 Mejora en Métricas (Set de Test)
El modelo fue evaluado en el último 15% de los datos cronológicos (~19,000 registros).

| Variable | MAE Original (NWS) | MAE Corregido (ML) | Mejora % |
| :--- | :---: | :---: | :---: |
| Temperatura | 1.1924 | 0.7860 | **34.08%** |
| Humedad | 6.7973 | 4.5501 | **33.06%** |

### 3.2 Análisis de Sesgo Horario
Como se observa en `eval_bias_hourly_temp.png` y `eval_bias_hourly_hum.png`, el modelo de ML reduce drásticamente el error promedio por hora. El sesgo de humedad se ha estabilizado notablemente durante el ciclo nocturno.

### 3.3 Importancia de Variables
El análisis en `eval_feature_importance_combined.png` muestra que:
*   **Temperatura:** Depende fuertemente del pronóstico base y los lags de error.
*   **Humedad:** Muestra una dependencia aún más marcada por los componentes temporales (`hour_sin/cos`), confirmando que el sesgo de humedad es altamente sensible al ciclo de evaporación.

## 4. Fase Experimental: Posibilidades de Mejora

### 4.1 Implementación de Memoria de Corto Plazo (Lags)
Se incorporó el error de la hora anterior (`lag1`). Esto elevó la mejora del 10% inicial al **34%**, indicando que el sesgo tiene una fuerte componente de persistencia.

### 4.2 Estadísticas Móviles (Rolling Window)
Se agregaron promedios y desviaciones estándar móviles de las últimas 3 horas del pronóstico, permitiendo al modelo detectar cambios bruscos de tendencia (frentes fríos) que el pronóstico horario suaviza.

### 4.3 Propuestas Futuras para los Docentes
1.  **Ensamblado Heterogéneo:** Combinar XGBoost con LightGBM mediante *Stacking*.
2.  **Embeddings de Estación:** Entrenar representaciones vectoriales por estación para capturar microclimas locales.
3.  **Redes Recurrentes (LSTM/GRU):** Modelar la secuencia temporal de forma nativa.
4.  **Datos de Satélite:** Integrar nubosidad en tiempo real (GOES-16) para corregir errores de radiación.

## 5. Conclusión
La implementación del modelo multi-salida XGBoost optimizado con Optuna logró una reducción del error superior al 33% en ambas variables. La documentación y visualizaciones generadas confirman que la corrección es robusta y físicamente coherente.
