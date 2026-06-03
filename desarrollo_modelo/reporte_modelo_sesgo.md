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
El modelo fue evaluado en el último 15% de los datos cronológicos (~66,000 registros). El dataset expandido comprende una ventana de 48 días (Abril-Junio 2026), entrenando con una ventana óptima de **21 días**.

| Variable | MAE Original (NWS) | MAE Corregido (ML) | Mejora % |
| :--- | :---: | :---: | :---: |
| Temperatura | 1.3049 | 0.7198 | **44.84%** |
| Humedad | 7.2023 | 3.8486 | **46.56%** |

### 3.2 Análisis de Sesgo Horario
Como se observa en `eval_bias_hourly_temp.png` y `eval_bias_hourly_hum.png`, el modelo de ML reduce drásticamente el error promedio por hora. La expansión del dataset ha permitido una corrección mucho más precisa de los picos de humedad nocturnos.

### 3.3 Importancia de Variables
El análisis muestra los predictores más relevantes:
*   **Temperatura:** El diferencial de pronóstico y los lags siguen siendo críticos, pero la latitud/longitud han ganado peso al usar un dataset más largo que cubre más variabilidad climática.
*   **Humedad:** La dependencia del ciclo diurno (`hour_sin/cos`) se mantiene como el factor dominante.

## 4. Fase Experimental: Resultados de Optimización

### 4.1 Optimización de la Ventana de Look-back
Se realizó un experimento variando la ventana de entrenamiento de 1 a 35 días. Se determinó que **21 días** es el punto óptimo, logrando un equilibrio entre capturar la dinámica climática reciente y evitar el *concept drift* de periodos muy antiguos. Esta optimización permitió subir la mejora del ~34% inicial a un **~45%** actual.

### 4.2 Estadísticas Móviles (Rolling Window)
Se agregaron promedios y desviaciones estándar móviles de las últimas 3 horas del pronóstico, permitiendo al modelo detectar cambios bruscos de tendencia (frentes fríos) que el pronóstico horario suaviza.

### 4.3 Propuestas Futuras para los Docentes
1.  **Optimización de la Ventana de Look-back:** Implementar una búsqueda de hiperparámetros (vía Optuna) para determinar el periodo de entrenamiento óptimo (ej. últimos 7 o 14 días). Esto permitiría al modelo adaptarse al *Concept Drift* estacional, ignorando datos históricos que ya no representen la dinámica climática actual.
2.  **Ensamblado Heterogéneo:** Combinar XGBoost con LightGBM mediante *Stacking* para reducir la varianza y mejorar la generalización.
3.  **Embeddings de Estación:** Entrenar representaciones vectoriales por estación que capturen microclimas locales y efectos de isla de calor urbana de forma latente.
4.  **Redes Recurrentes (LSTM/GRU):** Evaluar arquitecturas que modelen la secuencia temporal de forma nativa para comparación con el enfoque de lags manuales.

## 5. Conclusión
La implementación del modelo multi-salida XGBoost optimizado con Optuna logró una reducción del error superior al 33% en ambas variables. La documentación y visualizaciones generadas confirman que la corrección es robusta y físicamente coherente para el periodo analizado.
