# Taller de Análisis de Datos 1 — TP Final
## Corrección de Sesgos en Predicciones Meteorológicas con Machine Learning

Este repositorio contiene la solución completa para el proyecto final del taller. Se ha diseñado, evaluado e implementado un pipeline de datos de extremo a extremo que realiza la ingesta, limpieza, análisis y posterior corrección de sesgos sistemáticos en los pronósticos de temperatura y humedad provistos por la NWS (National Weather Service).

El modelo final basado en **XGBoost** logra corregir el patrón de error asociado al ciclo diurno y reduce el error de predicción en aproximadamente un **45%** para ambas variables físicas.

---

## 👥 Equipo
* **Damian Piuselli**
* **Julian Rolando**
* **Lucia Vazquez**
* **Mauro Frias**

---

## 🏗️ Arquitectura del Pipeline y Componentes

El proyecto se divide en las siguientes etapas consecutivas:

1. **Ingesta y Almacenamiento (AWS Cloud)**
   * **AWS Lambda**: Ubicado en [lambda/](file:///home/damianp/Proyectos/TallerAnalisisDatos/lambda), contiene las funciones (`Recolector datos` y `Clima Lambda`) para extraer pronósticos y observaciones reales de la API de weather.gov para las estaciones meteorológicas.
   * **AWS Glue**: Scripts de ETL y configuraciones ([Aws_glue/](file:///home/damianp/Proyectos/TallerAnalisisDatos/Aws_glue)) para la transformación inicial de los datos estructurados en Amazon S3 de capa *Bronze* a *Silver*.

2. **Procesamiento Local y Calidad de Datos ([local_cleanup/](file:///home/damianp/Proyectos/TallerAnalisisDatos/local_cleanup))**
   * **Unificación**: El script [unify_data.py](file:///home/damianp/Proyectos/TallerAnalisisDatos/local_cleanup/unify_data.py) unifica las predicciones horarias con las observaciones reales correspondientes.
   * **Calidad de Datos**: [data_quality.py](file:///home/damianp/Proyectos/TallerAnalisisDatos/local_cleanup/data_quality.py) limpia, detecta y corrige anomalías, imputa nulos y genera el dataset unificado final `clima_unificado.parquet`.
   * **Diagnósticos**: Scripts como [diagnose_temp_diff.py](file:///home/damianp/Proyectos/TallerAnalisisDatos/local_cleanup/diagnose_temp_diff.py) y [diagnose_wind.py](file:///home/damianp/Proyectos/TallerAnalisisDatos/local_cleanup/diagnose_wind.py) evalúan discrepancias y comportamientos físicos de las variables.

3. **Análisis del Sesgo Diurno ([propuesta_correccion_sesgo_diurno/](file:///home/damianp/Proyectos/TallerAnalisisDatos/propuesta_correccion_sesgo_diurno))**
   * Contiene los análisis visuales e hipótesis de investigación sobre los patrones de error horario de temperatura y humedad en el ciclo diurno (ver [propuesta_correcion_t_hr.md](file:///home/damianp/Proyectos/TallerAnalisisDatos/propuesta_correccion_sesgo_diurno/propuesta_correcion_t_hr.md)).

4. **Entrenamiento y Optimización de Modelos ([desarrollo_modelo/](file:///home/damianp/Proyectos/TallerAnalisisDatos/desarrollo_modelo))**
   * **Ingeniería de Features**: Implementación de lags (persistencia de error de horas previas), tendencias temporales y codificación circular del ciclo diurno (seno/coseno).
   * **Modelado**: Uso de un regresor multi-salida (`MultiOutputRegressor` con **XGBoost**) para predecir de forma conjunta el error de temperatura y humedad.
   * **Optimización**: Búsqueda bayesiana de hiperparámetros y optimización de la ventana de *look-back* (entrenamiento móvil) mediante **Optuna**, logrando un óptimo con los últimos 21 días de datos.

5. **Visualización y Reportes ([dashboard/](file:///home/damianp/Proyectos/TallerAnalisisDatos/dashboard))**
   * Generación automática de un reporte web interactivo ([dashboard/index.html](file:///home/damianp/Proyectos/TallerAnalisisDatos/dashboard/index.html)) utilizando Plotly. Permite inspeccionar métricas globales, sesgos por hora, importancia de variables e impactos geográficos por estación.

---

## 📊 Resultados Principales (Set de Evaluación)

| Métrica / Variable | MAE Pronóstico Original (NWS) | MAE Corregido (ML XGBoost) | Reducción del Error (%) |
| :--- | :---: | :---: | :---: |
| **Temperatura (°C)** | 1.3049 | 0.7198 | **44.84%** |
| **Humedad Relativa (%)** | 7.2023 | 3.8486 | **46.56%** |

---

## 📂 Estructura del Repositorio

```
TallerAnalisisDatos/
├── Aws_glue/                # ETL scripts para AWS Glue
├── dashboard/               # Dashboard interactivo y script generador (index.html)
├── datos_historicos/        # Scripts y datos históricos recopilados
├── desarrollo_modelo/       # Entrenamiento, optimización (Optuna) y evaluación de ML
├── entregables/             # PDF del trabajo final e informes complementarios
├── lambda/                  # Lambdas en AWS para recolección automatizada
├── local_cleanup/           # Unificación local, calidad de datos y análisis de error
├── propuesta_correccion_sesgo_diurno/ # Análisis inicial exploratorio de sesgos diurnos
├── EDA_Final_Climatico.ipynb  # Jupyter Notebook con el análisis exploratorio final
├── requirements.txt         # Dependencias globales del proyecto
├── requirements_local.txt   # Dependencias locales para el modelado y visualización
└── README.md                # Este archivo
```

---

## 🚀 Guía de Ejecución Rápida

### 1. Preparación del Entorno
Instalar todas las dependencias necesarias:
```bash
pip install -r requirements_local.txt
```

### 2. Procesar y Unificar Datos
Para limpiar, imputar nulos y unir pronósticos y observaciones:
```bash
python local_cleanup/unify_data.py
python local_cleanup/data_quality.py
```

### 3. Entrenar y Evaluar el Modelo
Para optimizar, entrenar y evaluar el corrector de sesgo:
```bash
python desarrollo_modelo/train_bias_model.py
python desarrollo_modelo/evaluate_model.py
```

### 4. Generar y Visualizar el Dashboard
Para crear el dashboard web estático actualizado:
```bash
python dashboard/generar_dashboard.py
```
A continuación, abrir en el navegador el archivo generado:
* [dashboard/index.html](file:///home/damianp/Proyectos/TallerAnalisisDatos/dashboard/index.html)

---

## 📝 Documentación Relevante
* **Informe Técnico de ML**: [reporte_modelo_sesgo.md](file:///home/damianp/Proyectos/TallerAnalisisDatos/desarrollo_modelo/reporte_modelo_sesgo.md)
* **Propuesta y Sustento Físico**: [propuesta_correcion_t_hr.md](file:///home/damianp/Proyectos/TallerAnalisisDatos/propuesta_correccion_sesgo_diurno/propuesta_correcion_t_hr.md)
* **Entregables Finales**: El informe final en formato paper PDF se encuentra en [entregables/](file:///home/damianp/Proyectos/TallerAnalisisDatos/entregables).
