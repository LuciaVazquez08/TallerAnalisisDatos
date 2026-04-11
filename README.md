# Taller de Análisis de Datos 1
## TP1 — Evaluación y Diagnóstico de Predicciones Meteorológicas

# 👥 Equipo
Damian Piuselli
Julian Rolando 
Lucia Vazquez
Mauro Frias

## 📌 Objetivo
Construir un pipeline reproducible para comparar predicciones meteorológicas con observaciones reales, evaluar el error y diagnosticar el desempeño del modelo.

El pipeline debe:
* Consumir datos desde APIs meteorológicas
* Unificar datos de distintas fuentes
* Limpiar y transformar los datos
* Generar un dataset final
* Permitir análisis del error de predicción

---
# 🏗️ Arquitectura del Pipeline
```
EventBridge (schedule)
        ↓
    Weather API 
        ↓
AWS Lambda (ingestión)
        ↓
    S3 raw storage
        ↓
    Transformación
        ↓
Dataset Final (CSV/Parquet)
```

---
# ☁️ Infraestructura AWS utilizada

* Amazon EventBridge → Corida programada
* AWS Lambda → extracción de datos
* Amazon S3 → almacenamiento de datos crudos y procesados
* IAM Roles → permisos de acceso
* CloudWatch → ejecución programada
---

# 📂 Estructura del proyecto

```
weather-pipeline/
│
├── lambda/
│   └── ingest_weather.py
│
├── data/
│   └── estaciones_560.csv
│
├── transform/
│   └── (pendiente)
│
├── dataset/
│   └── (pendiente)
│
├── requirements.txt
└── README.md
```

---

# 🔄 Pipeline actual

Actualmente implementado:

1. AWS Lambda consume API weather.gov
2. Obtiene:
   * estaciones
   * pronósticos
   * pronóstico hourly
   * observaciones actuales
   * historial
3. Guarda cada respuesta en S3 como JSON
4. Archivos versionados con timestamp
---

# 📁 Estructura esperada en S3

```
datos-clima-prueba/
 ├── raw/
 │    ├── stations/
 │    ├── forecast_periods/
 │    ├── forecast_hourly/
 │    ├── station_current/
 │    └── station_history/
 │
 └── processed/
      └── dataset_final.parquet
```

---
# 📍 Datos de entrada
* API meteorológica: weather.gov
* CSV con 560 estaciones meteorológicas
* Coordenadas y station_id
---

# ⚙️ Transformación (pendiente)

Se implementará un proceso que:

* Limpie datos nulos
* Unifique timestamps
* Seleccione variables relevantes
* Compare forecast vs observación real
* Calcule error de predicción
* Genere dataset final

Columnas esperadas del dataset final:

```
timestamp
station_id
temperature_real
temperature_forecast
error_temperature
humidity_real
humidity_forecast
error_humidity
```

---

# 📊 Dataset final
Formato requerido:

* CSV o Parquet
* limpio
* fusionado
* sin valores inconsistentes

Output esperado:

```
s3://datos-clima-prueba/processed/dataset_final.parquet
```

---

# 🔁 Reproducibilidad

El pipeline es reproducible porque:

* Código versionado en Git
* Lambda configurable
* Dataset generado automáticamente
* Estructura S3 definida
* Dependencias documentadas

---

# ✅ Checklist de entregables TP

## 1. Código del pipeline

* [x] Lambda de extracción
* [ ] Transformación
* [ ] Generación dataset final
* [x] Código documentado
* [x] Versionado en Git

## 2. Dataset final

* [ ] Limpio
* [ ] Fusionado
* [ ] CSV o Parquet
* [ ] Guardado en S3

## 3. Informe técnico

* [ ] Descripción pipeline
* [ ] Calidad de datos
* [ ] Análisis de error
* [ ] Diagnóstico del modelo

---
# 📌 Estado actual del proyecto

## Implementado
* Consumo de API weather.gov
* Lambda en AWS
* Guardado en S3
* Versionado temporal con timestamp
* CSV con 560 estaciones

## Pendiente
* Lectura automática del CSV de estaciones
* Pipeline de transformación
* Limpieza de datos
* Unificación forecast vs observaciones
* Cálculo de error
* Generación dataset final
* Automatización completa

---

# 🚀 Pasos siguientes
1. Leer CSV de 560 estaciones
2. Ejecutar Lambda por cada estación
3. Guardar datos raw en S3
4. Crear script de transformación
5. Fusionar datasets
6. Calcular errores
7. Generar dataset final
8. Guardar dataset en S3 processed/

---

# 🔗 Cómo conectar con AWS
## Requisitos

* AWS CLI configurado
* bucket S3 creado
* IAM Role con permisos S3
* Lambda desplegado

## Variables necesarias
```
BUCKET=datos-clima-prueba
REGION=us-east-1
```

## Permisos IAM requeridos
* s3:PutObject
* s3:GetObject
* s3:ListBucket

---
# ▶️ Ejecución del pipeline

El pipeline se ejecuta desde AWS Lambda.
Opciones:

* manual desde consola AWS
* programado con CloudWatch
* invocado desde script

---

# 📦 Dependencias
```
boto3
requests
pandas (para transformación futura)
pyarrow (para parquet)
```

---

# 📝 Notas

La transformación y generación del dataset final serán implementadas en la siguiente etapa del proyecto.
