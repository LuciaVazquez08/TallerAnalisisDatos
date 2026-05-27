Corrección de Sesgo en Temperatura y Humedad

  1. Definición del Problema
   - Variables a predecir: temp_real y hum_real.
   - Pregunta tentativa: "¿En qué medida la corrección conjunta de los errores de temperatura y humedad permite capturar mejor la dinámica de la capa límite atmosférica que el modelo de la NWS tiende a simplificar?"
   - Justificación de la elección: La temperatura y la humedad están ligadas por leyes físicas. Un error en la predicción del
     calentamiento solar impacta simultáneamente a ambas, pero en direcciones opuestas. Corregirlas en conjunto permite al modelo de ML mantener la
     consistencia física y tener en cuenta la estructura de correlacion inherente entre ambas.

  2. Estrategia de Features
   - Features clave: 
       - Pronóstico base: temp_fcst y hum_fcst.
       - Temporalidad: hour (transformada a seno/coseno para capturar la naturaleza cíclica) y is_daytime_fcst.
       - Geografía: lat_estacion y lon_estacion. No estan presentes en el dataset pero pueden calcularse usando QGIS y distintas bases de datos geograficas.
   - Feature de ingeniería: El "Diferencial Térmico" (cambio de temperatura respecto a la hora anterior) para identificar momentos de transición
     (amanecer/atardecer) donde el sesgo es máximo.

  3. Ventana Temporal y Alcance
   - Ventana: observaciones y pronosticos de los ultimos dos meses.
   - Jerarquización: 
       - Nivel Pronóstico: Corto plazo (t+1h).
       - Nivel Cortes: Comparación entre climas secos vs. húmedos (agrupando por longitud, separando el este y el oeste de la zona de estudio).
