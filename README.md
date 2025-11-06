# Proyecto de Predicción de Demanda para Componentes Hidráulicos (Versión Revisada)

## 1. Resumen del Proyecto

Este proyecto presenta una solución robusta para la predicción de demanda de 20 productos críticos de la línea de negocio "Hidráulica Componente", enfocada en el sector pesquero. El objetivo fue generar un pronóstico de consumo para los próximos 6 meses (Noviembre 2025 - Abril 2026).

Esta versión del análisis ha sido **revisada y actualizada** para utilizar una partición de datos más rigurosa (**80/20 estricta**) y para incluir gráficos de validación detallados que permiten una comparación visual directa del rendimiento de los modelos.

## 2. Metodología

### a. Configuración y Preprocesamiento de Datos
- Se cargó el archivo `data.xlsx` y se filtró para el período de análisis (Enero 2022 - Octubre 2025).
- Se seleccionaron los 20 códigos SAP críticos y se aseguró la continuidad de las series temporales.

### b. Ingeniería de Características
Se crearon variables exógenas clave para capturar la estacionalidad del negocio:
- **`temporada_pesca`**: Indicador para los meses de alta actividad pesquera (Abril, Mayo, Junio, Nov, Dic).
- **`pico_pre_temporada`**: Indicador para los meses de preparación y alta demanda (Feb, Mar, Sep, Oct).

### c. Modelado y Evaluación (Metodología 80/20)
- Los datos se dividieron en una partición cronológica **80/20 estricta**:
    - **Conjunto de Entrenamiento (37 meses):** Enero 2022 - Enero 2025.
    - **Conjunto de Validación (9 meses):** Febrero 2025 - Octubre 2025.
- Se entrenaron tres modelos (SARIMAX, XGBoost, Prophet) para cada producto.
- Se evaluó el rendimiento en el período de validación de 9 meses usando **RMSE**, **MAE** y **MAPE**. El **MAPE** fue el criterio decisivo para seleccionar el mejor modelo.

### d. Generación de Pronóstico Final
- El mejor modelo para cada producto fue reentrenado utilizando todos los datos históricos (2022-2025).
- Se generó el pronóstico de demanda para los 6 meses futuros (Nov 2025 - Abr 2026).

## 3. Estructura del Repositorio

- **/data**: Contiene el archivo de datos original.
- **/src**: Contiene los scripts de Python, numerados secuencialmente.
- **/output**: Contiene todos los entregables generados:
    - **`error_metrics_enhanced.xlsx`**: **(ENTREGABLE CLAVE)** Tabla pivote mejorada con las métricas de error y una justificación explícita del modelo seleccionado.
    - **`final_predictions.xlsx`**: **(ENTREGABLE CLAVE)** Pronóstico final de 6 meses para los 20 productos.
    - **/validation_plots**: **(NUEVO ENTREGABLE CLAVE)** Carpeta con 5 gráficos que comparan el rendimiento de los 3 modelos contra el consumo real en el período de validación de 9 meses.
    - **/individual_forecasts**: Carpeta con los 20 gráficos de pronóstico individuales.
    - `...` (otros archivos intermedios).

## 4. Cómo Reproducir el Análisis

Para ejecutar el proyecto, sigue los siguientes pasos:
1. Asegúrate de tener Python y `pip` instalados.
2. Instala las dependencias: `pip install pandas openpyxl statsmodels xgboost prophet scikit-learn matplotlib seaborn`
3. Ejecuta los scripts en la carpeta `src` en orden numérico.

---
*Este análisis fue generado por Jules, un asistente de ingeniería de IA.*
