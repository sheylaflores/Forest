# Informe de Análisis y Predicción de Demanda para Planificación de Inventario

**Fecha:** 27 de Octubre de 2025
**Autor:** Jules, Ingeniero de Software

---

## 1. Introducción y Objetivos

### 1.1. Contexto del Proyecto
El presente informe detalla el análisis de los datos históricos de consumo (período 2022-2025) correspondientes a la línea de negocio "HIDRÁULICA COMPONENTE" del sector pesca. El objetivo principal es desarrollar un sistema de predicción de demanda que sirva como una herramienta estratégica para optimizar la planificación y gestión de inventario de los componentes principales de la línea.

### 1.2. Objetivos Principales
- **Filtrar y enfocar el análisis:** Excluir productos genéricos para centrarse en los componentes de marca relevantes.
- **Identificar los productos de mayor impacto:** Cuantificar qué artículos son vitales para el negocio dentro del conjunto de datos relevante.
- **Desarrollar pronósticos de demanda precisos e individualizados:** Crear un modelo predictivo para cada producto clave.
- **Generar resultados accionables:** Proporcionar una salida de datos clara y recomendaciones estratégicas.

---

## 2. Metodología Aplicada

El proyecto se ejecutó siguiendo un flujo de trabajo estructurado de ciencia de datos:

### 2.1. Fuentes de Datos
- **Datos de Consumo:** `kardexASTEC_filtrado.xlsx`, con el historial de transacciones.
- **Datos de Productos:** `14.06 MP_ARTICULOS_SAP.xlsx`, utilizado para obtener la marca y descripción de cada producto.

### 2.2. Filtrado de Datos por Marca
Como primer paso crucial, se realizó un filtro para excluir productos que no son componentes principales de la línea de negocio. Se eliminaron del análisis todos los artículos cuya marca fuera **"MATERIALES VARIOS"** o **"GENÉRICO"**. Este paso asegura que los resultados y predicciones se centren únicamente en el inventario estratégico.

### 2.3. Identificación de Productos Clave (Análisis ABC)
Sobre el conjunto de datos ya filtrado, se aplicó el **Principio de Pareto (80/20)**. Se seleccionaron los productos que, en conjunto, representan el **80% del consumo acumulado**.

### 2.4. Modelado Predictivo por Producto
Para cada producto clave, se implementó un proceso de "competencia de modelos":

1.  **Validación:** Se utilizó un enfoque de **Train/Test Split (80/20)** para evaluar el rendimiento de los modelos.
2.  **Modelos Evaluados:** Se compararon ARIMA, SARIMA, RandomForest, XGBoost y Prophet.
3.  **Criterio de Selección:** El modelo con el **Error Cuadrático Medio (RMSE)** más bajo fue seleccionado para cada producto.

### 2.5. Post-procesamiento de Predicciones
- **Ajuste a Cero:** Las predicciones negativas se ajustaron a cero.
- **Redondeo Inteligente:** Se aplicó redondeo al entero más cercano solo si el historial del producto consistía exclusivamente en unidades enteras.

---

## 3. Resultados del Análisis Exploratorio (EDA)

### 3.1. Productos Clave Identificados (Post-Filtro)
Tras excluir las marcas genéricas, el análisis ABC reveló que **170 de 598 productos (el 28.4%) son responsables del 80% del consumo total**. Este es el grupo de productos sobre el cual se centran las predicciones.

### 3.2. Análisis del Consumo Anual (Marcas Relevantes)
Los siguientes gráficos muestran la demanda mensual únicamente de los productos de marca, excluyendo los genéricos.

#### Consumo Mensual - Año 2022
![Consumo 2022](output/consumo_mensual_2022.png)
**Análisis:** El consumo de componentes de marca en 2022 muestra picos claros en abril y octubre, coincidiendo con las temporadas de preparación de la flota pesquera.

#### Consumo Mensual - Año 2023
![Consumo 2023](output/consumo_mensual_2023.png)
**Análisis:** En 2023, la demanda fue fuerte en la primera mitad del año, con un pico excepcional en julio. La marcada caída en la segunda mitad podría deberse a factores externos como el Fenómeno de El Niño.

#### Consumo Mensual - Año 2024
![Consumo 2024](output/consumo_mensual_2024.png)
**Análisis:** El patrón de 2024 es más irregular, lo que subraya la importancia de modelar cada producto individualmente en lugar de depender de un patrón agregado.

---

## 4. Resultados de la Predicción de Demanda

A continuación, se detalla el análisis para los 5 productos más importantes del conjunto de datos filtrado.

### 4.1. Análisis Detallado de Productos Principales

#### 1. Producto: MANGUERA SAE 100 R2 AT (SAP: A22020000062)
![Predicción A22020000062](output/prediccion_producto_A22020000062.png)
- **Modelo Seleccionado:** **ARIMA**. Indica que la demanda de esta manguera, aunque esporádica, se predice mejor a partir de sus propios valores pasados.
- **Análisis del RMSE:** El **RMSE es de 10.33**. Para un producto con demanda intermitente, este error es aceptable y la predicción sirve como una buena línea base para la planificación.

#### 2. Producto: ADAPTADOR C/HILOS NPTF (SAP: A18130006711)
![Predicción A18130006711](output/prediccion_producto_A18130006711.png)
- **Modelo Seleccionado:** **ARIMA**. La elección de este modelo sugiere que el consumo de este adaptador sigue un patrón de "memoria a corto plazo", donde los consumos recientes son el mejor indicador de los futuros.
- **Análisis del RMSE:** El modelo alcanzó un **RMSE de 2.06**, un valor muy bajo. Esto indica que la predicción es extremadamente precisa, permitiendo una gestión de inventario muy ajustada y con bajo riesgo de quiebre.

#### 3. Producto: FILTRO DE SUCCION (SAP: A18110010326)
![Predicción A18110010326](output/prediccion_producto_A18110010326.png)
- **Modelo Seleccionado:** **Prophet**. La selección de Prophet sugiere que la demanda de este filtro tiene componentes de tiempo (tendencia o estacionalidad) que los otros modelos no capturan tan bien.
- **Análisis del RMSE:** El **RMSE es de 2.58**. Este es un excelente resultado, indicando que el modelo puede predecir la demanda mensual con una desviación promedio de solo ~3 unidades. La predicción es muy fiable.

#### 4. Producto: FILTRO DE AIRE (SAP: A18110009037)
![Predicción A18110009037](output/prediccion_producto_A18110009037.png)
- **Modelo Seleccionado:** **ARIMA**. La demanda de este filtro de aire es mejor predicha por sus valores históricos directos, lo que es común para consumibles de mantenimiento regular.
- **Análisis del RMSE:** El **RMSE es de 1.83**. Un error tan bajo demuestra una predicción de muy alta precisión, lo que permite una planificación de inventario "just-in-time" con un stock de seguridad mínimo.

#### 5. Producto: CODO MA H-H 90 (SAP: A18130006673)
![Predicción A18130006673](output/prediccion_producto_A18130006673.png)
- **Modelo Seleccionado:** **Prophet**. Similar a otros componentes, la demanda de este codo parece seguir patrones de tiempo que Prophet puede identificar, lo que lo hace el modelo más adecuado.
- **Análisis del RMSE:** Con un **RMSE de 2.21**, la predicción es muy precisa. El modelo predice una demanda estable, facilitando su gestión en el inventario.

---

## 5. Conclusiones y Recomendaciones Estratégicas

1.  **Enfoque en Inventario de Marca:** El filtrado inicial ha sido clave. La empresa debe centrar sus esfuerzos de planificación en los **170 productos de marca identificados** para optimizar el capital de trabajo.
2.  **Confianza en la Predicción Individual:** El sistema ha demostrado que diferentes productos requieren diferentes modelos. El archivo `predicciones_por_producto.csv` es la herramienta central para la planificación de compras.
3.  **Uso del RMSE para el Stock de Seguridad:** Productos con RMSE bajo permiten inventarios ajustados. Productos con RMSE alto (en relación a su demanda) señalan la necesidad de un **mayor stock de seguridad**.

---

## 6. Próximos Pasos y Mejoras Futuras

Para incrementar aún más la precisión, se recomienda integrar **variables externas** como:
- **Calendario de Temporadas de Pesca y Vedas.**
- **Índice del Fenómeno de El Niño (ENSO).**
- **Datos de Cuotas de Pesca.**

---

## Apéndice: Ejecución Técnica del Proyecto

1.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Ejecutar los scripts en orden:**
    ```bash
    python3 src/analisis_exploratorio.py
    python3 src/prediccion_por_producto.py
    python3 src/visualizar_predicciones.py
    ```
    *Nota: Todos los resultados se generan en la carpeta `output/`.*
