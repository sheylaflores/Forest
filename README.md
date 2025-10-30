# Informe Final: Análisis y Predicción de Demanda con Variables Exógenas

**Fecha:** 28 de Octubre de 2025


---

## 1. Introducción 

### 1.1. Sector pesquero
El portal **Observatorio PRODUCEmpresarial** ofrece información mensual (setiembre 2025), anuarios estadísticos (2023), fichas técnicas e informes coyunturales para la industria pesquera.

Recurso: https://www.producempresarial.pe/pesca-tablero/

Ficha Tecnica de la ANCHOVETA: https://www.producempresarial.pe/desenvolvimiento-socioeconomico-del-recurso-anchoveta-chi/
#### Índice de estacionalidad y pronóstico del desempeño pesquero
![Consumo 2022](output/Estacionalidad Pesca Anchoveta.png)
**Análisis:** El consumo en 2022 muestra picos claros en los meses definidos como "temporada de pesca" (Abril, Noviembre-Diciembre), validando la relevancia de esta variable.


Este informe presenta una metodología avanzada para la predicción de demanda de la línea de negocio "HIDRÁULICA COMPONENTE" en el sector pesca. A diferencia de análisis previos, este proyecto incorpora **reglas de negocio específicas** y **variables externas** para mejorar significativamente la precisión y relevancia de los pronósticos, con el objetivo final de optimizar la gestión de inventario.

### 1.2. Objetivos Específicos
- **Depurar el Conjunto de Datos:** Excluir productos genéricos y de baja relevancia para enfocar el análisis en el inventario estratégico.
- **Incorporar Inteligencia de Negocio:** Integrar una variable exógena que modele el impacto de las temporadas de pesca.
- **Desarrollar Pronósticos Individuales y Precisos:** Generar una predicción a 12 meses para cada producto clave, seleccionando el modelo más adecuado.
- **Producir Resultados Accionables:** Entregar un informe claro con conclusiones y recomendaciones basadas en datos.

---

## 2. Metodología Aplicada

El proyecto se ejecutó en tres fases principales, orquestadas por scripts de Python.

### 2.1. Fase 1: Limpieza de Datos y Análisis Exploratorio (EDA)

1.  **Filtrado por Marca:** Se utilizó el archivo `14.06 MP_ARTICULOS_SAP.xlsx` para identificar todos los productos cuya marca es **"MATERIALES VARIOS"** o **"GENÉRICO"**. Se generó una lista con **878 productos excluidos** (`output/productos_excluidos.txt`).
2.  **Depuración de Datos de Consumo:** Se eliminaron todas las transacciones asociadas a los productos excluidos del archivo `kardexASTEC_filtrado.xlsx`.
3.  **Análisis Descriptivo (2022-2025):** Sobre los datos limpios, se calculó un resumen estadístico del consumo mensual total.
4.  **Identificación de Productos Clave:** Se aplicó un **análisis de Pareto (80/20)**, resultando en una lista de **170 productos clave** que representan el 80% del consumo y que fue guardada en `output/productos_clave.txt`.

### 2.2. Fase 2: Modelado y Predicción con Variable Exógena

1.  **Creación de Variable Exógena:** Se definió una variable binaria `temporada_pesca`:
    - `1` = Temporada alta (Abril, Mayo, Junio, Noviembre, Diciembre).
    - `0` = Temporada baja/veda (resto de meses).
2.  **Competencia de Modelos por Producto:** Para cada uno de los 170 productos clave, se realizó una competencia entre cinco modelos, integrando la variable exógena:
    - **ARIMA/SARIMA:** Usando el parámetro `exog`.
    - **XGBoost/RandomForest:** Incluyendo `temporada_pesca` como una característica.
    - **Prophet:** Utilizando `add_regressor('temporada_pesca')`.
3.  **Selección y Predicción:** Se seleccionó el modelo con el menor **RMSE** en un set de prueba (20% de los datos). Luego, se re-entrenó con todos los datos y se generó un pronóstico a 12 meses, alimentándolo con los valores futuros de la `temporada_pesca`.
4.  **Post-procesamiento:** Se aplicó el **ajuste a cero** para predicciones negativas y el **redondeo inteligente** (entero vs. decimal) basado en el historial de cada producto.

### 2.3. Fase 3: Visualización de Resultados
Se generaron gráficos individuales para los 5 productos más importantes, comparando su historial con la predicción y mostrando **bandas de confianza** para los modelos ARIMA/SARIMA.

---

## 3. Resultados del Análisis Exploratorio (EDA)

### 3.1. Resumen Estadístico del Consumo
- CANTIDAD INCIAL ANTES DE DEPURACIÓN: 1050
- CANTIDAD FINAL DESPUES DE DEPURACIÓN: 931
- CANTIDAD FINAL DESPUES DE FILTRADO 2022 -20225: 598
Tras filtrar las marcas genéricas, el consumo mensual total para el período 2022-2025 presenta las siguientes características:
- **Media (mean):** 114.76 unidades/mes.
- **Desviación Estándar (std):** 55.83, indicando una volatilidad considerable.
- **Mínimo y Máximo:** El consumo mensual ha variado entre 33 y 258 unidades.
- **Cuartiles:** El 50% de los meses, el consumo se encuentra entre 72 y 144 unidades.

### 3.2. Análisis del Consumo Anual (Marcas Relevantes)

#### Consumo Mensual - Año 2022
![Consumo 2022](output/consumo_mensual_2022.png)
**Análisis:** El consumo en 2022 muestra picos claros en los meses definidos como "temporada de pesca" (Abril, Noviembre-Diciembre), validando la relevancia de esta variable.

#### Consumo Mensual - Año 2023
![Consumo 2023](output/consumo_mensual_2023.png)
**Análisis:** 2023 muestra un comportamiento atípico con un pico masivo en Julio (definido como temporada baja). Esto podría deberse a factores externos no modelados (ej. cambios regulatorios, efectos de El Niño), lo que justifica la necesidad de modelos robustos.

#### Consumo Mensual - Año 2024
![Consumo 2024](output/consumo_mensual_2024.png)
**Análisis:** El patrón de 2024 vuelve a alinearse parcialmente con las temporadas de pesca, reforzando que, aunque no es una regla perfecta, la estacionalidad es un factor importante.

---

## 4. Resultados de la Predicción de Demanda

A continuación, se presenta el análisis de las predicciones para los 5 productos más relevantes tras el filtrado.

### 4.1. Análisis Detallado de Productos Principales

#### 1. Producto: MANGUERA SAE 100 R2 AT (SAP: A22020000062)
![Predicción A22020000062](output/prediccion_producto_A22020000062.png)
- **Modelo Seleccionado:** **ARIMA**. A pesar de la inclusión de la variable de temporada, el modelo sigue considerando que el comportamiento histórico del producto es el mejor predictor.
- **Análisis del RMSE:** Con un **RMSE de 10.33**, la predicción es sólida para un artículo de demanda esporádica. Las bandas de confianza sugieren que, aunque la predicción base es baja, la empresa debe estar preparada para picos de hasta ~20 unidades en ciertos meses.

#### 2. Producto: ADAPTADOR C/HILOS NPTF (SAP: A18130006711)
![Predicción A18130006711](output/prediccion_producto_A18130006711.png)
- **Modelo Seleccionado:** **ARIMA**.
- **Análisis del RMSE:** El **RMSE de 2.06** es excepcionalmente bajo, lo que indica una predicción de muy alta fiabilidad. Las bandas de confianza son estrechas, permitiendo una planificación de inventario muy ajustada.

#### 3. Producto: FILTRO DE SUCCION (SAP: A18110010326)
![Predicción A18110010326](output/prediccion_producto_A18110010326.png)
- **Modelo Seleccionado:** **Prophet**. La elección de Prophet sugiere que la variable de temporada de pesca fue muy útil para este producto, ya que Prophet es excelente integrando regresores externos.
- **Análisis del RMSE:** El **RMSE de 2.58** es excelente. El modelo predice un aumento de la demanda en el próximo período de temporada alta, una visión que un modelo univariado podría no haber capturado con tanta claridad.

#### 4. Producto: FILTRO DE AIRE (SAP: A18110009037)
![Predicción A18110009037](output/prediccion_producto_A18110009037.png)
- **Modelo Seleccionado:** **ARIMA**.
- **Análisis del RMSE:** Con un **RMSE de 1.83**, la predicción es casi perfecta. Esto sugiere que el producto tiene un patrón de consumo muy estable y predecible.

#### 5. Producto: CODO MA H-H 90 (SAP: A18130006673)
![Predicción A18130006673](output/prediccion_producto_A18130006673.png)
- **Modelo Seleccionado:** **Prophet**.
- **Análisis del RMSE:** El **RMSE de 2.21** indica una predicción muy fiable. El modelo utiliza la variable de temporada para anticipar la demanda futura, mostrando picos en los períodos de alta actividad pesquera.

---

## 5. Conclusiones y Recomendaciones Estratégicas

1.  **La Limpieza de Datos es Clave:** Excluir las marcas genéricas fue fundamental. El análisis ahora se centra en **170 productos estratégicos**, lo que permite una gestión de inventario más eficiente.
2.  **La Variable de Temporada Mejora el Contexto:** La inclusión de la `temporada_pesca` demostró ser valiosa, especialmente para los modelos como **Prophet**, permitiéndoles anticipar cambios en la demanda que no son visibles solo en el historial.
3.  **Utilizar el `RMSE` para el Stock de Seguridad:** El RMSE de cada producto (disponible en el CSV final) es una guía directa para definir el stock de seguridad. Un RMSE de 10 significa que la predicción puede desviarse en 10 unidades, y el inventario debe poder absorber esa variabilidad.
4.  **Recomendación Principal:** La empresa debe adoptar el archivo `predicciones_por_producto.csv` como la herramienta central para su planificación de compras. Además, se recomienda **monitorear el RMSE de los modelos trimestralmente** y re-entrenarlos para mantener su precisión a lo largo del tiempo.

---

## Apéndice: Ejecución Técnica

1.  **Instalar dependencias:** `pip install -r requirements.txt`
2.  **Ejecutar los scripts en orden:**
    ```bash
    python3 src/1_analisis_exploratorio.py
    python3 src/2_entrenamiento_y_prediccion.py
    python3 src/3_visualizacion_resultados.py
    ```
    *Todos los resultados se generan en la carpeta `output/`.*
