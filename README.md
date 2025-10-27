# Análisis y Predicción de Demanda para Planificación de Inventario

## 1. Resumen del Proyecto

Este proyecto analiza los datos históricos de consumo (2022-2025) de la línea de negocio "HIDRÁULICA COMPONENTE" para optimizar la planificación de inventario.

El enfoque se centra en **identificar los productos más críticos** (aquellos que generan el 80% del consumo) y desarrollar **predicciones de demanda individuales y contextualizadas** para cada uno de ellos.

Las características clave del proyecto son:
- **Análisis Exploratorio (EDA)** enfocado en el período 2022-2025.
- **Identificación de Productos Clave** mediante un análisis ABC (Pareto).
- **Modelado Individual por Producto:** Se comparan 5 modelos para cada producto clave.
- **Predicción Inteligente a 12 Meses:**
    - Se añade la **descripción del producto** para mayor claridad.
    - Se aplica un **redondeo inteligente**: si un producto siempre se ha consumido en unidades enteras, la predicción se redondea al entero más cercano. De lo contrario, se mantiene en decimales.
    - Las predicciones **nunca son negativas** (se ajustan a cero).

---

## 2. Estructura del Repositorio

```
.
├── data/
│   └── kardexASTEC_filtrado.xlsx
├── output/
│   ├── consumo_mensual_2022.png ...
│   ├── prediccion_producto_A18110011069.png ...
│   ├── predicciones_por_producto.csv
│   └── productos_clave.txt
├── src/
│   ├── analisis_exploratorio.py
│   ├── prediccion_por_producto.py
│   └── visualizar_predicciones.py
├── requirements.txt
└── README.md
```

---

## 3. Cómo Ejecutar el Proyecto

1.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Ejecutar los scripts en orden:**
    ```bash
    python3 src/analisis_exploratorio.py
    python3 src/prediccion_por_producto.py
    python3 src/visualizar_predicciones.py
    ```

---

## 4. Análisis Exploratorio (2022-2025)

### a. Consumo Mensual por Año
El análisis anual revela patrones y picos de demanda que varían, lo que justifica la necesidad de un modelado individual por producto.

![Consumo 2022](output/consumo_mensual_2022.png)
![Consumo 2023](output/consumo_mensual_2023.png)

### b. Identificación de Productos Clave
El análisis ABC (Pareto) demostró que **90 de 666 productos (el 13.5%) son responsables del 80% del consumo total**. Esto permite a la empresa centrar sus esfuerzos de planificación en un grupo manejable de artículos de alto impacto.

---

## 5. Predicción de Demanda por Producto

Para cada uno de los 90 productos clave, se realizó un "campeonato de modelos". El modelo con el menor Error Cuadrático Medio (RMSE) fue seleccionado y utilizado para generar un pronóstico a 12 meses. El resultado final en `output/predicciones_por_producto.csv` contiene la predicción con su formato de redondeo correcto y la descripción del producto.

### Resultados para los 5 Productos Principales
A continuación se muestran los resultados visuales para los 5 productos más consumidos:

**1. Producto: A18110011069**
![Predicción A18110011069](output/prediccion_producto_A18110011069.png)

**2. Producto: A19010000519**
![Predicción A19010000519](output/prediccion_producto_A19010000519.png)

**3. Producto: A18110011082**
![Predicción A18110011082](output/prediccion_producto_A18110011082.png)

**4. Producto: A22020000062**
![Predicción A22020000062](output/prediccion_producto_A22020000062.png)

**5. Producto: A18130005783**
![Predicción A18130005783](output/prediccion_producto_A18130005783.png)

---

## 6. Conclusiones y Recomendaciones

1.  **Inventario Diferenciado y Preciso:** Utilizar las predicciones individuales del archivo CSV, que respetan el formato de unidad (entero/decimal), para establecer **niveles de stock de seguridad dinámicos** para cada uno de los 90 productos clave.
2.  **Revisión y Re-entrenamiento Periódico:** Se recomienda **ejecutar este análisis trimestralmente** para actualizar la lista de productos clave y re-entrenar los modelos con los datos más recientes.
3.  **Análisis de Causa Raíz:** Para los productos top con alta volatilidad, investigar las **causas de los picos de demanda** (proyectos, temporadas, etc.) para enriquecer el contexto del pronóstico.
4.  **Gestión Proactiva:** Usar el archivo de predicciones como base para **anticipar futuras necesidades de compra** y negociar mejores condiciones con proveedores.
