# -*- coding: utf-8 -*-
"""
Script para visualizar los resultados de la predicción por producto.

Pasos:
1. Carga los datos históricos y las predicciones consolidadas.
2. Identifica los 5 productos con mayor consumo total entre los productos clave.
3. Para cada uno de estos 5 productos, genera y guarda un gráfico que muestra:
   - El consumo histórico mensual desde 2022.
   - La demanda predicha para los próximos 12 meses.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuración ---
sns.set(style="whitegrid")
DATA_PATH = 'data/kardexASTEC_filtrado.xlsx'
PREDICCIONES_PATH = 'output/predicciones_por_producto.csv'
OUTPUT_DIR = 'output'
TOP_N_PRODUCTOS = 5 # Número de productos a visualizar

# --- Funciones Principales ---

def cargar_y_preparar_historico(path):
    """Carga y prepara los datos históricos de consumo."""
    try:
        df = pd.read_excel(path, usecols=['SAP', 'month_year', 'Consumo Total'])
        df.rename(columns={'month_year': 'fecha', 'Consumo Total': 'consumo', 'SAP': 'sap'}, inplace=True)
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df[df['fecha'] >= '2022-01-01'].set_index('fecha')
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de datos en {path}")
        return None

def generar_visualizaciones(df_historico, df_predicciones):
    """
    Identifica los productos top y genera gráficos para cada uno.

    Args:
        df_historico (pd.DataFrame): Datos históricos de consumo.
        df_predicciones (pd.DataFrame): Predicciones generadas.
    """
    if df_historico is None or df_predicciones.empty:
        print("No hay datos suficientes para generar visualizaciones.")
        return

    # 1. Identificar los productos TOP N por consumo histórico.
    consumo_total_por_sap = df_historico.groupby('sap')['consumo'].sum()
    productos_predichos = df_predicciones['sap'].unique()
    # Filtrar para considerar solo los productos para los que tenemos predicción.
    top_productos_sap = consumo_total_por_sap[consumo_total_por_sap.index.isin(productos_predichos)].nlargest(TOP_N_PRODUCTOS).index

    print(f"\n--- Generando gráficos para los {TOP_N_PRODUCTOS} productos más importantes ---")

    # 2. Iterar y generar un gráfico para cada producto top.
    for sap in top_productos_sap:
        # Preparar datos históricos para el producto.
        serie_historica = df_historico[df_historico['sap'] == sap]['consumo'].resample('ME').sum()

        # Obtener la predicción para el producto.
        prediccion_producto = df_predicciones[df_predicciones['sap'] == sap].set_index('fecha_prediccion')['demanda_predicha']

        # Información del modelo utilizado.
        mejor_modelo = df_predicciones[df_predicciones['sap'] == sap]['mejor_modelo'].iloc[0]
        rmse_modelo = df_predicciones[df_predicciones['sap'] == sap]['modelo_rmse'].iloc[0]

        # Crear el gráfico.
        plt.figure(figsize=(16, 8))

        plt.plot(serie_historica.index, serie_historica.values, label='Consumo Histórico', color='teal', marker='o', linestyle='-')
        plt.plot(prediccion_producto.index, prediccion_producto.values, label='Demanda Predicha', color='crimson', marker='x', linestyle='--')

        plt.title(f'Predicción de Demanda para el Producto: {sap}\n(Mejor Modelo: {mejor_modelo} | RMSE: {rmse_modelo:.2f})')
        plt.xlabel('Fecha')
        plt.ylabel('Consumo')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Guardar el gráfico.
        plot_path = os.path.join(OUTPUT_DIR, f'prediccion_producto_{sap}.png')
        plt.savefig(plot_path)
        print(f">>> Gráfico para el producto {sap} guardado en: {plot_path}")
        plt.close()

# --- Bloque Principal de Ejecución ---

def main():
    """Orquesta la generación de visualizaciones."""

    # Cargar datos.
    df_historico = cargar_y_preparar_historico(DATA_PATH)
    try:
        df_predicciones = pd.read_csv(PREDICCIONES_PATH, parse_dates=['fecha_prediccion'])
    except FileNotFoundError:
        print(f"Error: Archivo de predicciones no encontrado en {PREDICCIONES_PATH}")
        return

    # Generar gráficos.
    generar_visualizaciones(df_historico, df_predicciones)

if __name__ == "__main__":
    main()
