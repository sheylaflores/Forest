# -*- coding: utf-8 -*-
"""
Script para el Análisis Exploratorio de Datos (EDA) de la demanda de productos.

Este script carga los datos históricos, los filtra para el período 2022-2025,
genera visualizaciones anuales y mensuales, e identifica los productos clave
para la planificación de inventario utilizando un análisis ABC (Pareto).
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Configuración de Estilo para Gráficos ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# --- Carga de Datos ---
DATA_PATH = 'data/kardexASTEC_filtrado.xlsx'
OUTPUT_DIR = 'output'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def cargar_y_preprocesar_datos(path):
    """
    Carga, limpia y preprocesa los datos para el análisis.

    Args:
        path (str): Ruta al archivo .xlsx.

    Returns:
        pd.DataFrame: DataFrame preprocesado y listo para el análisis.
    """
    try:
        df = pd.read_excel(path)
        print(">>> Datos cargados exitosamente.")
    except FileNotFoundError:
        print(f"Error: El archivo no se encontró en la ruta: {path}")
        return None

    df.rename(columns={
        'month_year': 'fecha',
        'Consumo Total': 'consumo',
        'SAP': 'sap'
    }, inplace=True)

    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df.dropna(subset=['fecha', 'consumo', 'sap'], inplace=True)

    # --- Filtrado de Fechas: 2022 en adelante ---
    df = df[df['fecha'] >= '2022-01-01']
    print(f">>> Datos filtrados para el período de 2022 a {df['fecha'].max().year}.")

    df.set_index('fecha', inplace=True)
    df.sort_index(inplace=True)

    return df

def visualizar_consumo_por_año(df):
    """
    Genera y guarda un gráfico de consumo mensual para cada año en el DataFrame.

    Args:
        df (pd.DataFrame): DataFrame preprocesado.
    """
    if df is None:
        return

    años = df.index.year.unique()

    print("\n--- Generando gráficos de consumo mensual por año ---")

    for año in años:
        # Filtra los datos para el año actual.
        df_año = df[df.index.year == año]
        # Agrupa por mes.
        consumo_mensual_año = df_año['consumo'].resample('ME').sum()

        # Completa los meses faltantes con ceros para tener un gráfico de 12 meses.
        idx = pd.date_range(f'01-01-{año}', f'12-31-{año}', freq='ME')
        consumo_mensual_año = consumo_mensual_año.reindex(idx, fill_value=0)

        plt.figure(figsize=(12, 6))
        consumo_mensual_año.plot(kind='bar', color=sns.color_palette('viridis', 12))
        plt.title(f'Consumo Mensual del Año {año}')
        plt.xlabel('Mes')
        plt.ylabel('Consumo Total')
        plt.xticks(ticks=range(len(consumo_mensual_año)), labels=[d.strftime('%b') for d in consumo_mensual_año.index], rotation=45)
        plt.tight_layout()

        plot_path = os.path.join(OUTPUT_DIR, f'consumo_mensual_{año}.png')
        plt.savefig(plot_path)
        print(f">>> Gráfico para el año {año} guardado en: {plot_path}")
        plt.close()

def identificar_productos_clave(df):
    """
    Identifica los productos clave utilizando el análisis ABC (Pareto 80/20).
    Selecciona los productos que contribuyen al 80% del consumo total.

    Args:
        df (pd.DataFrame): DataFrame con los datos de consumo.

    Returns:
        list: Una lista con los códigos SAP de los productos clave.
    """
    if df is None:
        return []

    print("\n--- Identificando productos clave (Análisis ABC) ---")

    # Calcula el consumo total por producto.
    consumo_por_producto = df.groupby('sap')['consumo'].sum().sort_values(ascending=False)

    # Calcula el porcentaje acumulado.
    consumo_acumulado = consumo_por_producto.cumsum()
    consumo_total = consumo_por_producto.sum()
    porcentaje_acumulado = (consumo_acumulado / consumo_total) * 100

    # Identifica los productos que caen dentro del 80% del consumo.
    productos_clave = porcentaje_acumulado[porcentaje_acumulado <= 80].index.tolist()

    num_total_productos = len(consumo_por_producto)
    num_productos_clave = len(productos_clave)
    porcentaje_productos = (num_productos_clave / num_total_productos) * 100

    print(f">>> {num_productos_clave} de {num_total_productos} productos ({porcentaje_productos:.2f}%) representan el 80% del consumo total.")
    print("Productos clave identificados:")
    for sap in productos_clave:
        print(f"  - {sap}")

    # Guardar la lista de productos clave en un archivo.
    with open(os.path.join(OUTPUT_DIR, 'productos_clave.txt'), 'w') as f:
        for sap in productos_clave:
            f.write(f"{sap}\n")
    print(f"\n>>> Lista de productos clave guardada en: {os.path.join(OUTPUT_DIR, 'productos_clave.txt')}")

    return productos_clave

# --- Bloque Principal de Ejecución ---

def main():
    """
    Función principal que orquesta la ejecución del script de EDA.
    """
    # Paso 1: Cargar y preprocesar los datos.
    df_processed = cargar_y_preprocesar_datos(DATA_PATH)

    # Paso 2: Generar visualizaciones anuales.
    visualizar_consumo_por_año(df_processed)

    # Paso 3: Identificar y guardar los productos más importantes.
    identificar_productos_clave(df_processed)

    print("\n>>> Análisis exploratorio actualizado completado.")

if __name__ == "__main__":
    main()
