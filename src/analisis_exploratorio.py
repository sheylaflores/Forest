# -*- coding: utf-8 -*-
"""
Script para el Análisis Exploratorio de Datos (EDA) de la demanda de productos.

Este script carga los datos históricos, los filtra para el período 2022-2025,
excluye productos de marcas genéricas, genera visualizaciones anuales,
e identifica los productos clave para la planificación de inventario.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Configuración ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# --- Rutas de Archivos ---
CONSUMO_DATA_PATH = 'data/kardexASTEC_filtrado.xlsx'
SAP_INFO_PATH = 'data/14.06 MP_ARTICULOS_SAP.xlsx'
OUTPUT_DIR = 'output'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def cargar_y_filtrar_datos(consumo_path, sap_info_path):
    """
    Carga los datos de consumo y la información de SAP, filtra las marcas
    no deseadas ("MATERIALES VARIOS", "GENÉRICO") y preprocesa el DataFrame.

    Returns:
        pd.DataFrame: DataFrame preprocesado y filtrado, listo para el análisis.
    """
    # 1. Cargar la información de SAP para obtener las marcas
    try:
        df_sap = pd.read_excel(sap_info_path, usecols=['Codigo_SAP', 'Marca'])
        df_sap.rename(columns={'Codigo_SAP': 'sap'}, inplace=True)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de información de SAP en: {sap_info_path}")
        return None
    except Exception as e:
        print(f"Error al leer el archivo SAP: {e}")
        return None

    # 2. Identificar los SAP a excluir
    marcas_a_excluir = ['MATERIALES VARIOS', 'GENÉRICO']
    saps_a_excluir = df_sap[df_sap['Marca'].isin(marcas_a_excluir)]['sap'].unique().tolist()
    print(f">>> Se han identificado {len(saps_a_excluir)} productos de marcas a excluir.")

    # 3. Cargar los datos de consumo
    try:
        df_consumo = pd.read_excel(consumo_path)
        print(">>> Datos de consumo cargados exitosamente.")
    except FileNotFoundError:
        print(f"Error: El archivo de consumo no se encontró en la ruta: {consumo_path}")
        return None

    # 4. Preprocesar y filtrar el DataFrame de consumo
    df_consumo.rename(columns={
        'month_year': 'fecha',
        'Consumo Total': 'consumo',
        'SAP': 'sap'
    }, inplace=True)

    df_consumo['fecha'] = pd.to_datetime(df_consumo['fecha'], errors='coerce')
    df_consumo.dropna(subset=['fecha', 'consumo', 'sap'], inplace=True)

    # Aplicar el filtro por fecha
    df_consumo = df_consumo[df_consumo['fecha'] >= '2022-01-01']
    print(f">>> Datos filtrados para el período de 2022 a {df_consumo['fecha'].max().year}.")

    # Aplicar el filtro por marca
    registros_antes = len(df_consumo)
    df_consumo = df_consumo[~df_consumo['sap'].isin(saps_a_excluir)]
    registros_despues = len(df_consumo)
    print(f">>> Se han filtrado {registros_antes - registros_despues} registros de consumo pertenecientes a las marcas excluidas.")

    df_consumo.set_index('fecha', inplace=True)
    df_consumo.sort_index(inplace=True)

    return df_consumo

def visualizar_consumo_por_año(df):
    """
    Genera y guarda un gráfico de consumo mensual para cada año en el DataFrame.
    """
    if df is None:
        return

    años = df.index.year.unique()
    print("\n--- Generando gráficos de consumo mensual por año (filtrado) ---")

    for año in años:
        df_año = df[df.index.year == año]
        consumo_mensual_año = df_año['consumo'].resample('ME').sum()
        idx = pd.date_range(f'01-01-{año}', f'12-31-{año}', freq='ME')
        consumo_mensual_año = consumo_mensual_año.reindex(idx, fill_value=0)

        plt.figure(figsize=(12, 6))
        consumo_mensual_año.plot(kind='bar', color=sns.color_palette('viridis', 12))
        plt.title(f'Consumo Mensual del Año {año} (Marcas Relevantes)')
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
    """
    if df is None:
        return []

    print("\n--- Identificando productos clave (Análisis ABC sobre datos filtrados) ---")

    consumo_por_producto = df.groupby('sap')['consumo'].sum().sort_values(ascending=False)
    consumo_acumulado = consumo_por_producto.cumsum()
    consumo_total = consumo_por_producto.sum()
    porcentaje_acumulado = (consumo_acumulado / consumo_total) * 100

    productos_clave = porcentaje_acumulado[porcentaje_acumulado <= 80].index.tolist()

    num_total_productos = len(consumo_por_producto)
    num_productos_clave = len(productos_clave)
    porcentaje_productos = (num_productos_clave / num_total_productos) * 100

    print(f">>> {num_productos_clave} de {num_total_productos} productos ({porcentaje_productos:.2f}%) representan el 80% del consumo total.")
    print(f"Productos clave identificados (filtrados): {num_productos_clave}")

    with open(os.path.join(OUTPUT_DIR, 'productos_clave.txt'), 'w') as f:
        for sap in productos_clave:
            f.write(f"{sap}\n")
    print(f"\n>>> Lista de productos clave (filtrada) guardada en: {os.path.join(OUTPUT_DIR, 'productos_clave.txt')}")

    return productos_clave

# --- Bloque Principal de Ejecución ---

def main():
    """
    Función principal que orquesta la ejecución del script de EDA.
    """
    df_processed = cargar_y_filtrar_datos(CONSUMO_DATA_PATH, SAP_INFO_PATH)
    visualizar_consumo_por_año(df_processed)
    identificar_productos_clave(df_processed)

    print("\n>>> Análisis exploratorio actualizado con filtro de marca completado.")

if __name__ == "__main__":
    main()
