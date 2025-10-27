# -*- coding: utf-8 -*-
"""
Script para el Análisis Exploratorio de Datos (EDA) de la demanda de productos.

Este script carga los datos históricos de consumo, realiza una limpieza y
preprocesamiento inicial, y genera visualizaciones clave para entender las
tendencias y patrones de la demanda.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuración de Estilo para Gráficos ---
# Utiliza un estilo profesional y legible para las visualizaciones.
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# --- Carga de Datos ---
# Define la ruta al archivo de datos. Es una buena práctica para la reproducibilidad.
DATA_PATH = 'data/kardexASTEC_filtrado.xlsx'
OUTPUT_DIR = 'output'

# Crea el directorio de salida si no existe.
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def cargar_datos(path):
    """
    Carga los datos desde un archivo Excel.

    Args:
        path (str): Ruta al archivo .xlsx.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
        Retorna None si el archivo no se encuentra.
    """
    try:
        # Lee el archivo Excel y lo convierte en un DataFrame de pandas.
        df = pd.read_excel(path)
        print(">>> Datos cargados exitosamente.")
        return df
    except FileNotFoundError:
        # Manejo de error en caso de que el archivo no exista.
        print(f"Error: El archivo no se encontró en la ruta: {path}")
        return None

# --- Funciones de Análisis y Visualización ---

def analisis_exploratorio(df):
    """
    Realiza un análisis exploratorio básico de los datos.

    Args:
        df (pd.DataFrame): DataFrame con los datos de consumo.
    """
    if df is None:
        return

    # Muestra las primeras 5 filas para una vista previa de los datos.
    print("\n--- Vista Previa de los Datos ---")
    print(df.head())

    # Proporciona un resumen conciso del DataFrame.
    print("\n--- Información General del DataFrame ---")
    df.info()

    # Muestra estadísticas descriptivas para las columnas numéricas.
    print("\n--- Estadísticas Descriptivas ---")
    # Se seleccionan solo las columnas numéricas para el análisis.
    print(df.describe(include='number'))

def preprocesamiento_datos(df):
    """
    Limpia y preprocesa los datos para el análisis de series temporales.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame preprocesado y listo para el análisis.
    """
    if df is None:
        return None

    print("\n--- Preprocesamiento de Datos ---")
    # Se crea una copia para evitar advertencias de 'SettingWithCopyWarning'.
    df_copy = df.copy()

    # Renombrar columnas a nombres más manejables y consistentes.
    df_copy.rename(columns={
        'month_year': 'fecha',
        'Consumo Total': 'consumo',
        'SAP': 'sap'
    }, inplace=True)
    print("Columnas renombradas para mayor claridad.")

    # Convierte la columna 'fecha' a formato de fecha y hora.
    # `errors='coerce'` convierte las fechas inválidas en NaT (Not a Time).
    df_copy['fecha'] = pd.to_datetime(df_copy['fecha'], errors='coerce')
    print("Columna 'fecha' convertida a tipo datetime.")

    # Elimina filas donde la fecha no pudo ser convertida (si las hay).
    df_copy.dropna(subset=['fecha'], inplace=True)
    print("Filas con fechas inválidas eliminadas.")

    # Asegura que 'consumo' y 'sap' no tengan valores nulos que afecten los cálculos.
    df_copy.dropna(subset=['consumo', 'sap'], inplace=True)
    print("Filas con valores nulos en 'consumo' o 'sap' eliminadas.")

    # Establece la 'fecha' como el índice del DataFrame, crucial para series temporales.
    df_copy.set_index('fecha', inplace=True)

    # Ordena los datos por fecha para asegurar la secuencia temporal correcta.
    df_copy.sort_index(inplace=True)
    print("DataFrame indexado y ordenado por fecha.")

    return df_copy

def visualizar_tendencia_mensual(df):
    """
    Calcula y visualiza la tendencia de consumo total mensual.

    Args:
        df (pd.DataFrame): DataFrame preprocesado.
    """
    if df is None:
        return

    # Agrupa los datos por mes y suma el consumo. 'M' indica frecuencia mensual.
    consumo_mensual = df['consumo'].resample('M').sum()

    print("\n--- Estadísticas del Consumo Mensual ---")
    print(consumo_mensual.describe())

    # Creación del gráfico de tendencia mensual.
    plt.figure()
    consumo_mensual.plot(title='Tendencia Mensual de Consumo Total', color='teal')
    plt.xlabel('Fecha')
    plt.ylabel('Consumo Total')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Guarda el gráfico en el directorio de salida.
    plot_path = os.path.join(OUTPUT_DIR, 'tendencia_consumo_mensual.png')
    plt.savefig(plot_path)
    print(f"\n>>> Gráfico de tendencia mensual guardado en: {plot_path}")
    # plt.show() # Descomentar si se ejecuta localmente y se desea ver el gráfico.

def visualizar_top_productos(df, top_n=10):
    """
    Visualiza el consumo de los N productos principales.

    Args:
        df (pd.DataFrame): DataFrame preprocesado.
        top_n (int): Número de productos a mostrar en el ranking.
    """
    if df is None:
        return

    # Agrupa por 'sap' (producto) y suma el consumo para encontrar los más vendidos.
    consumo_por_producto = df.groupby('sap')['consumo'].sum().sort_values(ascending=False)

    # Selecciona los 'top_n' productos con mayor consumo.
    top_productos = consumo_por_producto.head(top_n)

    print(f"\n--- Top {top_n} Productos con Mayor Consumo ---")
    print(top_productos)

    # Creación del gráfico de barras para los productos principales.
    plt.figure()
    top_productos.plot(kind='bar', color=sns.color_palette('viridis', top_n))
    plt.title(f'Consumo por los {top_n} Productos Principales (SAP)')
    plt.xlabel('Código SAP del Producto')
    plt.ylabel('Consumo Total')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # Ajusta el gráfico para que no se corten las etiquetas.

    # Guarda el gráfico.
    plot_path = os.path.join(OUTPUT_DIR, 'top_productos_consumo.png')
    plt.savefig(plot_path)
    print(f">>> Gráfico de top productos guardado en: {plot_path}")
    # plt.show() # Descomentar para visualización interactiva.

# --- Bloque Principal de Ejecución ---

def main():
    """
    Función principal que orquesta la ejecución del script de EDA.
    """
    # Paso 1: Cargar los datos.
    df_raw = cargar_datos(DATA_PATH)

    # Paso 2: Realizar el análisis exploratorio inicial.
    analisis_exploratorio(df_raw)

    # Paso 3: Preprocesar los datos.
    df_processed = preprocesamiento_datos(df_raw)

    # Paso 4: Generar visualizaciones.
    if df_processed is not None:
        visualizar_tendencia_mensual(df_processed)
        visualizar_top_productos(df_processed)
        print("\n>>> Análisis exploratorio completado. Los gráficos se han guardado en la carpeta 'output'.")

if __name__ == "__main__":
    # Esta construcción asegura que el script solo se ejecute cuando es llamado directamente.
    main()
