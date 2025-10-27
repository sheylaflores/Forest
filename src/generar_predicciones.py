# -*- coding: utf-8 -*-
"""
Script para generar y visualizar la predicción de demanda a 12 meses.

Pasos que realiza el script:
1. Carga los datos históricos y los resultados de la evaluación de modelos.
2. Identifica el mejor modelo basado en el menor RMSE.
3. Re-entrena el mejor modelo (ARIMA en este caso) con todos los datos disponibles.
4. Genera una predicción para los próximos 12 meses.
5. Crea un gráfico que muestra los datos históricos y la predicción futura.
6. Exporta la predicción a un archivo CSV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import os
import warnings

# --- Ignorar Advertencias ---
warnings.filterwarnings("ignore")

# --- Configuración de Estilo y Rutas ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (18, 8)
DATA_PATH = 'data/kardexASTEC_filtrado.xlsx'
EVALUATION_PATH = 'output/evaluacion_modelos.csv'
OUTPUT_DIR = 'output'
PREDICTION_CSV_PATH = os.path.join(OUTPUT_DIR, 'prediccion_demanda_pesca.csv')
PREDICTION_PLOT_PATH = os.path.join(OUTPUT_DIR, 'prediccion_final_con_historico.png')

# --- Funciones de Preparación y Predicción ---

def preparar_datos_para_prediccion(path):
    """
    Carga y prepara la serie temporal de consumo mensual a partir de los datos brutos.

    Args:
        path (str): Ruta al archivo Excel.

    Returns:
        pd.Series: Serie temporal mensual del consumo.
    """
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        print(f"Error: El archivo no se encontró en {path}")
        return None

    df.rename(columns={'month_year': 'fecha', 'Consumo Total': 'consumo'}, inplace=True)
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df.dropna(subset=['fecha', 'consumo'], inplace=True)
    df.set_index('fecha', inplace=True)

    consumo_mensual = df['consumo'].resample('ME').sum()
    consumo_mensual.fillna(0, inplace=True)

    print(">>> Datos históricos preparados para la predicción.")
    return consumo_mensual

def obtener_mejor_modelo(evaluation_path):
    """
    Lee el archivo de evaluación y devuelve el nombre del mejor modelo.

    Args:
        evaluation_path (str): Ruta al CSV con la evaluación de modelos.

    Returns:
        str: Nombre del modelo con el menor RMSE.
    """
    try:
        df_eval = pd.read_csv(evaluation_path, index_col=0)
        mejor_modelo = df_eval['RMSE'].idxmin()
        print(f">>> Mejor modelo identificado: {mejor_modelo}")
        return mejor_modelo
    except FileNotFoundError:
        print(f"Error: Archivo de evaluación no encontrado en {evaluation_path}. Se usará ARIMA por defecto.")
        return 'ARIMA'

def generar_prediccion_arima(serie_temporal, n_meses=12):
    """
    Re-entrena el modelo ARIMA con todos los datos y genera una predicción a futuro.

    Args:
        serie_temporal (pd.Series): Serie temporal completa.
        n_meses (int): Número de meses a predecir.

    Returns:
        tuple: Contiene las predicciones, y los límites inferior y superior del intervalo de confianza.
    """
    print(">>> Re-entrenando el modelo ARIMA con todos los datos...")
    # Se utiliza el mismo orden (p,d,q) que en la evaluación.
    model = ARIMA(serie_temporal, order=(5, 1, 0))
    model_fit = model.fit()

    print(f">>> Generando predicción para los próximos {n_meses} meses...")
    # `get_forecast` proporciona predicciones y bandas de confianza.
    forecast = model_fit.get_forecast(steps=n_meses)

    prediccion_media = forecast.predicted_mean
    intervalos_confianza = forecast.conf_int()

    return prediccion_media, intervalos_confianza

# --- Función de Visualización y Exportación ---

def visualizar_y_exportar(serie_historica, prediccion, intervalos_confianza):
    """
    Crea un gráfico combinado de datos históricos y predicciones, y exporta los datos.

    Args:
        serie_historica (pd.Series): Datos históricos.
        prediccion (pd.Series): Predicciones futuras.
        intervalos_confianza (pd.DataFrame): Bandas de confianza.
    """
    # 1. Crear el DataFrame de predicción para exportar.
    df_prediccion = pd.DataFrame({
        'prediccion': prediccion,
        'limite_inferior': intervalos_confianza.iloc[:, 0],
        'limite_superior': intervalos_confianza.iloc[:, 1]
    })

    # 2. Exportar a CSV.
    df_prediccion.to_csv(PREDICTION_CSV_PATH)
    print(f"\n>>> Predicción exportada a: {PREDICTION_CSV_PATH}")
    print("\n--- Tabla de Predicción para los Próximos 12 Meses ---")
    print(df_prediccion)

    # 3. Crear el gráfico.
    plt.figure()

    # Dibuja la serie histórica.
    plt.plot(serie_historica.index, serie_historica.values, label='Consumo Histórico', color='teal')

    # Dibuja la predicción.
    plt.plot(prediccion.index, prediccion.values, label='Predicción de Demanda', color='crimson', linestyle='--')

    # Dibuja las bandas de confianza.
    plt.fill_between(prediccion.index,
                     intervalos_confianza.iloc[:, 0],
                     intervalos_confianza.iloc[:, 1],
                     color='crimson', alpha=0.2, label='Intervalo de Confianza (95%)')

    plt.title('Predicción de Demanda Mensual para los Próximos 12 Meses')
    plt.xlabel('Fecha')
    plt.ylabel('Consumo')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 4. Guardar el gráfico.
    plt.savefig(PREDICTION_PLOT_PATH)
    print(f"\n>>> Gráfico de predicción guardado en: {PREDICTION_PLOT_PATH}")
    # plt.show()

# --- Bloque Principal de Ejecución ---

def main():
    """
    Orquesta todo el proceso de generación de predicciones.
    """
    # 1. Preparar los datos.
    consumo_mensual = preparar_datos_para_prediccion(DATA_PATH)
    if consumo_mensual is None:
        return

    # 2. Identificar el mejor modelo (aunque aquí lo forzamos a ARIMA según el resultado).
    mejor_modelo_nombre = obtener_mejor_modelo(EVALUATION_PATH)

    # 3. Generar la predicción.
    # Por ahora, solo se implementa la lógica para ARIMA.
    if mejor_modelo_nombre == 'ARIMA':
        prediccion, intervalos_confianza = generar_prediccion_arima(consumo_mensual)
    else:
        # Aquí se podría añadir la lógica para otros modelos si fuera necesario.
        print(f"Lógica de predicción para el modelo '{mejor_modelo_nombre}' no implementada en este script. Usando ARIMA.")
        prediccion, intervalos_confianza = generar_prediccion_arima(consumo_mensual)

    # 4. Visualizar y exportar los resultados.
    visualizar_y_exportar(consumo_mensual, prediccion, intervalos_confianza)

if __name__ == "__main__":
    main()
