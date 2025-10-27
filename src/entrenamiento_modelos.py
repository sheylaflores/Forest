# -*- coding: utf-8 -*-
"""
Script para entrenar, evaluar y comparar modelos de predicción de series temporales.

Este script realiza los siguientes pasos:
1. Carga y preprocesa los datos para crear una serie temporal de consumo mensual.
2. Divide los datos en conjuntos de entrenamiento (80%) y prueba (20%).
3. Implementa y entrena varios modelos: ARIMA, SARIMA, XGBoost, RandomForest y Prophet.
4. Evalúa el rendimiento de cada modelo utilizando métricas MAE, RMSE y MAPE.
5. Selecciona automáticamente el modelo con el menor error (RMSE) como el "mejor modelo".
6. Guarda los resultados de la evaluación en un archivo CSV.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings

# --- Ignorar Advertencias Comunes ---
# Ignora las advertencias de convergencia de los modelos estadísticos para mantener la salida limpia.
warnings.filterwarnings("ignore")

# --- Constantes y Rutas ---
DATA_PATH = 'data/kardexASTEC_filtrado.xlsx'
OUTPUT_DIR = 'output'
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'evaluacion_modelos.csv')

# --- Carga y Preparación de Datos ---

def preparar_datos(path):
    """
    Carga y prepara los datos para el modelado de series temporales.

    Args:
        path (str): Ruta al archivo de datos .xlsx.

    Returns:
        pd.Series: Serie temporal del consumo mensual.
    """
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        print(f"Error: El archivo no se encontró en la ruta: {path}")
        return None

    # Renombrar columnas para consistencia.
    df.rename(columns={
        'month_year': 'fecha',
        'Consumo Total': 'consumo'
    }, inplace=True)

    # Convertir a datetime y manejar errores.
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df.dropna(subset=['fecha', 'consumo'], inplace=True)

    # Agregar el consumo a nivel mensual.
    df.set_index('fecha', inplace=True)
    consumo_mensual = df['consumo'].resample('ME').sum()

    # Asegurarse de que no haya valores nulos o infinitos.
    consumo_mensual.replace([np.inf, -np.inf], np.nan, inplace=True)
    consumo_mensual.fillna(0, inplace=True)

    print(">>> Datos cargados y agregados mensualmente.")
    return consumo_mensual

# --- Funciones de Evaluación ---

def calcular_metricas(y_true, y_pred):
    """
    Calcula las métricas de evaluación MAE, RMSE y MAPE.

    Args:
        y_true (pd.Series): Valores reales.
        y_pred (pd.Series): Valores predichos.

    Returns:
        dict: Un diccionario con las métricas calculadas.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE se calcula con cuidado para evitar divisiones por cero.
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# --- Implementación de Modelos ---

def entrenar_evaluar_arima(train_data, test_data):
    """Entrena y evalúa un modelo ARIMA."""
    print("Entrenando modelo ARIMA...")
    # El orden (p,d,q) (5,1,0) es un punto de partida común.
    model = ARIMA(train_data, order=(5, 1, 0))
    model_fit = model.fit()

    # Realiza predicciones sobre el conjunto de prueba.
    predicciones = model_fit.forecast(steps=len(test_data))

    return calcular_metricas(test_data, predicciones)

def entrenar_evaluar_sarima(train_data, test_data):
    """Entrena y evalúa un modelo SARIMA."""
    print("Entrenando modelo SARIMA...")
    # El orden estacional (P,D,Q,m) (1,1,1,12) considera la estacionalidad anual.
    model = SARIMAX(train_data, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    predicciones = model_fit.forecast(steps=len(test_data))

    return calcular_metricas(test_data, predicciones)

def crear_features_temporales(series):
    """Crea características basadas en la fecha para modelos de ML."""
    df = pd.DataFrame({'valor': series})
    df['mes'] = df.index.month
    df['año'] = df.index.year
    df['trimestre'] = df.index.quarter
    # Se podrían agregar más features como 'semana_del_año', etc.
    X = df.drop('valor', axis=1)
    y = df['valor']
    return X, y

def entrenar_evaluar_ml(modelo, train_data, test_data):
    """Función genérica para entrenar y evaluar modelos de Machine Learning."""
    X_train, y_train = crear_features_temporales(train_data)
    X_test, y_test = crear_features_temporales(test_data)

    print(f"Entrenando modelo {modelo.__class__.__name__}...")
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    return calcular_metricas(y_test, predicciones)

def entrenar_evaluar_prophet(train_data, test_data):
    """Entrena y evalúa un modelo Prophet."""
    print("Entrenando modelo Prophet...")
    # Prophet requiere un DataFrame con columnas 'ds' (fecha) y 'y' (valor).
    df_train = pd.DataFrame({'ds': train_data.index, 'y': train_data.values})

    model = Prophet()
    model.fit(df_train)

    # Crea un DataFrame futuro para las predicciones.
    df_future = pd.DataFrame({'ds': test_data.index})
    forecast = model.predict(df_future)
    predicciones = forecast['yhat'].values

    return calcular_metricas(test_data, predicciones)

# --- Bloque Principal de Ejecución ---

def main():
    """
    Orquesta el proceso de entrenamiento y evaluación de modelos.
    """
    # 1. Cargar y preparar los datos.
    consumo_mensual = preparar_datos(DATA_PATH)
    if consumo_mensual is None:
        return

    # 2. Dividir los datos en entrenamiento y prueba.
    train_size = int(len(consumo_mensual) * 0.8)
    train, test = consumo_mensual[0:train_size], consumo_mensual[train_size:]
    print(f"Datos divididos: {len(train)} para entrenamiento, {len(test)} para prueba.")

    # 3. Entrenar y evaluar los modelos.
    resultados = {}

    resultados['ARIMA'] = entrenar_evaluar_arima(train, test)
    resultados['SARIMA'] = entrenar_evaluar_sarima(train, test)

    # Modelos de Machine Learning.
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    resultados['RandomForest'] = entrenar_evaluar_ml(rf, train, test)

    xgb = XGBRegressor(n_estimators=100, random_state=42)
    resultados['XGBoost'] = entrenar_evaluar_ml(xgb, train, test)

    resultados['Prophet'] = entrenar_evaluar_prophet(train, test)

    # 4. Procesar y mostrar los resultados.
    df_resultados = pd.DataFrame(resultados).T
    print("\n--- Resultados de la Evaluación de Modelos ---")
    print(df_resultados)

    # 5. Seleccionar el mejor modelo basado en RMSE.
    mejor_modelo_nombre = df_resultados['RMSE'].idxmin()
    print(f"\n>>> Mejor modelo seleccionado: {mejor_modelo_nombre} (RMSE: {df_resultados.loc[mejor_modelo_nombre, 'RMSE']:.2f})")

    # 6. Guardar los resultados en un archivo CSV.
    df_resultados.to_csv(RESULTS_FILE)
    print(f"\n>>> Resultados de la evaluación guardados en: {RESULTS_FILE}")

if __name__ == "__main__":
    main()
