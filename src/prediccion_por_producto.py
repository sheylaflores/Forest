# -*- coding: utf-8 -*-
"""
Script para generar predicciones de demanda individuales para productos clave.

Flujo de trabajo:
1. Carga la lista de productos clave identificados en el EDA.
2. Para cada producto en la lista:
   a. Prepara su serie temporal de consumo mensual desde 2022.
   b. Divide los datos en entrenamiento (80%) y prueba (20%).
   c. Entrena y evalúa cinco modelos diferentes (ARIMA, SARIMA, ML, Prophet).
   d. Selecciona el mejor modelo para ese producto basado en el menor RMSE.
   e. Re-entrena el mejor modelo con todos los datos disponibles para el producto.
   f. Genera una predicción de demanda para los próximos 12 meses.
3. Consolida todas las predicciones en un único archivo CSV.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings
from tqdm import tqdm

# --- Configuraciones ---
warnings.filterwarnings("ignore")
DATA_PATH = 'data/kardexASTEC_filtrado.xlsx'
PRODUCTOS_CLAVE_PATH = 'output/productos_clave.txt'
OUTPUT_DIR = 'output'
PREDICCIONES_FINALES_CSV = os.path.join(OUTPUT_DIR, 'predicciones_por_producto.csv')

# --- Funciones de Preparación de Datos ---

def cargar_productos_clave(path):
    """Carga la lista de SAP de productos clave."""
    try:
        with open(path, 'r') as f:
            productos = [line.strip() for line in f]
        print(f">>> {len(productos)} productos clave cargados.")
        return productos
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de productos clave en {path}")
        return []

def preparar_serie_producto(df_full, sap_producto):
    """Prepara la serie temporal mensual para un único producto."""
    df_producto = df_full[df_full['sap'] == sap_producto].copy()

    # Asegurar que la serie temporal tenga frecuencia mensual, rellenando ceros.
    consumo_mensual = df_producto['consumo'].resample('ME').sum()

    # Rellenar meses faltantes en el rango de fechas del producto.
    start_date, end_date = df_full.index.min(), df_full.index.max()
    full_range = pd.date_range(start=start_date, end=end_date, freq='ME')
    consumo_mensual = consumo_mensual.reindex(full_range, fill_value=0)

    return consumo_mensual

# --- Funciones de Modelado y Evaluación (Similares a las anteriores) ---

def calcular_metricas(y_true, y_pred):
    """Calcula MAE, RMSE y MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def crear_features_temporales(series):
    """Crea características de fecha para modelos de ML."""
    df = pd.DataFrame({'valor': series})
    df['mes'] = df.index.month
    df['año'] = df.index.year
    df['trimestre'] = df.index.quarter
    X, y = df.drop('valor', axis=1), df['valor']
    return X, y

def evaluar_modelos(train_data, test_data):
    """Entrena y evalúa todos los modelos para una serie de datos."""
    resultados = {}

    # ARIMA
    try:
        model_arima = ARIMA(train_data, order=(5, 1, 0)).fit()
        pred_arima = model_arima.forecast(steps=len(test_data))
        resultados['ARIMA'] = calcular_metricas(test_data, pred_arima)
    except Exception:
        resultados['ARIMA'] = {'MAE': np.inf, 'RMSE': np.inf, 'MAPE': np.inf}

    # SARIMA
    try:
        model_sarima = SARIMAX(train_data, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        pred_sarima = model_sarima.forecast(steps=len(test_data))
        resultados['SARIMA'] = calcular_metricas(test_data, pred_sarima)
    except Exception:
        resultados['SARIMA'] = {'MAE': np.inf, 'RMSE': np.inf, 'MAPE': np.inf}

    # Modelos ML
    X_train, y_train = crear_features_temporales(train_data)
    X_test, y_test = crear_features_temporales(test_data)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    resultados['RandomForest'] = calcular_metricas(y_test, pred_rf)

    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    resultados['XGBoost'] = calcular_metricas(y_test, pred_xgb)

    # Prophet
    try:
        df_prophet_train = pd.DataFrame({'ds': train_data.index, 'y': train_data.values})
        model_prophet = Prophet().fit(df_prophet_train)
        future = pd.DataFrame({'ds': test_data.index})
        forecast = model_prophet.predict(future)
        pred_prophet = forecast['yhat'].values
        resultados['Prophet'] = calcular_metricas(test_data, pred_prophet)
    except Exception:
        resultados['Prophet'] = {'MAE': np.inf, 'RMSE': np.inf, 'MAPE': np.inf}

    return pd.DataFrame(resultados).T

# --- Función de Predicción Final ---

def generar_prediccion_final(serie_completa, nombre_modelo, n_meses=12):
    """Re-entrena el mejor modelo y genera la predicción a 12 meses."""

    # Crear fechas futuras para la predicción.
    fechas_futuras = pd.date_range(start=serie_completa.index.max() + pd.DateOffset(months=1), periods=n_meses, freq='ME')

    if nombre_modelo == 'ARIMA':
        model = ARIMA(serie_completa, order=(5, 1, 0)).fit()
        prediccion = model.forecast(steps=n_meses)
        prediccion.index = fechas_futuras
    elif nombre_modelo == 'SARIMA':
        model = SARIMAX(serie_completa, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        prediccion = model.forecast(steps=n_meses)
        prediccion.index = fechas_futuras
    elif nombre_modelo in ['RandomForest', 'XGBoost']:
        X, y = crear_features_temporales(serie_completa)
        modelo = RandomForestRegressor(n_estimators=100, random_state=42) if nombre_modelo == 'RandomForest' else XGBRegressor(n_estimators=100, random_state=42)
        modelo.fit(X, y)

        X_futuro, _ = crear_features_temporales(pd.Series(index=fechas_futuras))
        prediccion = pd.Series(modelo.predict(X_futuro), index=fechas_futuras)
    elif nombre_modelo == 'Prophet':
        df_prophet = pd.DataFrame({'ds': serie_completa.index, 'y': serie_completa.values})
        model = Prophet().fit(df_prophet)
        future = pd.DataFrame({'ds': fechas_futuras})
        forecast = model.predict(future)
        prediccion = pd.Series(forecast['yhat'].values, index=fechas_futuras)
    else: # Fallback
        return pd.Series([0]*n_meses, index=fechas_futuras)

    # Corregir predicciones negativas, ajustándolas a cero.
    prediccion = prediccion.clip(lower=0)

    return prediccion.round(2) # Redondear a 2 decimales.


# --- Bloque Principal de Ejecución ---

def main():
    """Orquesta el proceso completo de predicción por producto."""

    # 1. Cargar datos y productos clave.
    df_full = pd.read_excel(DATA_PATH, usecols=['SAP', 'month_year', 'Consumo Total'])
    df_full.rename(columns={'month_year': 'fecha', 'Consumo Total': 'consumo', 'SAP': 'sap'}, inplace=True)
    df_full['fecha'] = pd.to_datetime(df_full['fecha'])
    df_full = df_full[df_full['fecha'] >= '2022-01-01'].set_index('fecha')

    productos_clave = cargar_productos_clave(PRODUCTOS_CLAVE_PATH)

    if not productos_clave:
        return

    # 2. Bucle de predicción por producto.
    lista_predicciones = []

    print("\n--- Iniciando proceso de predicción para productos clave ---")
    for producto in tqdm(productos_clave, desc="Procesando productos"):
        serie_producto = preparar_serie_producto(df_full, producto)

        # Omitir productos con muy pocos datos.
        if len(serie_producto.dropna()) < 24:
            continue

        # División de datos.
        train_size = int(len(serie_producto) * 0.8)
        train, test = serie_producto[:train_size], serie_producto[train_size:]

        # Evaluar modelos y seleccionar el mejor.
        df_resultados = evaluar_modelos(train, test)
        mejor_modelo_nombre = df_resultados['RMSE'].idxmin()
        mejores_metricas = df_resultados.loc[mejor_modelo_nombre]

        # Generar predicción final.
        prediccion_final = generar_prediccion_final(serie_producto, mejor_modelo_nombre)

        # Guardar resultados.
        for fecha, valor_predicho in prediccion_final.items():
            lista_predicciones.append({
                'sap': producto,
                'fecha_prediccion': fecha,
                'demanda_predicha': valor_predicho,
                'mejor_modelo': mejor_modelo_nombre,
                'modelo_rmse': mejores_metricas['RMSE']
            })

    # 3. Consolidar y guardar resultados.
    if lista_predicciones:
        df_predicciones_finales = pd.DataFrame(lista_predicciones)
        df_predicciones_finales.to_csv(PREDICCIONES_FINALES_CSV, index=False)
        print(f"\n>>> Todas las predicciones han sido guardadas en: {PREDICCIONES_FINALES_CSV}")
    else:
        print("\nNo se generaron predicciones. Revisa los datos de entrada.")

if __name__ == "__main__":
    main()
