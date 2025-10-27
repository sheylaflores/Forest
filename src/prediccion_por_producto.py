# -*- coding: utf-8 -*-
"""
Script para generar predicciones de demanda individuales para productos clave.

Flujo de trabajo:
1. Carga la lista de productos clave y las descripciones de los productos.
2. Para cada producto en la lista:
   a. Prepara su serie temporal de consumo mensual desde 2022.
   b. Analiza si el producto históricamente usa unidades enteras o decimales.
   c. Entrena y evalúa cinco modelos diferentes para encontrar el mejor.
   d. Re-entrena el mejor modelo y genera una predicción a 12 meses.
   e. Aplica un "redondeo inteligente" al resultado (entero o decimal).
3. Consolida todas las predicciones, incluyendo la descripción del producto, en un único archivo CSV.
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
    consumo_mensual = df_producto['consumo'].resample('ME').sum()
    start_date, end_date = df_full.index.min(), df_full.index.max()
    full_range = pd.date_range(start=start_date, end=end_date, freq='ME')
    consumo_mensual = consumo_mensual.reindex(full_range, fill_value=0)
    return consumo_mensual

# --- Funciones de Modelado y Evaluación ---

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
    try:
        model_arima = ARIMA(train_data, order=(5, 1, 0)).fit()
        pred_arima = model_arima.forecast(steps=len(test_data))
        resultados['ARIMA'] = calcular_metricas(test_data, pred_arima)
    except Exception:
        resultados['ARIMA'] = {'MAE': np.inf, 'RMSE': np.inf, 'MAPE': np.inf}

    try:
        model_sarima = SARIMAX(train_data, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        pred_sarima = model_sarima.forecast(steps=len(test_data))
        resultados['SARIMA'] = calcular_metricas(test_data, pred_sarima)
    except Exception:
        resultados['SARIMA'] = {'MAE': np.inf, 'RMSE': np.inf, 'MAPE': np.inf}

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

# --- Función de Predicción Final con Lógica de Redondeo ---

def generar_prediccion_final(serie_completa, nombre_modelo, redondear_a_entero, n_meses=12):
    """
    Re-entrena el mejor modelo, genera la predicción y aplica el redondeo inteligente.
    """
    fechas_futuras = pd.date_range(start=serie_completa.index.max() + pd.DateOffset(months=1), periods=n_meses, freq='ME')

    # Generar predicción según el modelo
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
    else:
        prediccion = pd.Series([0]*n_meses, index=fechas_futuras)

    # Corregir predicciones negativas.
    prediccion = prediccion.clip(lower=0)

    # Aplicar redondeo inteligente.
    if redondear_a_entero:
        return prediccion.round(0).astype(int)
    else:
        return prediccion.round(2)

# --- Bloque Principal de Ejecución ---

def main():
    """Orquesta el proceso completo de predicción por producto."""

    # 1. Cargar datos completos, incluyendo la descripción.
    df_full = pd.read_excel(DATA_PATH, usecols=['SAP', 'Descripción del artículo', 'month_year', 'Consumo Total'])
    df_full.rename(columns={
        'month_year': 'fecha',
        'Consumo Total': 'consumo',
        'SAP': 'sap',
        'Descripción del artículo': 'descripcion'
    }, inplace=True)
    df_full['fecha'] = pd.to_datetime(df_full['fecha'])
    df_full = df_full[df_full['fecha'] >= '2022-01-01'].set_index('fecha')

    # Crear mapa de SAP a descripción.
    sap_a_descripcion = df_full[['sap', 'descripcion']].drop_duplicates().set_index('sap').to_dict()['descripcion']

    productos_clave = cargar_productos_clave(PRODUCTOS_CLAVE_PATH)
    if not productos_clave:
        return

    # 2. Bucle de predicción.
    lista_predicciones = []

    print("\n--- Iniciando proceso de predicción para productos clave ---")
    for producto in tqdm(productos_clave, desc="Procesando productos"):
        serie_producto = preparar_serie_producto(df_full, producto)

        if len(serie_producto.dropna()) < 24:
            continue

        # Lógica de redondeo inteligente: verificar si el consumo histórico siempre es entero.
        consumo_historico_producto = df_full[df_full['sap'] == producto]['consumo']
        redondear_a_entero = (consumo_historico_producto % 1 == 0).all()

        train_size = int(len(serie_producto) * 0.8)
        train, test = serie_producto[:train_size], serie_producto[train_size:]

        df_resultados = evaluar_modelos(train, test)
        mejor_modelo_nombre = df_resultados['RMSE'].idxmin()
        mejores_metricas = df_resultados.loc[mejor_modelo_nombre]

        prediccion_final = generar_prediccion_final(serie_producto, mejor_modelo_nombre, redondear_a_entero)

        for fecha, valor_predicho in prediccion_final.items():
            lista_predicciones.append({
                'sap': producto,
                'descripcion': sap_a_descripcion.get(producto, 'N/A'), # Añadir descripción
                'fecha_prediccion': fecha,
                'demanda_predicha': valor_predicho,
                'mejor_modelo': mejor_modelo_nombre,
                'modelo_rmse': mejores_metricas['RMSE']
            })

    # 3. Consolidar y guardar resultados.
    if lista_predicciones:
        df_predicciones_finales = pd.DataFrame(lista_predicciones)
        # Reordenar columnas para mayor claridad
        column_order = ['sap', 'descripcion', 'fecha_prediccion', 'demanda_predicha', 'mejor_modelo', 'modelo_rmse']
        df_predicciones_finales = df_predicciones_finales[column_order]

        df_predicciones_finales.to_csv(PREDICCIONES_FINALES_CSV, index=False)
        print(f"\n>>> Todas las predicciones han sido guardadas en: {PREDICCIONES_FINALES_CSV}")
    else:
        print("\nNo se generaron predicciones. Revisa los datos de entrada.")

if __name__ == "__main__":
    main()
