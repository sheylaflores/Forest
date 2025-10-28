# -*- coding: utf-8 -*-
"""
Script para generar predicciones de demanda individuales para productos clave,
excluyendo marcas genéricas y de materiales varios.
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
CONSUMO_DATA_PATH = 'data/kardexASTEC_filtrado.xlsx'
SAP_INFO_PATH = 'data/14.06 MP_ARTICULOS_SAP.xlsx'
PRODUCTOS_CLAVE_PATH = 'output/productos_clave.txt'
OUTPUT_DIR = 'output'
PREDICCIONES_FINALES_CSV = os.path.join(OUTPUT_DIR, 'predicciones_por_producto.csv')

# --- Funciones de Preparación de Datos ---

def cargar_productos_clave(path):
    """Carga la lista de SAP de productos clave (ya filtrada)."""
    try:
        with open(path, 'r') as f:
            productos = [line.strip() for line in f]
        print(f">>> {len(productos)} productos clave (filtrados) cargados para la predicción.")
        return productos
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de productos clave en {path}. Ejecuta primero el script de análisis.")
        return []

def cargar_y_filtrar_datos_consumo(consumo_path, sap_info_path):
    """
    Carga los datos de consumo y los filtra para excluir marcas no deseadas,
    y une la información de descripción del producto.
    """
    # 1. Cargar información de SAP para obtener marcas y descripciones
    try:
        df_sap = pd.read_excel(sap_info_path, usecols=['Codigo_SAP', 'Marca', 'Descripcion'])
        df_sap.rename(columns={'Codigo_SAP': 'sap'}, inplace=True)
        marcas_a_excluir = ['MATERIALES VARIOS', 'GENÉRICO']
        saps_a_excluir = df_sap[df_sap['Marca'].isin(marcas_a_excluir)]['sap'].unique().tolist()
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de información de SAP en {sap_info_path}")
        return None

    # 2. Cargar datos de consumo
    try:
        df_consumo = pd.read_excel(consumo_path, usecols=['SAP', 'month_year', 'Consumo Total'])
        df_consumo.rename(columns={'month_year': 'fecha', 'Consumo Total': 'consumo', 'SAP': 'sap'}, inplace=True)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de consumo en {consumo_path}")
        return None

    # 3. Unir la descripción del producto al DataFrame de consumo
    df_consumo = pd.merge(df_consumo, df_sap[['sap', 'Descripcion']], on='sap', how='left')
    df_consumo.rename(columns={'Descripcion': 'descripcion'}, inplace=True)
    df_consumo['descripcion'].fillna('N/A', inplace=True)

    # 4. Aplicar filtros
    df_consumo['fecha'] = pd.to_datetime(df_consumo['fecha'])
    df_consumo = df_consumo[df_consumo['fecha'] >= '2022-01-01']
    df_consumo = df_consumo[~df_consumo['sap'].isin(saps_a_excluir)]
    df_consumo = df_consumo.set_index('fecha')

    print(">>> Datos de consumo cargados, unidos con descripción y filtrados correctamente.")
    return df_consumo

def preparar_serie_producto(df_full, sap_producto):
    """Prepara la serie temporal mensual para un único producto."""
    df_producto = df_full[df_full['sap'] == sap_producto].copy()
    consumo_mensual = df_producto['consumo'].resample('ME').sum()
    start_date, end_date = df_full.index.min(), df_full.index.max()
    full_range = pd.date_range(start=start_date, end=end_date, freq='ME')
    consumo_mensual = consumo_mensual.reindex(full_range, fill_value=0)
    return consumo_mensual

# --- Funciones de Modelado y Evaluación (sin cambios) ---
def calcular_metricas(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def crear_features_temporales(series):
    df = pd.DataFrame({'valor': series})
    df['mes'] = df.index.month
    df['año'] = df.index.year
    df['trimestre'] = df.index.quarter
    X, y = df.drop('valor', axis=1), df['valor']
    return X, y

def evaluar_modelos(train_data, test_data):
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

def generar_prediccion_final(serie_completa, nombre_modelo, redondear_a_entero, n_meses=12):
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
    else:
        prediccion = pd.Series([0]*n_meses, index=fechas_futuras)
    prediccion = prediccion.clip(lower=0)
    if redondear_a_entero:
        return prediccion.round(0).astype(int)
    else:
        return prediccion.round(2)

# --- Bloque Principal ---

def main():
    """Orquesta el proceso de predicción con datos filtrados."""

    df_full = cargar_y_filtrar_datos_consumo(CONSUMO_DATA_PATH, SAP_INFO_PATH)
    if df_full is None:
        return

    sap_a_descripcion = df_full[['sap', 'descripcion']].drop_duplicates().set_index('sap').to_dict()['descripcion']

    productos_clave = cargar_productos_clave(PRODUCTOS_CLAVE_PATH)
    if not productos_clave:
        return

    lista_predicciones = []

    print("\n--- Iniciando proceso de predicción para productos clave (filtrados) ---")
    for producto in tqdm(productos_clave, desc="Procesando productos"):
        serie_producto = preparar_serie_producto(df_full, producto)

        if len(serie_producto.dropna()) < 24:
            continue

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
                'descripcion': sap_a_descripcion.get(producto, 'N/A'),
                'fecha_prediccion': fecha,
                'demanda_predicha': valor_predicho,
                'mejor_modelo': mejor_modelo_nombre,
                'modelo_rmse': mejores_metricas['RMSE']
            })

    if lista_predicciones:
        df_predicciones_finales = pd.DataFrame(lista_predicciones)
        column_order = ['sap', 'descripcion', 'fecha_prediccion', 'demanda_predicha', 'mejor_modelo', 'modelo_rmse']
        df_predicciones_finales = df_predicciones_finales[column_order]
        df_predicciones_finales.to_csv(PREDICCIONES_FINALES_CSV, index=False)
        print(f"\n>>> Todas las predicciones (filtradas) han sido guardadas en: {PREDICCIONES_FINALES_CSV}")
    else:
        print("\nNo se generaron predicciones. Revisa los datos de entrada.")

if __name__ == "__main__":
    main()
