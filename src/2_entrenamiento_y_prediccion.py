# -*- coding: utf-8 -*-
"""
Script 2: Entrenamiento de Modelos y Generación de Predicciones

Funciones:
1. Carga los datos limpios y la lista de productos clave del paso anterior.
2. Crea una variable exógena binaria para la temporada de pesca.
3. Itera a través de cada producto clave:
    a. Prepara la serie de tiempo del producto.
    b. Implementa la variable exógena en cinco modelos (ARIMA/SARIMA con `exog`,
       ML con features, Prophet con `add_regressor`).
    c. Realiza una competencia de modelos para seleccionar el mejor (basado en RMSE).
    d. Re-entrena el mejor modelo con todos los datos.
    e. Genera una predicción a 12 meses, utilizando los valores futuros de la variable exógena.
    f. Aplica post-procesamiento (ajuste a cero, redondeo inteligente).
4. Consolida y guarda todas las predicciones en un archivo CSV.
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

# --- Configuración ---
warnings.filterwarnings("ignore")
# (El resto de las rutas se definen en la función principal para claridad)

# --- Funciones de Preparación de Datos ---

def crear_variable_exogena(index):
    """Crea un DataFrame con la variable exógena 'temporada_pesca'."""
    df_exog = pd.DataFrame(index=index)
    # Temporada alta: Abril, Mayo, Junio, Noviembre, Diciembre (meses 4, 5, 6, 11, 12)
    df_exog['temporada_pesca'] = df_exog.index.month.isin([4, 5, 6, 11, 12]).astype(int)
    return df_exog

# --- Funciones de Modelado y Evaluación ---

def calcular_metricas(y_true, y_pred):
    """Calcula MAE, RMSE y MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def evaluar_modelos(train_data, test_data, train_exog, test_exog):
    """Entrena y evalúa todos los modelos para una serie de datos con variables exógenas."""
    resultados = {}

    # ARIMA con variable exógena
    try:
        model_arima = ARIMA(train_data, exog=train_exog, order=(5, 1, 0)).fit()
        pred_arima = model_arima.forecast(steps=len(test_data), exog=test_exog)
        resultados['ARIMA'] = calcular_metricas(test_data, pred_arima)
    except Exception:
        resultados['ARIMA'] = {'MAE': np.inf, 'RMSE': np.inf, 'MAPE': np.inf}

    # SARIMA con variable exógena
    try:
        model_sarima = SARIMAX(train_data, exog=train_exog, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        pred_sarima = model_sarima.forecast(steps=len(test_data), exog=test_exog)
        resultados['SARIMA'] = calcular_metricas(test_data, pred_sarima)
    except Exception:
        resultados['SARIMA'] = {'MAE': np.inf, 'RMSE': np.inf, 'MAPE': np.inf}

    # Modelos ML (RandomForest y XGBoost)
    X_train = train_exog.copy()
    X_train['mes'] = X_train.index.month
    X_train['año'] = X_train.index.year

    X_test = test_exog.copy()
    X_test['mes'] = X_test.index.month
    X_test['año'] = X_test.index.year

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, train_data)
    pred_rf = rf.predict(X_test)
    resultados['RandomForest'] = calcular_metricas(test_data, pred_rf)

    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, train_data)
    pred_xgb = xgb.predict(X_test)
    resultados['XGBoost'] = calcular_metricas(test_data, pred_xgb)

    # Prophet con variable exógena (regressor)
    try:
        df_prophet_train = pd.DataFrame({'ds': train_data.index, 'y': train_data.values})
        df_prophet_train['temporada_pesca'] = train_exog['temporada_pesca'].values

        model_prophet = Prophet()
        model_prophet.add_regressor('temporada_pesca')
        model_prophet.fit(df_prophet_train)

        future = pd.DataFrame({'ds': test_data.index})
        future['temporada_pesca'] = test_exog['temporada_pesca'].values

        forecast = model_prophet.predict(future)
        pred_prophet = forecast['yhat'].values
        resultados['Prophet'] = calcular_metricas(test_data, pred_prophet)
    except Exception:
        resultados['Prophet'] = {'MAE': np.inf, 'RMSE': np.inf, 'MAPE': np.inf}

    return pd.DataFrame(resultados).T

# --- Función de Predicción Final ---

def generar_prediccion_final(serie_completa, exog_completa, nombre_modelo, redondear_a_entero, n_meses=12):
    """Re-entrena el mejor modelo y genera la predicción a 12 meses."""

    # Preparar datos futuros
    fechas_futuras = pd.date_range(start=serie_completa.index.max() + pd.DateOffset(months=1), periods=n_meses, freq='ME')
    exog_futuro = crear_variable_exogena(fechas_futuras)

    # Lógica de predicción por modelo
    if nombre_modelo == 'ARIMA':
        model = ARIMA(serie_completa, exog=exog_completa, order=(5, 1, 0)).fit()
        prediccion = model.forecast(steps=n_meses, exog=exog_futuro)
    elif nombre_modelo == 'SARIMA':
        model = SARIMAX(serie_completa, exog=exog_completa, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        prediccion = model.forecast(steps=n_meses, exog=exog_futuro)
    elif nombre_modelo in ['RandomForest', 'XGBoost']:
        X_train = exog_completa.copy()
        X_train['mes'] = X_train.index.month
        X_train['año'] = X_train.index.year

        X_future = exog_futuro.copy()
        X_future['mes'] = X_future.index.month
        X_future['año'] = X_future.index.year

        modelo = RandomForestRegressor(n_estimators=100, random_state=42) if nombre_modelo == 'RandomForest' else XGBRegressor(n_estimators=100, random_state=42)
        modelo.fit(X_train, serie_completa)
        prediccion = pd.Series(modelo.predict(X_future), index=fechas_futuras)
    elif nombre_modelo == 'Prophet':
        df_prophet = pd.DataFrame({'ds': serie_completa.index, 'y': serie_completa.values})
        df_prophet['temporada_pesca'] = exog_completa['temporada_pesca'].values

        model = Prophet()
        model.add_regressor('temporada_pesca')
        model.fit(df_prophet)

        future = pd.DataFrame({'ds': fechas_futuras})
        future['temporada_pesca'] = exog_futuro['temporada_pesca'].values
        forecast = model.predict(future)
        prediccion = pd.Series(forecast['yhat'].values, index=fechas_futuras)
    else:
        prediccion = pd.Series([0]*n_meses, index=fechas_futuras)

    # Post-procesamiento
    prediccion = prediccion.clip(lower=0)
    if redondear_a_entero:
        return prediccion.round(0).astype(int)
    else:
        return prediccion.round(2)

# --- Bloque Principal de Ejecución ---

def main():
    # Definir rutas
    CONSUMO_DATA_PATH = 'data/kardexASTEC_filtrado.xlsx'
    SAP_INFO_PATH = 'data/14.06 MP_ARTICULOS_SAP.xlsx'
    PRODUCTOS_CLAVE_PATH = 'output/productos_clave.txt'
    OUTPUT_DIR = 'output'
    PREDICCIONES_FINALES_CSV = os.path.join(OUTPUT_DIR, 'predicciones_por_producto.csv')

    # 1. Cargar datos
    df_sap = pd.read_excel(SAP_INFO_PATH, usecols=['Codigo_SAP', 'Descripcion'])
    df_sap.rename(columns={'Codigo_SAP': 'sap'}, inplace=True)
    sap_a_descripcion = df_sap.set_index('sap')['Descripcion'].to_dict()

    df_consumo = pd.read_excel(CONSUMO_DATA_PATH, usecols=['SAP', 'month_year', 'Consumo Total'])
    df_consumo.rename(columns={'month_year': 'fecha', 'Consumo Total': 'consumo', 'SAP': 'sap'}, inplace=True)

    with open(os.path.join(OUTPUT_DIR, 'productos_excluidos.txt'), 'r') as f:
        saps_a_excluir = [line.strip() for line in f]

    df_consumo = df_consumo[~df_consumo['sap'].isin(saps_a_excluir)]
    df_consumo['fecha'] = pd.to_datetime(df_consumo['fecha'])
    df_consumo = df_consumo[df_consumo['fecha'] >= '2022-01-01'].set_index('fecha')

    productos_clave = [line.strip() for line in open(PRODUCTOS_CLAVE_PATH, 'r')]

    # 2. Bucle de predicción
    lista_predicciones = []
    print("\n--- Iniciando proceso de predicción con variable exógena ---")

    for producto in tqdm(productos_clave, desc="Procesando productos"):
        # Preparar datos del producto
        serie_producto = df_consumo[df_consumo['sap'] == producto]['consumo'].resample('ME').sum()
        start_date, end_date = df_consumo.index.min(), df_consumo.index.max()
        full_range = pd.date_range(start=start_date, end=end_date, freq='ME')
        serie_producto = serie_producto.reindex(full_range, fill_value=0)

        if len(serie_producto) < 24: continue

        # Redondeo inteligente
        consumo_historico = df_consumo[df_consumo['sap'] == producto]['consumo']
        redondear_a_entero = (consumo_historico % 1 == 0).all()

        # Preparar variable exógena
        df_exog = crear_variable_exogena(serie_producto.index)

        # División de datos (Train/Test)
        train_size = int(len(serie_producto) * 0.8)
        train_data, test_data = serie_producto[:train_size], serie_producto[train_size:]
        train_exog, test_exog = df_exog[:train_size], df_exog[train_size:]

        # Evaluar modelos
        df_resultados = evaluar_modelos(train_data, test_data, train_exog, test_exog)
        mejor_modelo_nombre = df_resultados['RMSE'].idxmin()
        mejores_metricas = df_resultados.loc[mejor_modelo_nombre]

        # Generar predicción final
        prediccion_final = generar_prediccion_final(serie_producto, df_exog, mejor_modelo_nombre, redondear_a_entero)

        for fecha, valor_predicho in prediccion_final.items():
            lista_predicciones.append({
                'sap': producto,
                'descripcion': sap_a_descripcion.get(producto, 'N/A'),
                'fecha_prediccion': fecha,
                'demanda_predicha': valor_predicho,
                'mejor_modelo': mejor_modelo_nombre,
                'modelo_rmse': mejores_metricas['RMSE']
            })

    # 3. Guardar resultados
    if lista_predicciones:
        df_predicciones = pd.DataFrame(lista_predicciones)
        df_predicciones.to_csv(PREDICCIONES_FINALES_CSV, index=False)
        print(f"\n>>> Predicciones con variable exógena guardadas en: {PREDICCIONES_FINALES_CSV}")

if __name__ == "__main__":
    main()
